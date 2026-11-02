r"""The shared resume pipeline graph. Every workflow uses the same extraction agents and
harness; a workflow only supplies its own resume creator node.

    START -> resume_parser --\
                              +-> check_extractions --(errors)--> END
    START -> jd_retriever  --/            \--(ok)--> resume_creator -> END

LangGraph runs the two extraction agents in the same step and only moves on to
check_extractions once both have finished, so no orchestrator LLM is needed.
"""
from langgraph.graph import END, START, StateGraph

from common.extraction import make_jd_retriever, make_resume_parser
from common.state import ResumeState


def check_extractions(state):
    """Join point for the two parallel agents. Routing happens in route_after_check."""
    return {}


def route_after_check(state):
    if state.get("errors") or not state.get("candidate_data") or not state.get("job_data"):
        return END
    return "resume_creator"


def build_pipeline(creator_node, resume_parser, jd_retriever):
    g = StateGraph(ResumeState)
    g.add_node("resume_parser", resume_parser)
    g.add_node("jd_retriever", jd_retriever)
    g.add_node("check_extractions", check_extractions)
    g.add_node("resume_creator", creator_node)
    g.add_edge(START, "resume_parser")
    g.add_edge(START, "jd_retriever")
    g.add_edge(["resume_parser", "jd_retriever"], "check_extractions")   # waits for both
    g.add_conditional_edges("check_extractions", route_after_check, ["resume_creator", END])
    g.add_edge("resume_creator", END)
    return g.compile()


def run_pipeline(creator_factory, tracker, resume_path, job_url="", pasted_jd="", run_id=None):
    """Runs the whole pipeline and returns the final state.

    creator_factory(tracker, resume_path=..., run_id=...) -> LangGraph node for the creator.
    The final state has resume_md / output_path / generated_resume on success, errors otherwise."""
    app = build_pipeline(creator_factory(tracker, resume_path=resume_path, run_id=run_id),
                         make_resume_parser(tracker), make_jd_retriever(tracker))
    return app.invoke({"resume_path": resume_path, "job_url": job_url or "", "pasted_jd": pasted_jd or "",
                       "errors": []})
