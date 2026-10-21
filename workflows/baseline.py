"""BASELINE (shared control): the agentic pipeline.

    resume parser agent + JD retrieval agent (in parallel) -> resume creator agent -> harness

The creator writes a JSON draft and can only use two tools, both run through the harness:
validate_resume (protected facts vs the parsed resume) and render_resume (fixed Markdown template,
saved inside OUTPUT_DIR). Other workflows reuse common/pipeline.py and swap only make_creator.
"""
from common.agent import run_agent
from common.harness import ToolHarness
from common.pipeline import run_pipeline
from common.prompts import CREATOR_SYSTEM, creator_prompt

NAME = "baseline"
NUDGE = ("You must use a tool. Call validate_resume with your full corrected draft as a JSON object, "
         "or render_resume if your latest draft is already valid.")


def make_creator(tracker, resume_path=None, run_id=None):
    def resume_creator(state):
        harness = ToolHarness(state["candidate_data"], resume_path=resume_path, run_id=run_id)
        run_agent(CREATOR_SYSTEM, creator_prompt(state["candidate_data"], state["job_data"]),
                  harness.tools(), tracker, executor=harness.execute, include_submit=False,
                  finish_tools=("render_resume",), nudge=NUDGE)
        return harness.outcome()
    return resume_creator


def generate(tracker, resume_path, job_url="", pasted_jd="", run_id=None):
    """Runs the baseline pipeline. Used by app.py and run()."""
    return run_pipeline(make_creator, tracker, resume_path, job_url=job_url, pasted_jd=pasted_jd, run_id=run_id)


def run(pair, tracker):
    state = generate(tracker, pair["resume_path"], pasted_jd=pair["jd"], run_id=f"{NAME}__{pair['pair_id']}")
    return {"resume_md": state.get("resume_md", ""),
            "hit_budget": bool(state.get("hit_budget")),
            "finished": bool(state.get("resume_md")),
            "errors": state.get("errors", []),
            "output_path": state.get("output_path")}
