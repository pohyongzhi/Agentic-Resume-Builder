"""W2 BASELINE (shared agentic control): the agentic pipeline.

    resume parser agent + JD retrieval agent (in parallel) -> resume creator agent -> harness

The creator writes a schema-constrained JSON draft. Python then invokes the two harness operations:
validate_resume (protected facts vs the parsed resume) and render_resume (fixed Markdown template,
saved inside OUTPUT_DIR). This avoids unreliable deeply nested tool arguments from small models
without bypassing any harness rule.
"""
import json
import time

from common.harness import ToolHarness
from common.llm import chat
from common.pipeline import run_pipeline
from common.prompts import STRUCTURED_CREATOR_SYSTEM, creator_prompt
from common.schemas import GeneratedResume, llm_schema

NAME = "w2_baseline"


def _execute(harness, tracker, name, args):
    """Run and record a harness operation just as the shared ReAct loop does."""
    start = time.time()
    ok, result, stop = harness.execute(name, args)
    tracker.log(type="tool", name=name, args=args, ok=ok, seconds=round(time.time() - start, 3))
    return ok, result, stop


def make_creator(tracker, resume_path=None, run_id=None):
    def resume_creator(state):
        harness = ToolHarness(state["candidate_data"], resume_path=resume_path, run_id=run_id)
        messages = [
            {"role": "system", "content": STRUCTURED_CREATOR_SYSTEM},
            {"role": "user", "content": creator_prompt(state["candidate_data"], state["job_data"])},
        ]

        while harness.calls < harness.max_calls:
            msg = chat(messages, tracker, format=llm_schema(GeneratedResume))
            draft = msg.content or ""
            _, result, stopped = _execute(harness, tracker, "validate_resume", {"resume": draft})
            if stopped:
                break
            feedback = json.loads(result)
            if feedback.get("valid"):
                _execute(harness, tracker, "render_resume", {})
                break
            messages += [
                {"role": "assistant", "content": draft},
                {"role": "user", "content":
                    "The harness rejected that draft. Return the complete corrected resume JSON. "
                    "Fix every error and do not add unsupported facts:\n" + result},
            ]
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
