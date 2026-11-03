"""W1 SINGLE PASS (lower bound). One prompt, one model call, no tools, no harness.
Shows how much the agentic baseline and its harness add."""
from baseline_tailor import tailor

NAME = "w1_single_pass"


def run(pair, tracker):
    resume, _ = tailor(pair["resume"], pair["jd"], tracker)
    return {"resume_md": resume, "hit_budget": False, "finished": True}
