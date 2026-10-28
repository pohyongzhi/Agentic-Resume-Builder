"""BASELINE (shared control). One prompt, no tools. Same logic as baseline_tailor.py,
so the web page and the experiment use exactly the same baseline."""
from baseline_tailor import tailor

NAME = "baseline"
DISPLAY_NAME = "Workflow 1 — Baseline"
STATUS = "implemented"


def run(pair, tracker):
    resume, _ = tailor(pair["resume"], pair["jd"], tracker)
    return {"resume_md": resume, "hit_budget": False, "finished": True}
