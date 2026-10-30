"""V5 Variant 1: one prompt, one base-model call, no tools."""

from baseline_tailor import tailor


NAME = "one_pass"
DISPLAY_NAME = "Variant 1 — One-pass"
STATUS = "implemented"


def run(pair, tracker):
    resume, _ = tailor(pair["resume"], pair["jd"], tracker)
    return {"resume_md": resume, "hit_budget": False, "finished": True}
