"""V5 Variant 5: one-pass generation with the selected V4.1 fine-tuned model."""

import config
from baseline_tailor import tailor


NAME = "fine_tuned_small_generator"
DISPLAY_NAME = "Variant 5 — Fine-tuned small generator"
STATUS = "implemented"


def run(pair, tracker):
    resume, _ = tailor(
        pair["resume"], pair["jd"], tracker, model=config.FINE_TUNED_MODEL)
    return {"resume_md": resume, "hit_budget": False, "finished": True}
