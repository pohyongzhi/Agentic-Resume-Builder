"""W5 fine-tuned small generator using the selected V4.1 Ollama model."""

import config
from baseline_tailor import tailor


NAME = "w5_fine_tuned"


def run(pair, tracker):
    resume, _ = tailor(
        pair["resume"],
        pair["jd"],
        tracker,
        model=config.FINE_TUNED_MODEL,
    )
    return {"resume_md": resume, "hit_budget": False, "finished": True}
