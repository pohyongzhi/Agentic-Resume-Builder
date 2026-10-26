import unittest

from workflows.w6_best_of_n import (
    _aggregate_and_select,
    _extract_json,
    _normalise_judgments,
    validate_candidate,
)


def complete_resume(extra=""):
    body = " ".join(["supported experience and skill"] * 25)
    return (
        f"# Summary\n{body}\n# Experience\n{body}\n# Education\n{body}\n"
        f"# Skills\n{body}\n{extra}"
    )


class BestOfNTests(unittest.TestCase):
    def test_extracts_fenced_json(self):
        self.assertEqual(_extract_json('```json\n{"evaluations": []}\n```'), {"evaluations": []})

    def test_deterministic_validator_requires_sections(self):
        self.assertTrue(validate_candidate(complete_resume())["valid"])
        self.assertFalse(validate_candidate("# Summary\nToo short")["valid"])

    def test_judge_scores_are_clamped_and_unsupported_claims_invalidate(self):
        raw = {"evaluations": [{
            "strategy": "skills_first",
            "faithfulness": 8,
            "jd_alignment": 4,
            "skills_coverage": 4,
            "writing_quality": 4,
            "valid": True,
            "unsupported_claims": ["Invented Python"],
        }]}
        result = _normalise_judgments(raw, {"skills_first"})["skills_first"]
        self.assertEqual(result["faithfulness"], 5)
        self.assertFalse(result["valid"])

    def test_judge_prose_cannot_admit_an_inference_and_mark_it_valid(self):
        raw = {"evaluations": [{
            "strategy": "skills_first",
            "faithfulness": 5,
            "jd_alignment": 5,
            "skills_coverage": 5,
            "writing_quality": 5,
            "valid": True,
            "unsupported_claims": [],
            "justification": "Advanced Excel is a reasonable inference from the source.",
        }]}
        result = _normalise_judgments(raw, {"skills_first"})["skills_first"]
        self.assertFalse(result["valid"])
        self.assertTrue(result["grounding_concern_in_justification"])

    def test_selector_only_uses_candidates_valid_in_both_orders(self):
        candidates = []
        for strategy in ("skills_first", "impact_first"):
            candidates.append({
                "strategy": strategy,
                "resume_md": complete_resume(strategy),
                "validation": validate_candidate(complete_resume(strategy)),
                "generation": {},
            })
        strong = {
            "faithfulness": 5,
            "jd_alignment": 5,
            "skills_coverage": 5,
            "writing_quality": 5,
            "valid": True,
            "unsupported_claims": [],
            "justification": "",
        }
        weak = {**strong, "jd_alignment": 3}
        invalid = {**strong, "valid": False, "unsupported_claims": ["unsupported"]}
        winner, _ = _aggregate_and_select(
            candidates,
            [
                {"skills_first": strong, "impact_first": weak},
                {"skills_first": invalid, "impact_first": weak},
            ],
        )
        self.assertEqual(winner["strategy"], "impact_first")


if __name__ == "__main__":
    unittest.main()
