import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from common.llm import Tracker
from workflows.w6_best_of_n import (
    _aggregate_and_select,
    _extract_json,
    _judge_node,
    _normalise_judgments,
    run,
    validate_candidate,
)

PAIR = {"pair_id": "p1", "jd": "Job description text.", "resume": "Original resume text."}


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

    @patch("workflows.w6_best_of_n.build_graph")
    def test_failed_selection_returns_runner_error(self, build_graph):
        build_graph.return_value.invoke.return_value = {
            "candidates": [],
            "selected": {
                "strategy": None,
                "resume_md": "",
                "finished": False,
                "candidate_scores": {},
                "reason": "No candidate passed.",
            },
        }
        result = run({"resume": "original"}, tracker=object())
        self.assertFalse(result["finished"])
        self.assertEqual(result["resume_md"], "")
        self.assertTrue(result["errors"][0].startswith("W6_NO_VALID_CANDIDATE:"))


def verdict(strategy, **overrides):
    item = {
        "strategy": strategy,
        "justification": "All claims are supported by the original resume.",
        "unsupported_claims": [],
        "faithfulness": 5,
        "jd_alignment": 4,
        "skills_coverage": 4,
        "writing_quality": 4,
        "valid": True,
    }
    item.update(overrides)
    return item


def judge_one(**overrides):
    raw = {"evaluations": [verdict("skills_first", **overrides)]}
    return _normalise_judgments(raw, {"skills_first"})["skills_first"]


def valid_candidate(strategy):
    text = complete_resume(strategy)
    return {"strategy": strategy, "resume_md": text,
            "validation": validate_candidate(text), "generation": {}}


class FakeChat:
    """Replays canned replies and records every call's messages and format argument."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages, tracker, format=None, **kwargs):
        self.calls.append({"messages": messages, "format": format})
        tracker.log(type="llm", input_tokens=0, output_tokens=0, thinking_chars=0, seconds=0)
        reply = self.replies.pop(0)
        if callable(reply):
            reply = reply(messages)
        return SimpleNamespace(content=reply)


class JustificationScopeTests(unittest.TestCase):
    def test_jd_alignment_remark_in_justification_does_not_reject(self):
        # Verbatim justification from se_resume_011__se_jd_03, previously rejected by the regex.
        result = judge_one(justification=(
            "Draft is identical to original resume; all claims are fully supported. Minor JD "
            "alignment gap: 'PostgreSQL' and 'Docker' are listed in skills but not explicitly "
            "mentioned in experience bullets."
        ))
        self.assertTrue(result["valid"])
        self.assertFalse(result["grounding_concern_in_justification"])

    def test_negated_invention_and_implied_jd_gap_do_not_reject(self):
        for text in (
            "Content is identical to the original resume with no invented facts.",
            "Faithfulness is perfect as no claims are invented.",
            "Minor gap in explicit 'stakeholder communication' phrasing compared to JD, though "
            "implied by collaboration details.",
            "Removed unsupported 'Python' and 'Security+' from skills list.",
            "Skills coverage is high, missing only 'Jest' which was not explicitly mentioned in "
            "the original.",
        ):
            with self.subTest(text=text):
                self.assertTrue(judge_one(justification=text)["valid"])

    def test_justification_admitting_added_content_still_rejects(self):
        for text in (
            "Added 'customer research' to skills which is not in the original resume.",
            "Faithful to original data; adds 'manages senior stakeholders' which is a reasonable "
            "inference from managing work items.",
            "Text is identical; the summary claim about mentoring is not in the original resume.",
        ):
            with self.subTest(text=text):
                result = judge_one(justification=text)
                self.assertFalse(result["valid"])
                self.assertTrue(result["grounding_concern_in_justification"])
                self.assertTrue(result["grounding_concern_evidence"])

    def test_listed_unsupported_claim_rejects_even_when_judge_says_valid(self):
        result = judge_one(unsupported_claims=["Kubernetes certification"])
        self.assertFalse(result["valid"])
        self.assertEqual(result["unsupported_claims"], ["Kubernetes certification"])

    def test_blank_unsupported_claim_entries_are_ignored(self):
        self.assertTrue(judge_one(unsupported_claims=["", "  "])["valid"])

    def test_faithfulness_below_five_rejects(self):
        self.assertFalse(judge_one(faithfulness=4)["valid"])

    def test_judge_valid_false_rejects(self):
        self.assertFalse(judge_one(valid=False)["valid"])


class JudgeCallTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [valid_candidate("skills_first"), valid_candidate("impact_first")]
        self.state = {"pair": PAIR, "candidates": self.candidates, "judge_passes": []}

    def run_judge(self, replies, reverse=False):
        fake = FakeChat(replies)
        tracker = Tracker()
        with patch("workflows.w6_best_of_n.chat", fake):
            out = _judge_node(tracker, reverse=reverse)(self.state)
        return out["judge_passes"][-1], fake, tracker

    def test_judge_call_passes_a_schema_restricted_to_the_candidates(self):
        reply = json.dumps({"evaluations": [verdict("skills_first"), verdict("impact_first")]})
        scores, fake, _ = self.run_judge([reply])
        self.assertEqual(set(scores), {"skills_first", "impact_first"})
        self.assertEqual(len(fake.calls), 1)
        schema = fake.calls[0]["format"]
        self.assertIsInstance(schema, dict)
        evaluations = schema["properties"]["evaluations"]
        self.assertEqual((evaluations["minItems"], evaluations["maxItems"]), (2, 2))
        item = schema["$defs"]["W6CandidateEvaluation"]
        self.assertEqual(set(item["properties"]["strategy"]["enum"]), {"skills_first", "impact_first"})
        for field in ("unsupported_claims", "faithfulness", "valid", "justification"):
            self.assertIn(field, item["required"])

    def test_retry_for_missing_strategy_is_also_schema_constrained(self):
        partial = json.dumps({"evaluations": [verdict("skills_first")]})
        retry = json.dumps({"evaluations": [verdict("impact_first")]})
        scores, fake, tracker = self.run_judge([partial, retry])
        self.assertEqual(set(scores), {"skills_first", "impact_first"})
        self.assertEqual(fake.calls[1]["format"]["$defs"]["W6CandidateEvaluation"]
                         ["properties"]["strategy"]["enum"], ["impact_first"])
        event = [e for e in tracker.events if e.get("event") == "w6_judge_pass"][0]
        self.assertEqual(event["retried_strategies"], ["impact_first"])

    def test_malformed_judge_output_is_retried_then_left_unscored(self):
        truncated = '{"evaluations": [{"strategy": "skills_first", "faithfulness": 5'
        scores, fake, _ = self.run_judge(["not json at all", truncated])
        self.assertEqual(scores, {})
        self.assertEqual(len(fake.calls), 2)
        winner, details = _aggregate_and_select(self.candidates, [scores, scores])
        self.assertIsNone(winner)
        self.assertFalse(details["skills_first"]["valid"])

    def test_malformed_first_reply_recovers_on_retry(self):
        retry = json.dumps({"evaluations": [verdict("skills_first"), verdict("impact_first")]})
        scores, _, _ = self.run_judge(["```json\n{broken", retry])
        self.assertTrue(scores["skills_first"]["valid"])
        self.assertTrue(scores["impact_first"]["valid"])

    def test_unknown_strategy_names_are_ignored(self):
        reply = json.dumps({"evaluations": [verdict("Candidate 1"), verdict("impact_first")]})
        retry = json.dumps({"evaluations": [verdict("skills_first", faithfulness=3)]})
        scores, _, _ = self.run_judge([reply, retry])
        self.assertEqual(set(scores), {"skills_first", "impact_first"})
        self.assertFalse(scores["skills_first"]["valid"])


class EndToEndTests(unittest.TestCase):
    def test_false_positive_justification_no_longer_blocks_selection(self):
        justification = ("All claims are fully supported. 'Docker' is listed in skills but not "
                         "explicitly mentioned in experience bullets.")

        def judge_reply(messages):
            prompt = messages[-1]["content"]
            names = [s for s in ("skills_first", "impact_first", "jd_wording_alignment")
                     if f": {s} ===" in prompt]
            return json.dumps({"evaluations": [
                verdict(name, justification=justification) for name in names
            ]})

        drafts = [complete_resume(s) for s in ("a", "b", "c")]
        fake = FakeChat(drafts + [judge_reply, judge_reply])
        with patch("workflows.w6_best_of_n.chat", fake):
            result = run(PAIR, Tracker())
        self.assertTrue(result["finished"], result["errors"])
        self.assertEqual(result["details"]["selected_strategy"], "skills_first")
        self.assertTrue(all(call["format"] is None for call in fake.calls[:3]))
        self.assertTrue(all(isinstance(call["format"], dict) for call in fake.calls[3:]))

    def test_listed_unsupported_claim_in_both_passes_fails_the_run(self):
        reply = json.dumps({"evaluations": [
            verdict(s, unsupported_claims=["AWS certification"])
            for s in ("skills_first", "impact_first", "jd_wording_alignment")
        ]})
        drafts = [complete_resume(s) for s in ("a", "b", "c")]
        fake = FakeChat(drafts + [reply, reply])
        with patch("workflows.w6_best_of_n.chat", fake):
            result = run(PAIR, Tracker())
        self.assertFalse(result["finished"])
        self.assertTrue(result["errors"][0].startswith("W6_NO_VALID_CANDIDATE:"))


if __name__ == "__main__":
    unittest.main()
