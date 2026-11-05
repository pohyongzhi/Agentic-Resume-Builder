import json
from types import SimpleNamespace

import pytest

from eval import quality_rubric as qr

ORIGINAL = "- Reduced manual preparation time by 30% with a reporting template."
JD = "Required: SQL, Excel and stakeholder communication."
TAILORED = "- Reduced manual preparation time by 30% with a reporting template.\n- Used SQL and Excel."


def _dimension(score, *sources):
    texts = {"tailored_resume": "Used SQL and Excel.", "job_description": "SQL, Excel",
             "original_resume": "reporting template"}
    return {"score": score, "explanation": "ok",
            "quotations": [{"source": s, "quote": texts[s]} for s in sources or ("tailored_resume",)]}


def _payload(**overrides):
    payload = {
        "quantified_evidence": _dimension(5),
        "clarity": _dimension(4),
        "concision": _dimension(5),
        "logical_order": _dimension(4),
    }
    payload.update(overrides)
    return payload


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        content = self.replies.pop(0)
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=usage)


def test_scores_and_mean_from_a_valid_reply():
    client = FakeClient([json.dumps(_payload())])
    record = qr.score(ORIGINAL, JD, TAILORED, client, "m", "medium")
    assert record["status"] == "scored"
    assert record["scores"]["quantified_evidence"] == 5
    assert record["overall"] == pytest.approx(4.5)
    assert record["mean"] == pytest.approx(4.5)


def test_invalid_quote_is_retried_with_feedback_then_scored():
    bad = _payload(clarity={"score": 3, "explanation": "x",
                            "quotations": [{"source": "tailored_resume", "quote": "not in the resume"}]})
    client = FakeClient([json.dumps(bad), json.dumps(_payload())])
    record = qr.score(ORIGINAL, JD, TAILORED, client, "m", "medium")
    assert record["status"] == "scored"
    assert record["attempts"][0]["status"] == "invalid"
    assert "unverifiable quotation" in record["attempts"][0]["error"]


def test_every_dimension_must_quote_the_tailored_resume():
    bad = _payload(clarity=_dimension(4, "original_resume"))
    client = FakeClient([json.dumps(bad)] * 3)
    record = qr.score(ORIGINAL, JD, TAILORED, client, "m", "medium")
    assert record["status"] == "unscored"
    assert record["failure_kind"] == "invalid_output"
    assert record["overall"] is None


def test_prompt_tells_the_judge_not_to_assess_truth():
    assert "do not judge whether claims are true" in qr.SYSTEM_PROMPT
    assert qr.DIMENSIONS == ("quantified_evidence", "clarity", "concision", "logical_order")
    assert sum(qr.WEIGHTS.values()) == pytest.approx(1.0)
