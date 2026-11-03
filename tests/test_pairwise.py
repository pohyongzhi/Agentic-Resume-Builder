import json
from types import SimpleNamespace

import pytest

from eval.judge import JudgeConfig
from eval.pairwise import (
    build_generation_pairs,
    candidate_text,
    evaluate_pairwise,
    summarise_pairwise,
)


ORIGINAL = "Built Python systems"
JD = "Need Python"
CHALLENGER = "Candidate alpha Python"
BASELINE = "Candidate beta Python"


def _valid_response(winner, candidate_a, candidate_b):
    def dimension(required_source=None):
        quotations = [
            {"source": "candidate_a", "quote": candidate_a},
            {"source": "candidate_b", "quote": candidate_b},
        ]
        if required_source == "job_description":
            quotations.insert(0, {"source": "job_description", "quote": JD})
        elif required_source == "original_resume":
            quotations.insert(0, {"source": "original_resume", "quote": ORIGINAL})
        return {
            "preferred": winner,
            "explanation": "Comparison grounded in the quoted text.",
            "quotations": quotations,
        }

    return json.dumps({
        "winner": winner,
        "explanation": "One candidate is better, or they are tied.",
        "relevance_to_jd": dimension("job_description"),
        "prioritisation_of_supported_experience": dimension("original_resume"),
        "clarity": dimension(),
        "concision": dimension(),
        "organisation_and_readability": dimension(),
    })


class FakeClient:
    def __init__(self, responses, digest="digest-a"):
        self.responses = list(responses)
        self.digest = digest
        self.calls = []

    def list(self):
        return SimpleNamespace(models=[
            SimpleNamespace(model="gemma-test", digest=self.digest)
        ])

    def show(self, model):
        return SimpleNamespace(modelinfo={
            "general.architecture": "gemma",
            "general.parameter_count": 3_000_000_000,
        })

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            message=SimpleNamespace(content=self.responses.pop(0)),
            prompt_eval_count=10,
            eval_count=20,
        )


def _config(**changes):
    values = {
        "model": "gemma-test",
        "provider": "ollama",
        "temperature": 0,
        "seed": 42,
        "num_ctx": 1024,
        "think": False,
        "reasoning_effort": "medium",
        "max_completion_tokens": 2048,
        "min_request_interval_seconds": 0,
        "max_retries": 1,
        "prompt_version": "single-judge-v1",
        "schema_version": "1",
    }
    values.update(changes)
    return JudgeConfig(**values)


def _evaluate(tmp_path, client, **changes):
    values = {
        "original_resume": ORIGINAL,
        "job_description": JD,
        "challenger_resume": CHALLENGER,
        "baseline_resume": BASELINE,
        "cache_dir": tmp_path,
        "challenger_workflow": "w3_critic",
        "baseline_workflow": "w2_baseline",
        "pair_id": "pair-1",
        "dataset_hash": "dataset-a",
        "challenger_config_hash": "c3",
        "baseline_config_hash": "c2",
        "judge_config": _config(),
        "client": client,
    }
    values.update(changes)
    return evaluate_pairwise(**values)


def test_two_orders_normalize_to_same_challenger_and_hide_workflows(tmp_path):
    client = FakeClient([
        _valid_response("candidate_a", CHALLENGER, BASELINE),
        _valid_response("candidate_b", BASELINE, CHALLENGER),
    ])
    record = _evaluate(tmp_path, client)

    assert record["status"] == "scored"
    assert record["position_swap_consistent"] is True
    assert record["consensus_winner"] == "challenger"
    assert record["orders"]["forward"]["normalized_winner"] == "challenger"
    assert record["orders"]["swapped"]["normalized_winner"] == "challenger"
    prompts = json.dumps(client.calls)
    assert "w3_critic" not in prompts
    assert "w2_baseline" not in prompts
    assert client.calls[0]["format"]["additionalProperties"] is False


def test_position_preference_is_inconsistent_and_has_no_consensus(tmp_path):
    client = FakeClient([
        _valid_response("candidate_a", CHALLENGER, BASELINE),
        _valid_response("candidate_a", BASELINE, CHALLENGER),
    ])
    record = _evaluate(tmp_path, client)

    assert record["status"] == "scored"
    assert record["position_swap_consistent"] is False
    assert record["consensus_winner"] is None


def test_tie_is_consistent_in_both_orders(tmp_path):
    client = FakeClient([
        _valid_response("tie", CHALLENGER, BASELINE),
        _valid_response("tie", BASELINE, CHALLENGER),
    ])
    record = _evaluate(tmp_path, client)
    assert record["position_swap_consistent"] is True
    assert record["consensus_winner"] == "tie"


def test_invalid_quotes_retry_then_fail_closed_without_consensus(tmp_path):
    invalid = _valid_response("candidate_a", "invented candidate", BASELINE)
    client = FakeClient([
        invalid,
        invalid,
        _valid_response("candidate_b", BASELINE, CHALLENGER),
    ])
    record = _evaluate(tmp_path, client)

    assert record["status"] == "unscored"
    assert record["position_swap_consistent"] is None
    assert record["consensus_winner"] is None
    assert len(record["orders"]["forward"]["attempts"]) == 2
    assert "unverifiable quotation" in record["failure_reason"]
    retry_prompt = client.calls[1]["messages"][1]["content"]
    assert "previous response was rejected" in retry_prompt


def test_cache_reuses_result_and_invalidates_on_inputs_and_model(tmp_path):
    first = _evaluate(tmp_path, FakeClient([
        _valid_response("candidate_a", CHALLENGER, BASELINE),
        _valid_response("candidate_b", BASELINE, CHALLENGER),
    ]))
    cached_client = FakeClient([])
    cached = _evaluate(tmp_path, cached_client)
    assert cached["cache_key"] == first["cache_key"]
    assert cached_client.calls == []

    changed_challenger = "Candidate alpha Python and SQL"
    changed_input = _evaluate(
        tmp_path,
        FakeClient([
            _valid_response("candidate_a", changed_challenger, BASELINE),
            _valid_response("candidate_b", BASELINE, changed_challenger),
        ]),
        challenger_resume=changed_challenger,
    )
    changed_model = _evaluate(
        tmp_path,
        FakeClient([
            _valid_response("candidate_a", CHALLENGER, BASELINE),
            _valid_response("candidate_b", BASELINE, CHALLENGER),
        ], digest="digest-b"),
    )
    assert changed_input["cache_key"] != first["cache_key"]
    assert changed_model["cache_key"] != first["cache_key"]


def _row(workflow, pair_id="p1", **changes):
    row = {
        "workflow": workflow,
        "dataset_hash": "dataset-a",
        "pair_id": pair_id,
        "repeat": 0,
        "config_hash": f"config-{workflow}",
        "resume_md": f"output-{workflow}",
        "error": None,
    }
    row.update(changes)
    return row


def test_generation_pairing_is_sorted_and_reports_missing_and_failed_runs():
    rows = [
        _row("w3_critic", "p2", error="generation failed"),
        _row("w2_baseline", "p2"),
        _row("w3_critic", "p1"),
        _row("w2_baseline", "p1"),
        _row("w2_baseline", "p3"),
    ]
    pairs = build_generation_pairs(rows, challenger_workflows=["w3_critic"])

    assert [pair["pair_id"] for pair in pairs] == ["p1", "p2", "p3"]
    assert [pair["status"] for pair in pairs] == [
        "ready", "generation_unavailable", "generation_unavailable"
    ]
    assert pairs[1]["failure_reason"] == "challenger_generation_failed"
    assert pairs[2]["failure_reason"] == "challenger_missing"


def test_pairing_rejects_duplicates_dataset_mismatch_and_ambiguous_configs():
    duplicate = [_row("w2_baseline"), _row("w2_baseline")]
    with pytest.raises(ValueError, match="duplicate generation row"):
        build_generation_pairs(duplicate, challenger_workflows=[])

    mismatch = [
        _row("w2_baseline"),
        _row("w3_critic", dataset_hash="dataset-b"),
    ]
    with pytest.raises(ValueError, match="dataset hash mismatch"):
        build_generation_pairs(mismatch, challenger_workflows=["w3_critic"])

    ambiguous = [
        _row("w2_baseline"),
        _row("w3_critic", "p1", config_hash="old"),
        _row("w3_critic", "p2", config_hash="new"),
    ]
    with pytest.raises(ValueError, match="multiple config hashes"):
        build_generation_pairs(ambiguous, challenger_workflows=["w3_critic"])


def test_candidate_text_rejects_failed_and_empty_outputs():
    assert candidate_text({"error": None, "resume_md": "Saved output"}) == "Saved output"
    with pytest.raises(ValueError, match="generation failed"):
        candidate_text({"error": "timeout", "resume_md": "partial"})
    with pytest.raises(ValueError, match="empty"):
        candidate_text({"error": None, "resume_md": "  "})


def test_summary_uses_two_order_denominators_and_reports_position_bias():
    def record(consistency, winner, forward="candidate_a", swapped="candidate_b"):
        return {
            "challenger_workflow": "w3_critic",
            "status": "scored",
            "position_swap_consistent": consistency,
            "consensus_winner": winner,
            "orders": {
                "forward": {"status": "scored", "winner_position": forward},
                "swapped": {"status": "scored", "winner_position": swapped},
            },
        }

    rows = [
        record(True, "challenger"),
        record(True, "tie", "tie", "tie"),
        record(False, None, "candidate_a", "candidate_a"),
        {
            "challenger_workflow": "w3_critic",
            "status": "unscored",
            "failure_reason": "invalid quote",
        },
    ]
    summary = summarise_pairwise(rows, ("challenger_workflow",))[0]

    assert summary["eligible_pairs"] == 4
    assert summary["two_order_scored"] == 3
    assert summary["consistent"] == 2
    assert summary["inconsistent"] == 1
    assert summary["challenger_wins"] == 1
    assert summary["ties"] == 1
    assert summary["strict_win_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert summary["half_tie_consensus_score"] == 0.5
    assert summary["first_position_selection_rate"] == 0.75
    assert summary["failure_reasons"] == {"invalid quote": 1}
