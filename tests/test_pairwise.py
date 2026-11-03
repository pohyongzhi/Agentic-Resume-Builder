import json
import socket
from types import SimpleNamespace

import pytest

import eval.pairwise as pairwise
from eval.judge import JudgeConfig
from eval.pairwise import (
    build_generation_pairs,
    candidate_text,
    evaluate_generation_pair,
    evaluate_pairwise,
    summarise_pairwise,
)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted in a unit test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(
        pairwise, "_default_client",
        lambda judge_config: (_ for _ in ()).throw(AssertionError("default client used")),
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
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(
            message=SimpleNamespace(content=response),
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
    assert record["failure_kind"] == "invalid_output"
    assert record["orders_scored"] == 1
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


def test_summary_uses_explicit_denominators_and_reports_position_bias():
    def record(consistency, winner, forward="candidate_a", swapped="candidate_b"):
        return {
            "challenger_workflow": "w3_critic",
            "status": "scored",
            "position_swap_consistent": consistency,
            "consensus_winner": winner,
            "dimension_position_swap_consistent": {
                name: consistency for name in pairwise.DIMENSIONS
            },
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
            "failure_kind": "invalid_output",
            "failure_reason": "invalid quote",
        },
        {
            "challenger_workflow": "w3_critic",
            "status": "unscored",
            "failure_kind": "judge_failed",
            "failure_reason": "timeout",
        },
        {
            "challenger_workflow": "w3_critic",
            "status": "generation_unavailable",
            "failure_kind": "generation_unavailable",
            "failure_reason": "challenger_missing",
        },
    ]
    summary = summarise_pairwise(rows, ("challenger_workflow",))[0]

    assert summary["total_pairs"] == 6
    assert summary["generation_unavailable"] == 1
    assert summary["eligible"] == 5
    assert summary["scored"] == 3
    assert summary["unscored"] == 2
    assert summary["failed"] == 1
    assert summary["invalid"] == 1
    assert summary["consistent"] == 2
    assert summary["inconsistent"] == 1
    assert summary["challenger_wins"] == 1
    assert summary["baseline_wins"] == 0
    assert summary["ties"] == 1
    assert summary["strict_win_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert summary["strict_loss_rate"] == 0
    # Inconsistent outcome counts as a tie: (1 + 0.5 * (1 + 1)) / 3.
    assert summary["position_neutral_score"] == pytest.approx(2 / 3, abs=1e-6)
    assert summary["first_position_selection_rate"] == 0.75
    assert summary["dimension_position_swap_consistency"]["clarity"] == {
        "consistent": 2, "scored": 3, "rate": pytest.approx(2 / 3, abs=1e-6),
    }
    assert summary["failure_reasons"] == {
        "challenger_missing": 1, "invalid quote": 1, "timeout": 1,
    }


def test_summary_with_nothing_scored_reports_none_rates():
    rows = [{"status": "unscored", "failure_kind": "judge_failed", "failure_reason": "down"}]
    summary = summarise_pairwise(rows)[0]
    assert summary["eligible"] == 1
    assert summary["scored"] == 0
    assert summary["failed"] == 1
    assert summary["strict_win_rate"] is None
    assert summary["position_swap_consistency"] is None
    assert summary["position_neutral_score"] is None
    assert summary["first_position_selection_rate"] is None


def test_judge_failure_stays_unscored_and_is_not_cached(tmp_path):
    client = FakeClient([
        ConnectionError("judge unavailable"),
        ConnectionError("judge unavailable"),
        _valid_response("candidate_b", BASELINE, CHALLENGER),
    ])
    record = _evaluate(tmp_path, client)

    assert record["status"] == "unscored"
    assert record["failure_kind"] == "judge_failed"
    assert record["consensus_winner"] is None
    assert record["position_swap_consistent"] is None
    forward = record["orders"]["forward"]
    assert forward["normalized_winner"] is None
    assert [a["status"] for a in forward["attempts"]] == ["failed", "failed"]
    assert "judge unavailable" in record["failure_reason"]
    # Transport retries resend the original prompt, not a validation correction.
    assert "previous response was rejected" not in client.calls[1]["messages"][1]["content"]
    assert list(tmp_path.glob("*.json")) == []

    retry = _evaluate(tmp_path, FakeClient([
        _valid_response("candidate_a", CHALLENGER, BASELINE),
        _valid_response("candidate_b", BASELINE, CHALLENGER),
    ]))
    assert retry["status"] == "scored"
    assert retry["consensus_winner"] == "challenger"


def test_empty_judge_output_is_invalid_not_failed(tmp_path):
    client = FakeClient(["", "", "", ""])
    record = _evaluate(tmp_path, client)
    assert record["status"] == "unscored"
    assert record["failure_kind"] == "invalid_output"
    assert record["orders_scored"] == 0
    assert record["orders"]["swapped"]["attempts"][0]["status"] == "invalid"


def test_dimension_preferences_are_normalised_and_checked_for_consistency(tmp_path):
    forward = json.loads(_valid_response("candidate_a", CHALLENGER, BASELINE))
    swapped = json.loads(_valid_response("candidate_b", BASELINE, CHALLENGER))
    swapped["clarity"]["preferred"] = "candidate_a"  # baseline under swap: disagrees
    record = _evaluate(tmp_path, FakeClient([json.dumps(forward), json.dumps(swapped)]))

    assert record["orders"]["swapped"]["normalized_dimensions"]["clarity"] == "baseline"
    assert record["dimension_position_swap_consistent"]["clarity"] is False
    assert record["dimension_position_swap_consistent"]["concision"] is True
    assert record["dimension_consensus"]["clarity"] is None
    assert record["dimension_consensus"]["concision"] == "challenger"


def test_preresolved_model_info_skips_lookup_and_cache_hit_needs_no_client(tmp_path):
    model_info = {"provider": "ollama", "tag": "gemma-test", "digest": "digest-a"}

    class NoListClient(FakeClient):
        def list(self):
            raise AssertionError("model lookup should be skipped")

    first = _evaluate(tmp_path, NoListClient([
        _valid_response("candidate_a", CHALLENGER, BASELINE),
        _valid_response("candidate_b", BASELINE, CHALLENGER),
    ]), model_info=model_info)
    cached = _evaluate(tmp_path, None, model_info=model_info)
    assert cached == first


def test_cache_identity_covers_versions_provider_and_comparison(tmp_path, monkeypatch):
    model_info = {"provider": "ollama", "tag": "gemma-test", "digest": "digest-a"}

    def run(**changes):
        return _evaluate(tmp_path, FakeClient([
            _valid_response("candidate_a", CHALLENGER, BASELINE),
            _valid_response("candidate_b", BASELINE, CHALLENGER),
        ]), **{"model_info": model_info, **changes})

    base = run()["cache_key"]
    assert run(judge_config=_config(temperature=0.5))["cache_key"] != base
    assert run(repeat=1)["cache_key"] != base
    assert run(challenger_config_hash="other")["cache_key"] != base
    assert run(model_info={**model_info, "provider": "other"})["cache_key"] != base
    monkeypatch.setattr(pairwise, "PAIRWISE_PROMPT_VERSION", "pairwise-v2")
    assert run()["cache_key"] != base
    monkeypatch.setattr(pairwise, "PAIRWISE_PROMPT_VERSION", "pairwise-v1")
    monkeypatch.setattr(pairwise, "PAIRWISE_SCHEMA_VERSION", "2")
    assert run()["cache_key"] != base
    monkeypatch.setattr(pairwise, "PAIRWISE_SCHEMA_VERSION", "1")
    monkeypatch.setattr(pairwise, "PAIRWISE_EVALUATOR_VERSION", "other")
    assert run()["cache_key"] != base


def test_corrupt_cache_is_recomputed(tmp_path):
    first = _evaluate(tmp_path, FakeClient([
        _valid_response("candidate_a", CHALLENGER, BASELINE),
        _valid_response("candidate_b", BASELINE, CHALLENGER),
    ]))
    (tmp_path / f"{first['cache_key']}.json").write_text("{not json", encoding="utf-8")
    client = FakeClient([
        _valid_response("tie", CHALLENGER, BASELINE),
        _valid_response("tie", BASELINE, CHALLENGER),
    ])
    again = _evaluate(tmp_path, client)
    assert len(client.calls) == 2
    assert again["consensus_winner"] == "tie"


def test_rejects_empty_inputs_and_same_workflow(tmp_path):
    with pytest.raises(ValueError, match="challenger_resume"):
        _evaluate(tmp_path, FakeClient([]), challenger_resume="  ")
    with pytest.raises(ValueError, match="must differ"):
        _evaluate(tmp_path, FakeClient([]), challenger_workflow="w2_baseline")


def test_generation_pair_runner_skips_unavailable_and_judges_ready(tmp_path):
    rows = [
        _row("w2_baseline", "p1", resume_md=BASELINE),
        _row("w3_critic", "p1", resume_md=CHALLENGER),
        _row("w2_baseline", "p2"),
        _row("w3_critic", "p2", resume_md="   "),
        _row("w2_baseline", "p3"),
    ]
    pairs = build_generation_pairs(rows, challenger_workflows=["w3_critic"])
    idle = FakeClient([])
    unavailable = evaluate_generation_pair(
        pairs[2], ORIGINAL, JD, tmp_path, judge_config=_config(), client=idle,
    )
    unreadable = evaluate_generation_pair(
        pairs[1], ORIGINAL, JD, tmp_path, judge_config=_config(), client=idle,
    )
    assert idle.calls == []
    assert unavailable["status"] == "generation_unavailable"
    assert unavailable["failure_reason"] == "challenger_missing"
    assert unreadable["status"] == "generation_unavailable"
    assert unreadable["failure_reason"].startswith("challenger_unreadable")

    ready = evaluate_generation_pair(
        pairs[0], ORIGINAL, JD, tmp_path, judge_config=_config(),
        client=FakeClient([
            _valid_response("candidate_a", CHALLENGER, BASELINE),
            _valid_response("candidate_b", BASELINE, CHALLENGER),
        ]),
    )
    assert ready["status"] == "scored"
    assert ready["challenger_config_hash"] == "config-w3_critic"
    assert ready["baseline_config_hash"] == "config-w2_baseline"

    summary = summarise_pairwise([ready, unavailable, unreadable])[0]
    assert summary["total_pairs"] == 3
    assert summary["eligible"] == 1
    assert summary["scored"] == 1
    assert summary["generation_unavailable"] == 2
