import json

from eval.human_review import calculate_pairwise_agreement, write_review_artifacts
from eval.reporting import paired_delta_summary, summarise


def test_summary_has_explicit_denominators_and_missing_failures(monkeypatch):
    monkeypatch.setattr("config.REPORT_BOOTSTRAP_SAMPLES", 100)
    rows = [
        {
            "dataset_hash": "d", "workflow": "w1", "error": None,
            "task_success": True, "faithfulness": 0.9, "evaluation_failures": {},
        },
        {
            "dataset_hash": "d", "workflow": "w1", "error": None,
            "task_success": None, "faithfulness": None,
            "evaluation_failures": {"faithfulness": "model unavailable"},
        },
        {
            "dataset_hash": "d", "workflow": "w1", "error": "generation failed",
            "task_success": False, "evaluation_failures": {},
        },
    ]
    table = summarise(rows, ("workflow",))
    task = next(row for row in table if row["metric"] == "task_success")
    faithfulness = next(row for row in table if row["metric"] == "faithfulness")
    assert (task["numerator"], task["denominator"]) == (1, 2)
    assert (faithfulness["eligible"], faithfulness["scored"], faithfulness["unscored"]) == (2, 1, 1)
    assert faithfulness["evaluation_failures"] == 1


def test_summary_reports_faithfulness_components_separately(monkeypatch):
    monkeypatch.setattr("config.REPORT_BOOTSTRAP_SAMPLES", 100)
    rows = [{
        "dataset_hash": "d",
        "workflow": "w1",
        "error": None,
        "claim_faithfulness": 0.75,
        "unsupported_claim_rate": 0.25,
        "protected_field_violation": False,
        "fabrication_bait_violation": True,
        "evaluation_failures": {},
    }]
    table = summarise(rows, ("workflow",))
    indexed = {row["metric"]: row for row in table}
    assert indexed["claim_faithfulness"]["mean"] == 0.75
    assert indexed["unsupported_claim_rate"]["mean"] == 0.25
    assert indexed["protected_field_violation"]["numerator"] == 0
    assert indexed["fabrication_bait_violation"]["numerator"] == 1


def test_robustness_deltas_are_paired_and_separate():
    rows = [
        {"workflow": "w1", "base_pair_id": "p1", "variant": "original",
         "faithfulness": 0.9, "sim_after": 0.8},
        {"workflow": "w1", "base_pair_id": "p1", "variant": "formatting",
         "faithfulness": 0.7, "sim_after": 0.85},
    ]
    result = paired_delta_summary(rows)
    faithfulness = next(row for row in result if row["metric"] == "faithfulness")
    similarity = next(row for row in result if row["metric"] == "sim_after")
    assert faithfulness["mean_delta"] == -0.2
    assert similarity["mean_delta"] == 0.05


def test_human_agreement_is_pairwise():
    records = [
        {"review_id": "a", "reviewer_id": "r1", "rating": 1},
        {"review_id": "a", "reviewer_id": "r2", "rating": 1},
        {"review_id": "b", "reviewer_id": "r1", "rating": 3},
        {"review_id": "b", "reviewer_id": "r2", "rating": 3},
    ]
    result = calculate_pairwise_agreement(
        records, ("review_id",), "rating", (1, 2, 3, 4, 5)
    )
    assert result[0]["paired_items"] == 2
    assert result[0]["weighted_kappa"] == 1.0


def test_review_ids_distinguish_workflows_and_match_every_artifact(tmp_path):
    common = {
        "dataset_hash": "dataset",
        "pair_id": "pair-1",
        "repeat": 0,
        "config_hash": "config",
        "category": "general",
        "fit_level": "strong",
        "original_resume": "Original",
        "job_description": "Role",
        "claim_scores": [{"claim": "Supported claim"}],
        "error": None,
    }
    rows = [
        {**common, "workflow": "w1_single_pass", "resume_md": "Output one"},
        {**common, "workflow": "w4_evidence", "resume_md": "Output two"},
    ]
    write_review_artifacts(tmp_path, rows, limit=2, seed=1)

    packet = [json.loads(line) for line in (tmp_path / "review_packet.jsonl").read_text().splitlines()]
    mapping = json.loads((tmp_path / "review_mapping.private.json").read_text())
    claim_rows = (tmp_path / "claim_support_ratings.csv").read_text().splitlines()[1:]
    packet_ids = {item["review_id"] for item in packet}
    assert len(packet_ids) == 2
    assert {item["review_id"] for item in mapping} == packet_ids
    assert {line.split(",", 1)[0] for line in claim_rows} == packet_ids


def test_robustness_excludes_failed_or_empty_outputs_from_paired_scores():
    original = {'workflow': 'w1', 'base_pair_id': 'p1', 'variant': 'original',
                'faithfulness': .9, 'resume_md': 'valid', 'finished': True}
    failed = {**original, 'variant': 'formatting', 'faithfulness': .1, 'error': 'failure'}
    empty = {**original, 'variant': 'reordered_sections', 'faithfulness': .1, 'resume_md': ''}
    assert paired_delta_summary([original, failed, empty]) == []
