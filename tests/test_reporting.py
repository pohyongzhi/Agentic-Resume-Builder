from eval.human_review import calculate_pairwise_agreement
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
