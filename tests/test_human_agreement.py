import csv
import json

import pytest

from eval.human_review import RUBRIC_DIMENSIONS, calculate_completed_forms, write_review_artifacts


def _row(index, judge_score, supported=True):
    dimensions = {
        dimension: {"score": judge_score} for dimension in RUBRIC_DIMENSIONS
    }
    return {
        "dataset_hash": "dataset-v1",
        "evaluator_hash": "evaluator-v1",
        "workflow": "w1_single_pass",
        "pair_id": f"pair-{index}",
        "repeat": 0,
        "config_hash": "generation-v1",
        "category": "general",
        "fit_level": "strong",
        "original_resume": f"Original resume {index}",
        "job_description": f"Job description {index}",
        "resume_md": f"Tailored resume {index}",
        "claim_scores": [{"claim": f"Claim {index}", "supported": supported}],
        "judge_status": "scored" if judge_score is not None else "unscored",
        "judge_cache_key": f"judge-cache-{index}",
        "judge_record": {"evaluator": {"evaluator_hash": "judge-v1"}},
        "judge_dimensions": dimensions if judge_score is not None else {},
        "judge_overall_mean": judge_score,
        "error": None,
    }


def _complete_forms(review_dir, rubric_scores, claim_labels):
    mapping = json.loads((review_dir / "review_mapping.private.json").read_text())
    with (review_dir / "rubric_ratings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["review_id", "reviewer_id", *RUBRIC_DIMENSIONS, "notes"]
        )
        writer.writeheader()
        for item in mapping:
            score = rubric_scores[item["pair_id"]]
            writer.writerow({
                "review_id": item["review_id"],
                "reviewer_id": "reviewer-1",
                **{dimension: score for dimension in RUBRIC_DIMENSIONS},
            })
    with (review_dir / "claim_support_ratings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "review_id", "claim_index", "claim_id", "reviewer_id", "rating", "notes",
            ],
        )
        writer.writeheader()
        for item in mapping:
            claim = item["claims"][0]
            writer.writerow({
                "review_id": item["review_id"],
                "claim_index": claim["claim_index"],
                "claim_id": claim["claim_id"],
                "reviewer_id": "reviewer-1",
                "rating": claim_labels[item["pair_id"]],
            })


def test_completed_forms_link_judge_and_claim_scores_with_missing_unscored(tmp_path):
    rows = [_row(1, 1, True), _row(2, 3, False), _row(3, None, True)]
    write_review_artifacts(tmp_path, rows, limit=3, seed=1)
    _complete_forms(
        tmp_path,
        {"pair-1": 1, "pair-2": 3, "pair-3": 5},
        {"pair-1": "supported", "pair-2": "unsupported", "pair-3": "supported"},
    )

    report = calculate_completed_forms(
        tmp_path, evaluations=rows, bootstrap_samples=20, seed=7
    )

    relevance = report["judge_vs_human"]["dimensions"]["relevance_to_jd"]
    assert relevance["eligible_items"] == 3
    assert relevance["paired_items"] == 2
    assert relevance["unscored_items"] == 1
    assert relevance["spearman"] == 1.0
    assert relevance["quadratic_weighted_kappa"] == 1.0
    claims = report["judge_vs_human"]["claim_support"]
    assert claims["paired_claims"] == 3
    assert claims["accuracy"] == 1.0
    assert claims["cohen_kappa"] == 1.0
    assert (tmp_path / "judge_human_agreement.csv").exists()
    assert (tmp_path / "claim_human_agreement.csv").exists()


def test_linkage_rejects_stale_tailored_resume(tmp_path):
    rows = [_row(1, 4)]
    write_review_artifacts(tmp_path, rows, limit=1)
    _complete_forms(tmp_path, {"pair-1": 4}, {"pair-1": "supported"})
    changed = [{**rows[0], "resume_md": "Changed after review packet creation"}]

    with pytest.raises(ValueError, match="stale tailored_resume_sha256"):
        calculate_completed_forms(tmp_path, evaluations=changed, bootstrap_samples=0)


def test_constant_human_ratings_leave_spearman_undefined(tmp_path):
    rows = [_row(1, 1), _row(2, 5)]
    write_review_artifacts(tmp_path, rows, limit=2)
    _complete_forms(
        tmp_path,
        {"pair-1": 3, "pair-2": 3},
        {"pair-1": "supported", "pair-2": "supported"},
    )

    report = calculate_completed_forms(tmp_path, evaluations=rows, bootstrap_samples=10)

    relevance = report["judge_vs_human"]["dimensions"]["relevance_to_jd"]
    assert relevance["spearman"] is None
    assert relevance["spearman_ci_95"] is None


def test_claim_identity_mismatch_fails_closed(tmp_path):
    rows = [_row(1, 4)]
    write_review_artifacts(tmp_path, rows, limit=1)
    _complete_forms(tmp_path, {"pair-1": 4}, {"pair-1": "supported"})
    path = tmp_path / "claim_support_ratings.csv"
    lines = path.read_text().splitlines()
    lines[1] = lines[1].replace(",claim_", ",claim_wrong_", 1)
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="claim identity mismatch"):
        calculate_completed_forms(tmp_path, evaluations=rows, bootstrap_samples=0)
