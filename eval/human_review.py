"""Deterministic blinded review packets and post-annotation agreement statistics."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from eval.judge import RUBRIC

RUBRIC_DIMENSIONS = (
    "relevance_to_jd",
    "prioritisation_of_supported_experience",
    "clarity",
    "concision",
    "organisation_and_readability",
)


def _blind_id(dataset_hash, pair_id, repeat):
    return "review_" + hashlib.sha256(
        f"{dataset_hash}:{pair_id}:{repeat}".encode()
    ).hexdigest()[:12]


def select_review_sample(rows, limit=20, seed=42):
    """Stratify by category/fit, then prefer metric extremes and unscored cases."""
    candidates = [row for row in rows if not row.get("error") and row.get("resume_md")]
    buckets = defaultdict(list)
    for row in candidates:
        buckets[(row.get("category"), row.get("fit_level"))].append(row)
    rng = random.Random(seed)
    selected = []
    bucket_keys = sorted(buckets)
    while len(selected) < min(limit, len(candidates)) and bucket_keys:
        made_progress = False
        for key in list(bucket_keys):
            bucket = buckets[key]
            if not bucket:
                bucket_keys.remove(key)
                continue
            bucket.sort(key=lambda row: (
                row.get("judge_overall_mean") is None,
                row.get("faithfulness") if row.get("faithfulness") is not None else -1,
                row["pair_id"],
            ))
            choose_high = len(selected) % 2 == 1
            selected.append(bucket.pop(-1 if choose_high else 0))
            made_progress = True
            if len(selected) >= limit:
                break
        if not made_progress:
            break
    rng.shuffle(selected)
    return selected


def write_review_artifacts(output_dir, rows, limit=20, seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = select_review_sample(rows, limit, seed)
    packet_path = output_dir / "review_packet.jsonl"
    mapping = []
    with packet_path.open("w", encoding="utf-8") as packet:
        for row in sample:
            review_id = _blind_id(row["dataset_hash"], row["pair_id"], row.get("repeat", 0))
            packet.write(json.dumps({
                "review_id": review_id,
                "original_resume": row["original_resume"],
                "job_description": row["job_description"],
                "tailored_resume": row["resume_md"],
                "claims": [
                    score["claim"] for score in row.get("claim_scores", [])
                ],
            }, ensure_ascii=False) + "\n")
            mapping.append({
                "review_id": review_id,
                "workflow": row["workflow"],
                "pair_id": row["pair_id"],
                "repeat": row.get("repeat", 0),
            })

    with (output_dir / "rubric_ratings.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["review_id", "reviewer_id", *RUBRIC_DIMENSIONS, "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in mapping:
            writer.writerow({"review_id": item["review_id"]})
    with (output_dir / "claim_support_ratings.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["review_id", "claim_index", "reviewer_id", "rating", "notes"],
        )
        writer.writeheader()
        for row in sample:
            review_id = _blind_id(row["dataset_hash"], row["pair_id"], row.get("repeat", 0))
            for index, _ in enumerate(row.get("claim_scores", [])):
                writer.writerow({"review_id": review_id, "claim_index": index})
    (output_dir / "review_mapping.private.json").write_text(
        json.dumps(mapping, indent=2) + "\n", encoding="utf-8"
    )
    ranked = sorted(
        sample,
        key=lambda row: (
            row.get("task_success") is True,
            row.get("judge_overall_mean") if row.get("judge_overall_mean") is not None else -1,
            row.get("faithfulness") if row.get("faithfulness") is not None else -1,
        ),
    )
    qualitative = []
    for label, candidates in (("automated_failure", ranked[:3]), ("automated_success", ranked[-3:])):
        for row in candidates:
            qualitative.append({
                "review_id": _blind_id(
                    row["dataset_hash"], row["pair_id"], row.get("repeat", 0)
                ),
                "selection": label,
                "human_confirmed": False,
                "task_success": row.get("task_success"),
                "judge_overall_mean": row.get("judge_overall_mean"),
                "faithfulness": row.get("faithfulness"),
            })
    (output_dir / "qualitative_candidates.json").write_text(
        json.dumps(qualitative, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "INSTRUCTIONS.md").write_text(
        "# Independent human review\n\n"
        "Reviewers must work independently and must not inspect `review_mapping.private.json`.\n"
        "Read all three documents in each packet item. Rate every rubric dimension from 1 to 5\n"
        "using the same anchors as the automated judge. For each extracted claim, use exactly one\n"
        "claim-support label: `supported`, `contradicted`, or `unsupported`. Do not infer facts not\n"
        "present in the original resume. The forms are intentionally blank; automated findings are\n"
        "not human-confirmed until completed ratings are imported and agreement is calculated.\n\n"
        + RUBRIC,
        encoding="utf-8",
    )
    return {"sample_size": len(sample), "packet": str(packet_path)}


def _weighted_kappa(left, right, categories):
    if len(left) != len(right) or not left:
        return None
    size = len(categories)
    index = {value: position for position, value in enumerate(categories)}
    observed = [[0.0] * size for _ in range(size)]
    for a, b in zip(left, right):
        observed[index[a]][index[b]] += 1
    total = len(left)
    row = [sum(values) / total for values in observed]
    column = [sum(observed[i][j] for i in range(size)) / total for j in range(size)]
    ordinal = all(isinstance(value, int) for value in categories)
    def disagreement(i, j):
        return (
            (abs(i - j) / max(size - 1, 1)) ** 2
            if ordinal else float(i != j)
        )
    observed_disagreement = sum(
        disagreement(i, j) * observed[i][j] / total
        for i in range(size) for j in range(size)
    )
    expected_disagreement = sum(
        disagreement(i, j) * row[i] * column[j]
        for i in range(size) for j in range(size)
    )
    return None if expected_disagreement == 0 else 1 - observed_disagreement / expected_disagreement


def calculate_pairwise_agreement(records, item_fields, rating_field, categories):
    """Calculate pairwise agreement separately for every reviewer pair."""
    by_item = defaultdict(dict)
    for record in records:
        key = tuple(record[field] for field in item_fields)
        by_item[key][record["reviewer_id"]] = record[rating_field]
    reviewers = sorted({record["reviewer_id"] for record in records})
    results = []
    for left_index, left_reviewer in enumerate(reviewers):
        for right_reviewer in reviewers[left_index + 1:]:
            paired = [
                (ratings[left_reviewer], ratings[right_reviewer])
                for ratings in by_item.values()
                if left_reviewer in ratings and right_reviewer in ratings
            ]
            left = [item[0] for item in paired]
            right = [item[1] for item in paired]
            kappa = _weighted_kappa(left, right, categories)
            results.append({
                "reviewer_a": left_reviewer,
                "reviewer_b": right_reviewer,
                "paired_items": len(paired),
                "agreement_type": (
                    "quadratic_weighted_kappa"
                    if all(isinstance(value, int) for value in categories)
                    else "unweighted_cohen_kappa"
                ),
                "weighted_kappa": round(kappa, 6) if kappa is not None else None,
            })
    return results


def calculate_completed_forms(output_dir):
    """Validate completed CSV forms and write inter-human agreement without inventing ratings."""
    output_dir = Path(output_dir)
    rubric_records = []
    with (output_dir / "rubric_ratings.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get("reviewer_id"):
                continue
            for dimension in RUBRIC_DIMENSIONS:
                try:
                    rating = int(row[dimension])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{row.get('review_id')}/{dimension}: rating must be an integer"
                    ) from exc
                if rating not in range(1, 6):
                    raise ValueError(f"{row.get('review_id')}/{dimension}: rating outside 1-5")
                rubric_records.append({
                    "review_id": row["review_id"],
                    "dimension": dimension,
                    "reviewer_id": row["reviewer_id"],
                    "rating": rating,
                })
    claim_records = []
    with (output_dir / "claim_support_ratings.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            if not row.get("reviewer_id"):
                continue
            rating = row.get("rating", "").lower()
            if rating not in {"supported", "contradicted", "unsupported"}:
                raise ValueError(
                    f"{row.get('review_id')}/{row.get('claim_index')}: invalid claim rating"
                )
            claim_records.append({
                "review_id": row["review_id"],
                "claim_index": row["claim_index"],
                "reviewer_id": row["reviewer_id"],
                "rating": rating,
            })
    report = {
        "rubric_inter_human": calculate_pairwise_agreement(
            rubric_records,
            ("review_id", "dimension"),
            "rating",
            (1, 2, 3, 4, 5),
        ),
        "claim_support_inter_human": calculate_pairwise_agreement(
            claim_records,
            ("review_id", "claim_index"),
            "rating",
            ("supported", "contradicted", "unsupported"),
        ),
        "note": (
            "Inter-human agreement is reported separately. Judge-versus-human agreement should "
            "be calculated only after linking completed forms to saved automated scores."
        ),
    }
    (output_dir / "agreement.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("review_directory")
    arguments = parser.parse_args()
    print(json.dumps(calculate_completed_forms(arguments.review_directory), indent=2))
