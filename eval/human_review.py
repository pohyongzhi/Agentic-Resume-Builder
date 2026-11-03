"""Deterministic blinded review packets and post-annotation agreement statistics."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from eval.judge import RUBRIC

RUBRIC_DIMENSIONS = (
    "relevance_to_jd",
    "prioritisation_of_supported_experience",
    "clarity",
    "concision",
    "organisation_and_readability",
)

AGREEMENT_SCHEMA_VERSION = "1.0"


def _sha256(value):
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _claim_id(review_id, index, claim):
    return "claim_" + _sha256(f"{review_id}:{index}:{_sha256(claim)}")[:12]


def _blind_id(dataset_hash, pair_id, repeat, workflow, generation_identity):
    return "review_" + hashlib.sha256(
        f"{dataset_hash}:{pair_id}:{repeat}:{workflow}:{generation_identity}".encode()
    ).hexdigest()[:12]


def _row_blind_id(row):
    content_hash = _sha256(row.get("resume_md", ""))
    generation_identity = f"{row.get('config_hash', '')}:{content_hash}"
    return _blind_id(
        row["dataset_hash"], row["pair_id"], row.get("repeat", 0),
        row["workflow"], generation_identity,
    )


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
            review_id = _row_blind_id(row)
            packet.write(json.dumps({
                "review_id": review_id,
                "original_resume": row["original_resume"],
                "job_description": row["job_description"],
                "tailored_resume": row["resume_md"],
                "claims": [
                    score["claim"] for score in row.get("claim_scores", [])
                ],
            }, ensure_ascii=False) + "\n")
            claims = []
            for index, score in enumerate(row.get("claim_scores", [])):
                claim = score["claim"]
                claims.append({
                    "claim_index": index,
                    "claim_id": _claim_id(review_id, index, claim),
                    "claim_text_sha256": _sha256(claim),
                })
            mapping.append({
                "review_id": review_id,
                "dataset_hash": row["dataset_hash"],
                "workflow": row["workflow"],
                "pair_id": row["pair_id"],
                "repeat": row.get("repeat", 0),
                "variant": row.get("variant", ""),
                "config_hash": row.get("config_hash", ""),
                "evaluator_hash": row.get("evaluator_hash", ""),
                "tailored_resume_sha256": _sha256(row.get("resume_md", "")),
                "original_resume_sha256": _sha256(row.get("original_resume", "")),
                "job_description_sha256": _sha256(row.get("job_description", "")),
                "judge_cache_key": row.get("judge_cache_key"),
                "judge_evaluator_hash": (
                    row.get("judge_record", {}).get("evaluator", {}).get("evaluator_hash")
                ),
                "claims": claims,
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
            fieldnames=[
                "review_id", "claim_index", "claim_id", "reviewer_id", "rating", "notes",
            ],
        )
        writer.writeheader()
        for row in sample:
            review_id = _row_blind_id(row)
            for index, score in enumerate(row.get("claim_scores", [])):
                writer.writerow({
                    "review_id": review_id,
                    "claim_index": index,
                    "claim_id": _claim_id(review_id, index, score["claim"]),
                })
    (output_dir / "review_mapping.private.json").write_text(
        json.dumps(mapping, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": AGREEMENT_SCHEMA_VERSION,
        "seed": seed,
        "requested_limit": limit,
        "sample_size": len(sample),
        "dataset_hashes": sorted({row["dataset_hash"] for row in sample}),
        "evaluator_hashes": sorted({
            row.get("evaluator_hash", "") for row in sample if row.get("evaluator_hash")
        }),
        "review_ids": sorted(item["review_id"] for item in mapping),
    }
    (output_dir / "review_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
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
                "review_id": _row_blind_id(row),
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


def _average_ranks(values):
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = (start + 1 + end) / 2
        for index, _ in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson(left, right):
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_scale = sum((value - left_mean) ** 2 for value in left)
    right_scale = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_scale * right_scale)
    return None if denominator == 0 else numerator / denominator


def _spearman(left, right):
    if len(left) != len(right) or len(left) < 2:
        return None
    return _pearson(_average_ranks(left), _average_ranks(right))


def _percentile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _bootstrap_ci(pairs, statistic, samples, seed):
    if len(pairs) < 2 or samples < 1:
        return None
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        resampled = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        estimate = statistic(resampled)
        if estimate is not None and math.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return None
    return [
        round(_percentile(estimates, 0.025), 6),
        round(_percentile(estimates, 0.975), 6),
    ]


def _unique_mode(values):
    counts = defaultdict(int)
    for value in values:
        counts[value] += 1
    if not counts:
        return None
    highest = max(counts.values())
    winners = [value for value, count in counts.items() if count == highest]
    return winners[0] if len(winners) == 1 else None


def _agreement_summary(pairs, eligible_items, bootstrap_samples, seed):
    """Summarise judge/human pairs; human_mean may differ from ordinal consensus."""
    judge = [pair["judge"] for pair in pairs]
    human = [pair["human_mean"] for pair in pairs]
    ordinal = [pair for pair in pairs if pair.get("human_ordinal") is not None]
    ordinal_judge = [int(pair["judge"]) for pair in ordinal]
    ordinal_human = [pair["human_ordinal"] for pair in ordinal]
    spearman = _spearman(judge, human)
    kappa = _weighted_kappa(ordinal_judge, ordinal_human, (1, 2, 3, 4, 5))
    absolute = [abs(a - b) for a, b in zip(judge, human)]
    signed = [a - b for a, b in zip(judge, human)]
    result = {
        "eligible_items": eligible_items,
        "paired_items": len(pairs),
        "unscored_items": eligible_items - len(pairs),
        "ordinal_consensus_items": len(ordinal),
        "spearman": round(spearman, 6) if spearman is not None else None,
        "quadratic_weighted_kappa": round(kappa, 6) if kappa is not None else None,
        "exact_agreement": (
            round(sum(a == b for a, b in zip(ordinal_judge, ordinal_human)) / len(ordinal), 6)
            if ordinal else None
        ),
        "within_one_point": (
            round(sum(abs(a - b) <= 1 for a, b in zip(judge, human)) / len(pairs), 6)
            if pairs else None
        ),
        "mean_absolute_error": round(mean(absolute), 6) if absolute else None,
        "signed_bias": round(mean(signed), 6) if signed else None,
    }
    statistics = {
        "spearman": lambda sample: _spearman(
            [item["judge"] for item in sample],
            [item["human_mean"] for item in sample],
        ),
        "quadratic_weighted_kappa": lambda sample: _weighted_kappa(
            [int(item["judge"]) for item in sample if item.get("human_ordinal") is not None],
            [item["human_ordinal"] for item in sample if item.get("human_ordinal") is not None],
            (1, 2, 3, 4, 5),
        ),
        "exact_agreement": lambda sample: (
            mean(
                int(item["judge"]) == item["human_ordinal"]
                for item in sample if item.get("human_ordinal") is not None
            )
            if any(item.get("human_ordinal") is not None for item in sample) else None
        ),
        "within_one_point": lambda sample: mean(
            abs(item["judge"] - item["human_mean"]) <= 1 for item in sample
        ),
        "mean_absolute_error": lambda sample: mean(
            abs(item["judge"] - item["human_mean"]) for item in sample
        ),
        "signed_bias": lambda sample: mean(
            item["judge"] - item["human_mean"] for item in sample
        ),
    }
    for offset, (name, statistic) in enumerate(statistics.items()):
        result[f"{name}_ci_95"] = _bootstrap_ci(
            pairs, statistic, bootstrap_samples, seed + offset
        )
    null_reasons = {}
    if result["spearman"] is None:
        null_reasons["spearman"] = (
            "fewer than two pairs or zero rank variance"
        )
    if result["quadratic_weighted_kappa"] is None:
        null_reasons["quadratic_weighted_kappa"] = (
            "no unique ordinal consensus or zero expected disagreement"
        )
    result["null_reasons"] = null_reasons
    return result


def _load_evaluation_rows(source):
    if isinstance(source, (str, Path)):
        path = Path(source)
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return list(source)


def _link_evaluations(mapping, evaluation_rows):
    linked = {}
    for item in mapping:
        required = {
            "dataset_hash", "workflow", "pair_id", "repeat", "variant", "config_hash",
            "evaluator_hash", "tailored_resume_sha256", "original_resume_sha256",
            "job_description_sha256", "claims",
        }
        missing = sorted(required - item.keys())
        if missing:
            raise ValueError(
                f"{item.get('review_id')}: private mapping lacks identity fields: {missing}"
            )
        candidates = []
        for row in evaluation_rows:
            if any((
                row.get("dataset_hash") != item["dataset_hash"],
                row.get("workflow") != item["workflow"],
                row.get("pair_id") != item["pair_id"],
                row.get("repeat", 0) != item["repeat"],
                row.get("variant", "") != item["variant"],
                row.get("config_hash", "") != item["config_hash"],
                row.get("evaluator_hash", "") != item["evaluator_hash"],
            )):
                continue
            candidates.append(row)
        if len(candidates) != 1:
            raise ValueError(
                f"{item['review_id']}: expected exactly one evaluation record, found {len(candidates)}"
            )
        row = candidates[0]
        hashes = {
            "tailored_resume_sha256": _sha256(row.get("resume_md", "")),
            "original_resume_sha256": _sha256(row.get("original_resume", "")),
            "job_description_sha256": _sha256(row.get("job_description", "")),
        }
        for field, actual in hashes.items():
            if item[field] != actual:
                raise ValueError(f"{item['review_id']}: stale {field}")
        if item.get("judge_cache_key") != row.get("judge_cache_key"):
            raise ValueError(f"{item['review_id']}: judge cache identity mismatch")
        judge_hash = row.get("judge_record", {}).get("evaluator", {}).get("evaluator_hash")
        if item.get("judge_evaluator_hash") != judge_hash:
            raise ValueError(f"{item['review_id']}: judge evaluator identity mismatch")
        scores = row.get("claim_scores", [])
        if len(scores) != len(item["claims"]):
            raise ValueError(f"{item['review_id']}: claim count mismatch")
        for claim_mapping, score in zip(item["claims"], scores):
            if claim_mapping["claim_text_sha256"] != _sha256(score.get("claim", "")):
                raise ValueError(f"{item['review_id']}: stale claim mapping")
        linked[item["review_id"]] = row
    return linked


def _read_completed_forms(output_dir, valid_review_ids):
    rubric_records = []
    seen = set()
    with (output_dir / "rubric_ratings.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get("reviewer_id"):
                continue
            if row.get("review_id") not in valid_review_ids:
                raise ValueError(f"unknown review_id in rubric form: {row.get('review_id')}")
            for dimension in RUBRIC_DIMENSIONS:
                try:
                    rating = int(row[dimension])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{row.get('review_id')}/{dimension}: rating must be an integer"
                    ) from exc
                if rating not in range(1, 6):
                    raise ValueError(f"{row.get('review_id')}/{dimension}: rating outside 1-5")
                key = (row["review_id"], dimension, row["reviewer_id"])
                if key in seen:
                    raise ValueError(f"duplicate rubric rating: {'/'.join(key)}")
                seen.add(key)
                rubric_records.append({
                    "review_id": row["review_id"],
                    "dimension": dimension,
                    "reviewer_id": row["reviewer_id"],
                    "rating": rating,
                })

    claim_records = []
    seen = set()
    with (output_dir / "claim_support_ratings.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            if not row.get("reviewer_id"):
                continue
            if row.get("review_id") not in valid_review_ids:
                raise ValueError(f"unknown review_id in claim form: {row.get('review_id')}")
            rating = row.get("rating", "").lower()
            if rating not in {"supported", "contradicted", "unsupported"}:
                raise ValueError(
                    f"{row.get('review_id')}/{row.get('claim_index')}: invalid claim rating"
                )
            try:
                claim_index = int(row["claim_index"])
            except (TypeError, ValueError) as exc:
                raise ValueError("claim_index must be an integer") from exc
            key = (row["review_id"], claim_index, row["reviewer_id"])
            if key in seen:
                raise ValueError(f"duplicate claim rating: {key}")
            seen.add(key)
            claim_records.append({
                "review_id": row["review_id"],
                "claim_index": claim_index,
                "claim_id": row.get("claim_id", ""),
                "reviewer_id": row["reviewer_id"],
                "rating": rating,
            })
    return rubric_records, claim_records


def _judge_dimension(row, dimension):
    value = row.get(f"judge_{dimension}")
    if value is None:
        value = row.get("judge_dimensions", {}).get(dimension, {}).get("score")
    return value


def calculate_judge_human_agreement(
    mapping, linked, rubric_records, claim_records, bootstrap_samples=1000, seed=42,
):
    rubric_groups = defaultdict(list)
    for record in rubric_records:
        rubric_groups[(record["review_id"], record["dimension"])].append(record["rating"])

    details = []
    dimensions = {}
    for dimension_index, dimension in enumerate(RUBRIC_DIMENSIONS):
        pairs = []
        human_rated = 0
        judge_scored = 0
        for item in mapping:
            review_id = item["review_id"]
            ratings = rubric_groups.get((review_id, dimension), [])
            judge_score = _judge_dimension(linked[review_id], dimension)
            human_rated += bool(ratings)
            judge_scored += judge_score is not None
            if ratings and judge_score is not None:
                pair = {
                    "review_id": review_id,
                    "judge": float(judge_score),
                    "human_mean": mean(ratings),
                    "human_median": median(ratings),
                    "human_ordinal": _unique_mode(ratings),
                    "reviewer_count": len(ratings),
                }
                pairs.append(pair)
                details.append({"dimension": dimension, **pair})
        dimensions[dimension] = _agreement_summary(
            pairs, len(mapping), bootstrap_samples, seed + dimension_index * 10
        )
        dimensions[dimension].update({
            "human_rated_items": human_rated,
            "judge_scored_items": judge_scored,
        })

    overall_pairs = []
    for item in mapping:
        review_id = item["review_id"]
        human_dimensions = []
        for dimension in RUBRIC_DIMENSIONS:
            ratings = rubric_groups.get((review_id, dimension), [])
            if ratings:
                human_dimensions.append(mean(ratings))
        judge_score = linked[review_id].get("judge_overall_mean")
        if len(human_dimensions) == len(RUBRIC_DIMENSIONS) and judge_score is not None:
            overall_pairs.append({
                "review_id": review_id,
                "judge": float(judge_score),
                "human_mean": mean(human_dimensions),
                "human_ordinal": None,
                "reviewer_count": len({
                    record["reviewer_id"] for record in rubric_records
                    if record["review_id"] == review_id
                }),
            })
    overall = _agreement_summary(
        overall_pairs, len(mapping), bootstrap_samples, seed + 100
    )
    overall["human_rated_items"] = sum(
        all(rubric_groups.get((item["review_id"], dimension)) for dimension in RUBRIC_DIMENSIONS)
        for item in mapping
    )
    overall["judge_scored_items"] = sum(
        linked[item["review_id"]].get("judge_overall_mean") is not None for item in mapping
    )
    overall["quadratic_weighted_kappa"] = None
    overall["exact_agreement"] = None
    overall["note"] = "Human overall is the mean of the five dimension consensus means."

    claim_lookup = {
        (item["review_id"], claim["claim_index"]): claim
        for item in mapping for claim in item["claims"]
    }
    claim_groups = defaultdict(list)
    for record in claim_records:
        key = (record["review_id"], record["claim_index"])
        if key not in claim_lookup:
            raise ValueError(f"unknown claim rating target: {key}")
        expected_id = claim_lookup[key]["claim_id"]
        if record.get("claim_id") and record["claim_id"] != expected_id:
            raise ValueError(f"{key}: claim identity mismatch")
        claim_groups[key].append(record["rating"] == "supported")

    claim_pairs = []
    eligible_claims = sum(len(item["claims"]) for item in mapping)
    consensus_ties = 0
    for item in mapping:
        review_id = item["review_id"]
        scores = linked[review_id].get("claim_scores", [])
        for claim in item["claims"]:
            index = claim["claim_index"]
            ratings = claim_groups.get((review_id, index), [])
            if not ratings:
                continue
            consensus = _unique_mode(ratings)
            if consensus is None:
                consensus_ties += 1
                continue
            automated = scores[index].get("supported")
            if not isinstance(automated, bool):
                continue
            claim_pairs.append({
                "review_id": review_id,
                "claim_index": index,
                "claim_id": claim["claim_id"],
                "automated_supported": automated,
                "human_supported": consensus,
                "reviewer_count": len(ratings),
            })
    tp = sum(pair["automated_supported"] and pair["human_supported"] for pair in claim_pairs)
    fp = sum(pair["automated_supported"] and not pair["human_supported"] for pair in claim_pairs)
    fn = sum(not pair["automated_supported"] and pair["human_supported"] for pair in claim_pairs)
    tn = sum(not pair["automated_supported"] and not pair["human_supported"] for pair in claim_pairs)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0 if precision is not None and recall is not None else None
    )
    claim_kappa = _weighted_kappa(
        [pair["automated_supported"] for pair in claim_pairs],
        [pair["human_supported"] for pair in claim_pairs],
        (False, True),
    )
    claim_summary = {
        "eligible_claims": eligible_claims,
        "paired_claims": len(claim_pairs),
        "unscored_claims": eligible_claims - len(claim_pairs),
        "consensus_ties": consensus_ties,
        "cohen_kappa": round(claim_kappa, 6) if claim_kappa is not None else None,
        "accuracy": round((tp + tn) / len(claim_pairs), 6) if claim_pairs else None,
        "precision_supported": round(precision, 6) if precision is not None else None,
        "recall_supported": round(recall, 6) if recall is not None else None,
        "f1_supported": round(f1, 6) if f1 is not None else None,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "label_mapping": {
            "supported": True,
            "contradicted": False,
            "unsupported": False,
        },
    }
    return {
        "dimensions": dimensions,
        "overall": overall,
        "claim_support": claim_summary,
    }, details, claim_pairs


def _write_agreement_details(output_dir, rubric_details, claim_details):
    rubric_fields = [
        "review_id", "dimension", "judge", "human_mean", "human_median",
        "human_ordinal", "reviewer_count",
    ]
    with (output_dir / "judge_human_agreement.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rubric_fields)
        writer.writeheader()
        writer.writerows(rubric_details)
    claim_fields = [
        "review_id", "claim_index", "claim_id", "automated_supported",
        "human_supported", "reviewer_count",
    ]
    with (output_dir / "claim_human_agreement.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=claim_fields)
        writer.writeheader()
        writer.writerows(claim_details)


def calculate_completed_forms(
    output_dir, evaluations=None, bootstrap_samples=1000, seed=42,
):
    """Validate completed forms and optionally link them to saved evaluator records."""
    output_dir = Path(output_dir)
    mapping = json.loads((output_dir / "review_mapping.private.json").read_text(encoding="utf-8"))
    valid_review_ids = {item["review_id"] for item in mapping}
    if len(valid_review_ids) != len(mapping):
        raise ValueError("duplicate review_id in private mapping")
    rubric_records, claim_records = _read_completed_forms(output_dir, valid_review_ids)
    report = {
        "schema_version": AGREEMENT_SCHEMA_VERSION,
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
        "judge_vs_human": None,
    }
    if evaluations is not None:
        evaluation_rows = _load_evaluation_rows(evaluations)
        linked = _link_evaluations(mapping, evaluation_rows)
        agreement, rubric_details, claim_details = calculate_judge_human_agreement(
            mapping,
            linked,
            rubric_records,
            claim_records,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        report["judge_vs_human"] = agreement
        report["linkage"] = {
            "mapped_reviews": len(mapping),
            "evaluation_rows_considered": len(evaluation_rows),
            "evaluator_hashes": sorted({item["evaluator_hash"] for item in mapping}),
        }
        _write_agreement_details(output_dir, rubric_details, claim_details)
    (output_dir / "agreement.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("review_directory")
    parser.add_argument("--evaluations")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    print(json.dumps(calculate_completed_forms(
        arguments.review_directory,
        evaluations=arguments.evaluations,
        bootstrap_samples=arguments.bootstrap_samples,
        seed=arguments.seed,
    ), indent=2))
