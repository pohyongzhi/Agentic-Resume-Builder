"""Transparent summaries with explicit denominators and deterministic confidence intervals."""
from __future__ import annotations

import csv
import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

import config

METRICS = (
    "execution_success",
    "task_success",
    "sim_after",
    "sim_gain",
    "kw_after",
    "faithfulness",
    "claim_faithfulness",
    "unsupported_claim_rate",
    "protected_field_violation",
    "protected_field_violation_count",
    "fabrication_bait_violation",
    "fabrication_bait_match_count",
    "judge_overall_mean",
    "judge_relevance_to_jd",
    "judge_prioritisation_of_supported_experience",
    "judge_clarity",
    "judge_concision",
    "judge_organisation_and_readability",
    "required_sections_rate",
    "step_efficiency",
    "total_tokens",
    "generation_model_seconds",
    "historical_run_wall_seconds",
    "nli_seconds",
    "judge_seconds",
)


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _bootstrap_ci(values, seed, samples):
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = sorted(
        st.mean(rng.choices(values, k=len(values)))
        for _ in range(samples)
    )
    low = means[int(0.025 * (samples - 1))]
    high = means[int(0.975 * (samples - 1))]
    return low, high


def _wilson_ci(successes, total, z=1.959963984540054):
    if not total:
        return None, None
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _metric_value(row, metric):
    if metric == "execution_success":
        return not bool(row.get("error"))
    if metric == "generation_model_seconds":
        return row.get("seconds")
    if metric == "historical_run_wall_seconds":
        return row.get("wall_seconds")
    return row.get(metric)


def _eligible(row, metric):
    if metric in {"execution_success", "task_success"}:
        return True
    return not bool(row.get("error"))


def summarise(rows, group_fields=()):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)
    output = []
    for group_key, group_rows in sorted(grouped.items()):
        group = dict(zip(group_fields, group_key))
        for metric in METRICS:
            eligible_rows = [row for row in group_rows if _eligible(row, metric)]
            values = [
                _metric_value(row, metric)
                for row in eligible_rows
                if _metric_value(row, metric) is not None
            ]
            failures = [
                row for row in eligible_rows
                if metric in (row.get("evaluation_failures") or {})
            ]
            numeric = [float(value) for value in values]
            binary = metric in {
                "execution_success",
                "task_success",
                "protected_field_violation",
                "fabrication_bait_violation",
            }
            if binary:
                successes = sum(bool(value) for value in values)
                low, high = _wilson_ci(successes, len(values))
            else:
                successes = None
                low, high = _bootstrap_ci(
                    numeric,
                    config.REPORT_BOOTSTRAP_SEED,
                    config.REPORT_BOOTSTRAP_SAMPLES,
                )
            failure_reasons = defaultdict(int)
            for row in failures:
                failure_reasons[str(row["evaluation_failures"][metric])] += 1
            output.append({
                **group,
                "metric": metric,
                "eligible": len(eligible_rows),
                "scored": len(values),
                "unscored": len(eligible_rows) - len(values),
                "evaluation_failures": len(failures),
                "failure_reasons": "; ".join(
                    f"{reason} x{count}" for reason, count in sorted(failure_reasons.items())
                ),
                "numerator": successes,
                "denominator": len(values) if binary else None,
                "mean": round(st.mean(numeric), 6) if numeric else None,
                "standard_deviation": (
                    round(st.stdev(numeric), 6) if len(numeric) > 1 else None
                ),
                "ci_95_low": round(low, 6) if low is not None else None,
                "ci_95_high": round(high, 6) if high is not None else None,
            })
    return output


def paired_delta_summary(rows, metrics=("faithfulness", "judge_overall_mean", "kw_after", "sim_after")):
    """Summarize original-to-variant changes, resampling complete pair deltas."""
    by_pair = defaultdict(list)
    for row in rows:
        by_pair[(row["workflow"], row["base_pair_id"])].append(row)
    delta_records = []
    for (workflow, _), group in by_pair.items():
        original = next((row for row in group if row["variant"] == "original"), None)
        if not original:
            continue
        for variant in (row for row in group if row["variant"] != "original"):
            for metric in metrics:
                if original.get(metric) is not None and variant.get(metric) is not None:
                    delta_records.append({
                        "workflow": workflow,
                        "variant": variant["variant"],
                        "metric": metric,
                        "category": original.get("category", ""),
                        "fit_level": original.get("fit_level", ""),
                        "delta": float(variant[metric]) - float(original[metric]),
                    })
    output = []
    for scope, field in (("overall", None), ("category", "category"), ("fit_level", "fit_level")):
        deltas = defaultdict(list)
        for record in delta_records:
            stratum = record[field] if field else "all"
            deltas[
                (record["workflow"], record["variant"], record["metric"], stratum)
            ].append(record["delta"])
        for (workflow, variant, metric, stratum), values in sorted(deltas.items()):
            low, high = _bootstrap_ci(
                values, config.REPORT_BOOTSTRAP_SEED, config.REPORT_BOOTSTRAP_SAMPLES
            )
            output.append({
                "scope": scope,
                "stratum": stratum,
                "workflow": workflow,
                "variant": variant,
                "metric": metric,
                "paired_n": len(values),
                "mean_delta": round(st.mean(values), 6),
                "mean_absolute_change": round(st.mean(abs(value) for value in values), 6),
                "worst_degradation": round(min(values), 6),
                "ci_95_low": round(low, 6),
                "ci_95_high": round(high, 6),
            })
    return output


def write_table(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_reports(output_dir, rows, robustness_rows=()):
    output_dir = Path(output_dir)
    reports = {
        "overall": summarise(rows, ("dataset_hash", "workflow")),
        "by_category": summarise(rows, ("dataset_hash", "workflow", "category")),
        "by_fit_level": summarise(rows, ("dataset_hash", "workflow", "fit_level")),
    }
    for name, table in reports.items():
        write_table(output_dir / f"{name}.csv", table)
        _atomic_text(
            output_dir / f"{name}.json",
            json.dumps(table, indent=2, ensure_ascii=False) + "\n",
        )
    if robustness_rows:
        robustness = paired_delta_summary(robustness_rows)
        write_table(output_dir / "robustness.csv", robustness)
        _atomic_text(
            output_dir / "robustness.json",
            json.dumps(robustness, indent=2, ensure_ascii=False) + "\n",
        )
        robustness_status = summarise(robustness_rows, ("workflow", "variant"))
        write_table(output_dir / "robustness_status.csv", robustness_status)
        _atomic_text(
            output_dir / "robustness_status.json",
            json.dumps(robustness_status, indent=2, ensure_ascii=False) + "\n",
        )
    return reports
