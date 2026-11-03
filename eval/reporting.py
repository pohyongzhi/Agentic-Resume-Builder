"""Transparent summaries with explicit denominators and deterministic confidence intervals.

Stage-specific report files are produced by ``STAGE_REPORTERS``: a mapping from the stage name
used in ``evaluate.STAGES`` to ``writer(output_dir, rows) -> dict[str, list[dict]]``.
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

import config
from eval.injection import summarise_injection
from eval.pairwise import summarise_pairwise

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
    "requirement_coverage",
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
    if metric == "requirement_coverage" and (
        row.get("requirement_coverage_status") == "no_supportable_requirements"
    ):
        # No supportable requirement means the metric is undefined, not failed or zero.
        return False
    return not bool(row.get("error"))


def summarise(rows, group_fields=(), metrics=METRICS):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)
    output = []
    for group_key, group_rows in sorted(grouped.items()):
        group = dict(zip(group_fields, group_key))
        for metric in metrics:
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


def _write_pair(output_dir, name, table):
    """Write ``name.csv`` and ``name.json``; the JSON is written even for empty tables."""
    write_table(output_dir / f"{name}.csv", table)
    _atomic_text(
        output_dir / f"{name}.json",
        json.dumps(table, indent=2, ensure_ascii=False) + "\n",
    )


def requirement_coverage_summary(rows, group_fields=("dataset_hash", "workflow")):
    """Coverage status accounting plus macro and micro coverage per group.

    ``outputs`` counts every row; ``generation_failures`` and ``not_applicable`` (no supportable
    requirement) are excluded from ``eligible``. ``scored + unscored == eligible``; ``failed`` is
    the subset of unscored rows with a recorded evaluator failure.
    """
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)
    output = []
    for group_key, group_rows in sorted(grouped.items()):
        generation_failures = [row for row in group_rows if row.get("error")]
        not_applicable = [
            row for row in group_rows
            if not row.get("error") and not _eligible(row, "requirement_coverage")
        ]
        eligible = [
            row for row in group_rows
            if not row.get("error") and _eligible(row, "requirement_coverage")
        ]
        scored = [row for row in eligible if row.get("requirement_coverage") is not None]
        failed = [
            row for row in eligible
            if row.get("requirement_coverage") is None
            and "requirement_coverage" in (row.get("evaluation_failures") or {})
        ]
        values = [float(row["requirement_coverage"]) for row in scored]
        low, high = _bootstrap_ci(
            values, config.REPORT_BOOTSTRAP_SEED, config.REPORT_BOOTSTRAP_SAMPLES
        )
        supportable = sum(row.get("requirement_coverage_supportable_count") or 0 for row in scored)
        covered = sum(row.get("requirement_coverage_covered_count") or 0 for row in scored)
        output.append({
            **dict(zip(group_fields, group_key)),
            "outputs": len(group_rows),
            "generation_failures": len(generation_failures),
            "not_applicable": len(not_applicable),
            "eligible": len(eligible),
            "scored": len(scored),
            "unscored": len(eligible) - len(scored),
            "failed": len(failed),
            "mean_coverage": round(st.mean(values), 6) if values else None,
            "ci_95_low": round(low, 6) if low is not None else None,
            "ci_95_high": round(high, 6) if high is not None else None,
            "supportable_requirements": supportable,
            "covered_requirements": covered,
            "micro_coverage": round(covered / supportable, 6) if supportable else None,
        })
    return output


INJECTION_DIAGNOSTIC_METRICS = (
    "faithfulness",
    "claim_faithfulness",
    "unsupported_claim_rate",
    "protected_field_violation",
    "fabrication_bait_violation",
)


def injection_summary(rows):
    """Attack success rates with explicit eligible/scored/unscored/failed denominators.

    Every attempted attack is eligible; only completed generations are scored. Generation
    failures are counted as ``failed`` and never as a resisted or successful attack.
    """
    output = []
    for row in summarise_injection(rows):
        output.append({
            **row,
            "eligible": row["attempted"],
            "scored": row["completed"],
            "unscored": row["attempted"] - row["completed"],
            "failed": row["generation_failures"],
        })
    return output


def write_robustness_reports(output_dir, rows):
    reports = {
        "robustness": paired_delta_summary(rows),
        "robustness_status": summarise(rows, ("workflow", "variant")),
    }
    for name, table in reports.items():
        _write_pair(output_dir, name, table)
    return reports


def write_injection_reports(output_dir, rows):
    reports = {
        "injection": injection_summary(rows),
        "injection_diagnostics": summarise(
            rows, ("workflow", "channel"), INJECTION_DIAGNOSTIC_METRICS,
        ),
    }
    for name, table in reports.items():
        _write_pair(output_dir, name, table)
    return reports


def pairwise_summary(records, group_fields=()):
    """Pairwise aggregates with nested values flattened to JSON strings for CSV output."""
    flat = []
    for row in summarise_pairwise(list(records), tuple(group_fields)):
        flat.append({
            key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
            for key, value in row.items()
        })
    return flat


def write_pairwise_reports(output_dir, rows):
    reports = {
        "pairwise": pairwise_summary(rows, ("baseline_workflow", "challenger_workflow")),
        "pairwise_by_fit_level": pairwise_summary(
            rows, ("baseline_workflow", "challenger_workflow", "fit_level"),
        ),
    }
    for name, table in reports.items():
        _write_pair(output_dir, name, table)
    return reports


# Stage name (evaluate.STAGES) -> writer(output_dir, rows). Writers run only for non-empty rows.
STAGE_REPORTERS = {
    "robustness": write_robustness_reports,
    "injection": write_injection_reports,
    "pairwise": write_pairwise_reports,
}


def write_reports(output_dir, rows, robustness_rows=(), stage_rows=None):
    output_dir = Path(output_dir)
    reports = {
        "overall": summarise(rows, ("dataset_hash", "workflow")),
        "by_category": summarise(rows, ("dataset_hash", "workflow", "category")),
        "by_fit_level": summarise(rows, ("dataset_hash", "workflow", "fit_level")),
        "requirement_coverage": requirement_coverage_summary(rows),
        "requirement_coverage_by_fit_level": requirement_coverage_summary(
            rows, ("dataset_hash", "workflow", "fit_level"),
        ),
    }
    for name, table in reports.items():
        write_table(output_dir / f"{name}.csv", table)
        _atomic_text(
            output_dir / f"{name}.json",
            json.dumps(table, indent=2, ensure_ascii=False) + "\n",
        )
    stage_rows = dict(stage_rows or {})
    if robustness_rows and not stage_rows.get("robustness"):
        stage_rows["robustness"] = robustness_rows
    for name, rows_for_stage in stage_rows.items():
        writer = STAGE_REPORTERS.get(name)
        if writer and rows_for_stage:
            reports.update(writer(output_dir, list(rows_for_stage)))
    return reports
