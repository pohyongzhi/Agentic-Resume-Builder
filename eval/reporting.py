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
from collections import Counter, defaultdict
from pathlib import Path

import config
from eval.injection import summarise_injection
from eval.pairwise import summarise_pairwise
from eval.robustness import (
    DELTA_METRICS,
    paired_deltas,
    robustness_comparisons,
    stability_assessment,
    variant_family,
    variant_scope,
)

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
            if any(row.get("error") or row.get("finished") is False
                   or ("resume_md" in row and not row["resume_md"].strip())
                   for row in (original, variant)):
                continue
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


def _rate(numerator, denominator):
    return round(numerator / denominator, 6) if denominator else None


def _wilson_fields(prefix, successes, total):
    low, high = _wilson_ci(successes, total)
    return {
        f"{prefix}_ci_95_low": round(low, 6) if low is not None else None,
        f"{prefix}_ci_95_high": round(high, 6) if high is not None else None,
    }


def robustness_delta_table(rows):
    """In-scope paired deltas per workflow x variant x metric (variant minus original).

    ``runs``: variant runs; ``original_unavailable``: runs whose original is missing or failed;
    ``eligible``: runs whose original is scored on the metric; ``failed``: eligible runs whose
    variant generation failed; ``scored``: eligible runs with a variant score (a paired delta);
    ``unscored = eligible - scored`` (generation failures plus evaluation failures, never zero).
    ``worst_degradation`` follows the metric direction (largest harm, positive = worse).
    """
    grouped = defaultdict(list)
    for comparison in robustness_comparisons(rows):
        grouped[(comparison["workflow"], comparison["family"], comparison["variant"])].append(
            comparison
        )
    output = []
    for (workflow, family, variant), comparisons in sorted(grouped.items()):
        for metric, direction in DELTA_METRICS.items():
            usable = [
                item for item in comparisons
                if item["original"] is not None and not item["original"].get("error")
            ]
            eligible = [item for item in usable if item["original"].get(metric) is not None]
            failed = [item for item in eligible if item["row"].get("error")]
            deltas = [
                paired_deltas(item["original"], item["row"])[metric] for item in eligible
            ]
            deltas = [value for value in deltas if value is not None]
            evaluation_failures = [
                item for item in eligible
                if not item["row"].get("error")
                and metric in (item["row"].get("evaluation_failures") or {})
            ]
            low, high = _bootstrap_ci(
                deltas, config.REPORT_BOOTSTRAP_SEED, config.REPORT_BOOTSTRAP_SAMPLES
            )
            output.append({
                "workflow": workflow,
                "family": family,
                "variant": variant,
                "metric": metric,
                "direction": "higher_is_better" if direction > 0 else "lower_is_better",
                "runs": len(comparisons),
                "original_unavailable": len(comparisons) - len(usable),
                "eligible": len(eligible),
                "scored": len(deltas),
                "unscored": len(eligible) - len(deltas),
                "failed": len(failed),
                "evaluation_failures": len(evaluation_failures),
                "mean_delta": round(st.mean(deltas), 6) if deltas else None,
                "mean_absolute_change": (
                    round(st.mean(abs(value) for value in deltas), 6) if deltas else None
                ),
                "worst_degradation": (
                    round(max(-direction * value for value in deltas), 6) if deltas else None
                ),
                "ci_95_low": round(low, 6) if low is not None else None,
                "ci_95_high": round(high, 6) if high is not None else None,
            })
    return output


def _in_scope_stats(comparisons):
    usable = [
        item for item in comparisons
        if item["original"] is not None and not item["original"].get("error")
    ]
    failed = [item for item in usable if item["row"].get("error")]
    outcomes = [item["row"].get("generation_outcome") for item in failed]
    assessments = [
        stability_assessment(item["original"], item["row"])
        for item in usable if not item["row"].get("error")
    ]
    scored = [assessment for assessment in assessments if assessment["stable"] is not None]
    stable = sum(assessment["stable"] for assessment in scored)
    record = {
        "runs": len(comparisons),
        "original_unavailable": len(comparisons) - len(usable),
        "eligible": len(usable),
        "failed": len(failed),
        "crashed": outcomes.count("crashed"),
        "handled_errors": len(failed) - outcomes.count("crashed"),
        "execution_retention": _rate(len(usable) - len(failed), len(usable)),
        "stability_eligible": len(assessments),
        "scored": len(scored),
        "unscored": len(usable) - len(scored),
        "stable": stable,
        "stability_score": _rate(stable, len(scored)),
        **_wilson_fields("stability", stable, len(scored)),
        "stability_tolerance": config.ROBUSTNESS_STABILITY_TOLERANCE,
    }
    violations = Counter(
        metric for assessment in scored for metric in assessment["violations"]
    )
    record["instability_reasons"] = "; ".join(
        f"{metric} x{count}" for metric, count in sorted(violations.items())
    )
    for metric in DELTA_METRICS:
        deltas = [paired_deltas(item["original"], item["row"])[metric] for item in usable]
        deltas = [value for value in deltas if value is not None]
        record[f"{metric}_paired_n"] = len(deltas)
        record[f"{metric}_mean_delta"] = round(st.mean(deltas), 6) if deltas else None
    return record


def _out_of_scope_stats(rows):
    outcomes = Counter(row.get("generation_outcome") for row in rows)
    completed = [row for row in rows if row.get("generation_outcome") == "completed"]
    scored = [row for row in rows if row.get("expected_behaviour_pass") is not None]
    passes = sum(bool(row["expected_behaviour_pass"]) for row in scored)

    def check(field):
        values = [row.get(field) for row in completed if row.get(field) is not None]
        return len(values), sum(not value for value in values)

    fabrication_n, fabrication = check("oos_no_fabrication")
    protected_n, protected = check("oos_protected_fields_preserved")
    unsupported = [
        float(row["unsupported_claim_rate"]) for row in completed
        if row.get("unsupported_claim_rate") is not None
    ]
    crashed = outcomes.get("crashed", 0)
    return {
        "runs": len(rows),
        "eligible": len(rows),
        "scored": len(scored),
        "unscored": len(rows) - len(scored),
        "failed": crashed,
        "completed": len(completed),
        "refused": outcomes.get("refused", 0),
        "handled_errors": outcomes.get("handled_error", 0),
        "crashed": crashed,
        "graceful_handling_rate": _rate(len(rows) - crashed, len(rows)),
        "expected_behaviour_passes": passes,
        "expected_behaviour_pass_rate": _rate(passes, len(scored)),
        **_wilson_fields("expected_behaviour", passes, len(scored)),
        "fabrication_checked": fabrication_n,
        "fabrication_failures": fabrication,
        "fabrication_rate": _rate(fabrication, fabrication_n),
        "protected_fields_checked": protected_n,
        "protected_field_failures": protected,
        "protected_field_violation_rate": _rate(protected, protected_n),
        "fabrication_bait_violations": sum(
            bool(row.get("fabrication_bait_violation")) for row in completed
        ),
        "mean_unsupported_claim_rate": round(st.mean(unsupported), 6) if unsupported else None,
    }


def _robustness_table(rows, key_field):
    """Group by workflow and ``key_field`` (variant or family) into in-scope and OOS tables."""
    in_scope, out_of_scope = [], []
    comparisons = defaultdict(list)
    for comparison in robustness_comparisons(rows):
        comparisons[(comparison["workflow"], comparison[key_field])].append(comparison)
    for (workflow, key), items in sorted(comparisons.items()):
        in_scope.append({
            "workflow": workflow,
            "family": items[0]["family"],
            "variant": key if key_field == "variant" else ",".join(
                sorted({item["variant"] for item in items})
            ),
            **_in_scope_stats(items),
        })
    grouped = defaultdict(list)
    for row in rows:
        if variant_scope(row) == "out_of_scope":
            family = row.get("variant_family") or variant_family(row["variant"])
            grouped[(row["workflow"], row["variant"] if key_field == "variant" else family)].append(
                row
            )
    for (workflow, key), items in sorted(grouped.items()):
        out_of_scope.append({
            "workflow": workflow,
            "family": items[0].get("variant_family") or variant_family(items[0]["variant"]),
            "variant": key if key_field == "variant" else ",".join(
                sorted({row["variant"] for row in items})
            ),
            **_out_of_scope_stats(items),
        })
    return in_scope, out_of_scope


def robustness_variant_tables(rows):
    """Per workflow x variant: (in-scope stability and deltas, out-of-scope expected behaviour)."""
    return _robustness_table(rows, "variant")


def robustness_family_tables(rows):
    """Per workflow x family: pooled versions of ``robustness_variant_tables``."""
    return _robustness_table(rows, "family")


def write_robustness_reports(output_dir, rows):
    in_scope = [row for row in rows if variant_scope(row) == "in_scope"]
    variants, out_of_scope = robustness_variant_tables(rows)
    families, out_of_scope_families = robustness_family_tables(rows)
    reports = {
        "robustness": paired_delta_summary(in_scope),
        "robustness_status": summarise(rows, ("workflow", "variant")),
        "robustness_deltas": robustness_delta_table(rows),
        "robustness_variants": variants,
        "robustness_families": families,
        "robustness_out_of_scope": out_of_scope,
        "robustness_out_of_scope_families": out_of_scope_families,
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
