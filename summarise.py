"""Turns results.jsonl into one table: the average of each metric per workflow.

    python summarise.py
    python summarise.py --results-file results/evaluated_results.jsonl

success_% counts every run, including errored ones, so a workflow that crashes or stops on hard
cases can't look better than one that tries. The metric averages are over successful runs only.
"""
import argparse
import csv
import json
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

import config
from eval.reporting import write_reports

METRICS = [
    "sim_after", "sim_gain", "kw_after", "faithfulness",
    "required_sections_rate", "step_efficiency", "budget_utilisation",
    "tool_call_accuracy", "tool_argument_accuracy", "tool_sequence_valid", "tool_recovery_rate",
    "input_tokens", "output_tokens", "total_tokens", "wall_seconds",
    "tokens_per_successful_task", "seconds_per_successful_task", "cost_quality_ratio",
    "tool_calls", "tool_success", "tool_calls_repeated",
]


def mean_sd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return ""
    return f"{st.mean(vals):.3f}" + (f" ± {st.stdev(vals):.3f}" if len(vals) > 1 else "")


def _table(rows, group_fields):
    runs = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field, "") for field in group_fields)
        runs[key].append(row)
    table = []
    for key, all_rs in sorted(runs.items()):
        ok = [r for r in all_rs if not r.get("error")]
        failed = [r for r in all_rs if r.get("error")]
        codes = defaultdict(int)
        for r in failed:
            codes[r["error"].split(":", 1)[0]] += 1
        line = {field: value for field, value in zip(group_fields, key)}
        task_values = [r.get("task_success") for r in all_rs if r.get("task_success") is not None]
        line.update({"runs": len(all_rs),
                "success_%": f"{100 * len(ok) / len(all_rs):.0f}",
                "task_success_%": (
                    f"{100 * sum(task_values) / len(task_values):.0f}" if task_values else ""
                ),
                "errors": "; ".join(f"{c} x{n}" for c, n in sorted(codes.items()))})
        for m in METRICS:
            line[m] = mean_sd([r.get(m) for r in ok])
        table.append(line)
    return table


def _write_csv(path, table):
    if not table:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)


def main(results_file=None):
    results_file = results_file or config.RESULTS_FILE
    if not os.path.exists(results_file):
        raise SystemExit(f"No results yet: {results_file} not found. Run run_all.py first.")
    with open(results_file, encoding="utf-8") as f:
        rows = list(map(json.loads, f))
    if not rows:
        raise SystemExit("results.jsonl is empty. Run run_all.py first.")

    result_dir = Path(results_file).parent
    if any(row.get("evaluator_hash") for row in rows):
        write_reports(result_dir / "summaries", rows)
        print(f"Wrote evaluator summaries for {len(rows)} records.")
        return
    overall = _table(rows, ["dataset_version", "dataset_hash", "workflow"])
    by_category = _table(rows, ["dataset_version", "dataset_hash", "workflow", "category"])
    by_fit = _table(rows, ["dataset_version", "dataset_hash", "workflow", "fit_level"])
    _write_csv(result_dir / "summaries" / "overall.csv", overall)
    _write_csv(result_dir / "summaries" / "by_category.csv", by_category)
    _write_csv(result_dir / "summaries" / "by_fit_level.csv", by_fit)
    _write_csv(result_dir / "summary.csv", overall)  # backwards-compatible path
    for line in overall:
        print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", default=config.RESULTS_FILE)
    args = parser.parse_args()
    main(args.results_file)
