"""Turns results.jsonl into one table: the average of each metric per workflow.

    python summarise.py      -> prints the table and writes results/summary.csv

success_% counts every run, including errored ones, so a workflow that crashes or stops on hard
cases can't look better than one that tries. The metric averages are over successful runs only.
"""
import csv
import json
import os
import statistics as st
from collections import defaultdict

import config

METRICS = ["sim_after", "sim_gain", "kw_after", "faithfulness", "input_tokens", "output_tokens",
           "wall_seconds", "tool_calls", "tool_success", "tool_calls_repeated"]


def mean_sd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return ""
    return f"{st.mean(vals):.3f}" + (f" ± {st.stdev(vals):.3f}" if len(vals) > 1 else "")


def main():
    if not os.path.exists(config.RESULTS_FILE):
        raise SystemExit(f"No results yet: {config.RESULTS_FILE} not found. Run run_all.py first.")
    runs = defaultdict(list)
    with open(config.RESULTS_FILE, encoding="utf-8") as f:
        for r in map(json.loads, f):
            runs[r["workflow"]].append(r)
    if not runs:
        raise SystemExit("results.jsonl is empty. Run run_all.py first.")

    table = []
    for wf, all_rs in runs.items():
        ok = [r for r in all_rs if not r.get("error")]
        failed = [r for r in all_rs if r.get("error")]
        codes = defaultdict(int)
        for r in failed:
            codes[r["error"].split(":", 1)[0]] += 1
        line = {"workflow": wf, "runs": len(all_rs),
                "success_%": f"{100 * len(ok) / len(all_rs):.0f}",
                "errors": "; ".join(f"{c} x{n}" for c, n in sorted(codes.items()))}
        for m in METRICS:
            line[m] = mean_sd([r.get(m) for r in ok])
        table.append(line)

    with open(os.path.join(os.path.dirname(config.RESULTS_FILE), "summary.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    for line in table:
        print(line)


if __name__ == "__main__":
    main()
