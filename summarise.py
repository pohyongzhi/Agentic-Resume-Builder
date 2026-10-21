"""Turns results.jsonl into one table: the average of each metric per workflow.

    python summarise.py      -> prints the table and writes results/summary.csv
"""
import csv
import json
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
    rows = defaultdict(list)
    with open(config.RESULTS_FILE, encoding="utf-8") as f:
        for r in map(json.loads, f):
            if not r.get("error"):
                rows[r["workflow"]].append(r)

    table = []
    for wf, rs in rows.items():
        line = {"workflow": wf, "runs": len(rs),
                "finished_%": f"{100 * sum(r['finished'] for r in rs) / len(rs):.0f}"}
        for m in METRICS:
            line[m] = mean_sd([r.get(m) for r in rs])
        table.append(line)

    with open("results/summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    for line in table:
        print(line)


if __name__ == "__main__":
    main()
