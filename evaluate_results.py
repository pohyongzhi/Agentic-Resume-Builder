"""Rescore saved outputs without rerunning expensive generation workflows.

Usage:
    python evaluate_results.py
    python evaluate_results.py --recompute-quality
"""
import argparse
import json
from pathlib import Path

import config
from common.data import load_pairs
from eval import metrics
from eval.operational import evaluate_operational


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _quality(pair, markdown, skip_faithfulness):
    row = {"words": len(markdown.split())}
    row.update(metrics.jd_similarity(pair["resume"], markdown, pair["jd"]))
    before = metrics.keyword_coverage(pair["resume"], pair["keywords"])
    after = metrics.keyword_coverage(markdown, pair["keywords"])
    row.update({
        "kw_before": round(before["coverage"], 4),
        "kw_after": round(after["coverage"], 4),
        "kw_missing": after["missing"],
    })
    row.update(metrics.protected_field_metrics(pair["resume"], markdown))
    row.update(metrics.fabrication_bait_metrics(
        pair["resume"], markdown, pair.get("keywords", []),
    ))
    if not skip_faithfulness:
        row.update(metrics.faithfulness(pair["resume"], markdown))
    return row


def evaluate(results_file, output_file, recompute_quality=False, skip_faithfulness=False):
    pairs = {pair["pair_id"]: pair for pair in load_pairs()}
    results_file = Path(results_file)
    rows = [json.loads(line) for line in results_file.read_text(encoding="utf-8").splitlines() if line]
    evaluated = []

    for row in rows:
        pair = pairs.get(row["pair_id"])
        if pair is None:
            raise KeyError(f"result references unknown pair_id: {row['pair_id']}")
        resume_path = Path(row["resume_path"])
        trace_path = Path(row["trace_path"])
        markdown = resume_path.read_text(encoding="utf-8") if resume_path.exists() else ""
        trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else []
        if recompute_quality and markdown:
            row.update(_quality(pair, markdown, skip_faithfulness))

        output = {
            "resume_md": markdown,
            "finished": not row.get("error") and bool(markdown.strip()),
            "hit_budget": row.get("hit_budget", False),
            "errors": [row["error"]] if row.get("error") else [],
        }
        row.update(evaluate_operational(
            row["workflow"],
            pair,
            output,
            row,
            trace,
            scored_row=row,
            error=row.get("error"),
        ))
        evaluated.append(row)

    _atomic_text(
        output_file,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in evaluated),
    )
    return evaluated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", default=config.RESULTS_FILE)
    parser.add_argument(
        "--output-file",
        default=str(Path(config.RESULTS_FILE).with_name("evaluated_results.jsonl")),
    )
    parser.add_argument("--recompute-quality", action="store_true")
    parser.add_argument("--skip-faithfulness", action="store_true")
    args = parser.parse_args()
    rows = evaluate(
        args.results_file,
        args.output_file,
        recompute_quality=args.recompute_quality,
        skip_faithfulness=args.skip_faithfulness,
    )
    print(f"Evaluated {len(rows)} runs -> {args.output_file}")
