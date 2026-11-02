"""Run all workflows on meaning-preserving input variants for a small held-out subset."""
import argparse
import json
import time
import traceback
from pathlib import Path

import config
import run_all
from common.data import dataset_hash, load_pairs
from common.llm import Tracker
from eval.operational import evaluate_operational
from eval.robustness import (
    VARIANTS,
    perturb_pair,
    robustness_manifest,
    select_stratified_pairs,
    summarise_robustness,
    write_robustness_csv,
)
from workflows import WORKFLOWS


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _done(path):
    if not path.exists():
        return set()
    return {
        (row["workflow"], row["base_pair_id"], row["variant"],
         row["config_hash"], row["dataset_hash"])
        for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflows", nargs="*", default=list(WORKFLOWS))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--skip-faithfulness", action="store_true")
    args = parser.parse_args()

    all_pairs = load_pairs(split="test")
    pairs = select_stratified_pairs(all_pairs, config.ROBUSTNESS_SEED)[:args.limit]
    chash, dhash = run_all.config_hash(), dataset_hash()
    result_dir = Path(config.RESULTS_FILE).parent
    _atomic_text(
        result_dir / "robustness_sample.json",
        json.dumps(robustness_manifest(all_pairs, dhash, config.ROBUSTNESS_SEED), indent=2) + "\n",
    )
    results_path = result_dir / "robustness.jsonl"
    done = _done(results_path)

    for workflow in args.workflows:
        for base_pair in pairs:
            for variant in args.variants:
                key = (workflow, base_pair["pair_id"], variant, chash, dhash)
                if key in done:
                    continue
                resume, jd = perturb_pair(base_pair, variant)
                run_dir = (
                    result_dir / "robustness_runs" / chash / workflow
                    / base_pair["pair_id"] / variant
                )
                input_resume = run_dir / "input_resume.md"
                _atomic_text(input_resume, resume)
                _atomic_text(run_dir / "input_jd.md", jd)
                pair = {
                    **base_pair,
                    "pair_id": f'{base_pair["pair_id"]}__{variant}',
                    "resume_path": str(input_resume),
                    "resume": resume,
                    "jd": jd,
                }
                tracker = Tracker()
                started = time.time()
                try:
                    output = WORKFLOWS[workflow](pair, tracker)
                    if output.get("errors") or not output.get("resume_md", "").strip():
                        row = {
                            **tracker.totals(),
                            "hit_budget": output.get("hit_budget", False),
                            "finished": False,
                            "error": "; ".join(output.get("errors") or ["EMPTY_OUTPUT"]),
                        }
                    else:
                        row = run_all.score(
                            pair, output, tracker, skip_faithfulness=args.skip_faithfulness
                        )
                        row["error"] = None
                except Exception as exc:
                    output = {"resume_md": "", "finished": False, "hit_budget": False}
                    row = {**tracker.totals(), "error": f"{type(exc).__name__}: {exc}"}
                    traceback.print_exc()

                row.update(evaluate_operational(
                    workflow, base_pair, output, tracker.totals(), tracker.events,
                    scored_row=row, error=row.get("error"),
                ))
                output_path = run_dir / "resume.md"
                trace_path = run_dir / "trace.json"
                _atomic_text(output_path, output.get("resume_md", ""))
                _atomic_text(trace_path, json.dumps(tracker.events, indent=2) + "\n")
                row.update({
                    "workflow": workflow,
                    "base_pair_id": base_pair["pair_id"],
                    "pair_id": pair["pair_id"],
                    "category": base_pair["category"],
                    "fit_level": base_pair["fit_level"],
                    "variant": variant,
                    "config_hash": chash,
                    "dataset_hash": dhash,
                    "wall_seconds": round(time.time() - started, 1),
                    "resume_path": str(output_path),
                    "trace_path": str(trace_path),
                })
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[{workflow}] {base_pair['pair_id']} {variant}: "
                      f"{'ok' if not row['error'] else 'ERROR'}")

    rows = [
        row for row in map(json.loads, results_path.read_text(encoding="utf-8").splitlines())
        if row.get("config_hash") == chash and row.get("dataset_hash") == dhash
    ]
    summary = summarise_robustness(rows)
    write_robustness_csv(result_dir / "summaries" / "robustness.csv", summary)
    print(f"Wrote {len(summary)} workflow summaries.")


if __name__ == "__main__":
    main()
