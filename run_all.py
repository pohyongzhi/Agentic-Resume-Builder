"""Runs every workflow on every pair, scores it, and appends one JSON line per run.

    python run_all.py                         # everything, 2 repeats
    python run_all.py --workflows one_pass    # just one variant
    python run_all.py --limit 2 --repeats 1   # quick test on 2 pairs

Safe to stop and restart: runs already in results.jsonl are skipped.
Workflows run one after another (not in parallel) so they don't fight over the GPU,
otherwise the timing numbers would be meaningless.
"""
import argparse
import json
import os
import time
import traceback

import config
from common.data import load_pairs
from common.llm import Tracker
from eval import metrics
from workflows import WORKFLOWS, WORKFLOW_SPECS


def done_keys():
    if not os.path.exists(config.RESULTS_FILE):
        return set()
    with open(config.RESULTS_FILE, encoding="utf-8") as f:
        return {
            (r["workflow"], r["pair_id"], r["repeat"], r.get("model"))
            for r in map(json.loads, f)
        }


def actual_model(tracker):
    models = {event["model"] for event in tracker.events if event["type"] == "llm"}
    if len(models) != 1:
        raise RuntimeError(f"Expected one writer model per workflow run, found {sorted(models)}")
    return models.pop()


def score(pair, out, tracker, skip_faithfulness=False):
    totals = tracker.totals()
    row = {**totals,
           "tool_success": metrics.tool_success(totals),
           "hit_budget": out["hit_budget"], "finished": out["finished"],
           "words": len(out["resume_md"].split())}
    row.update(metrics.jd_similarity(pair["resume"], out["resume_md"], pair["jd"]))
    kw_before = metrics.keyword_coverage(pair["resume"], pair["keywords"])
    kw_after = metrics.keyword_coverage(out["resume_md"], pair["keywords"])
    row.update({"kw_before": round(kw_before["coverage"], 4), "kw_after": round(kw_after["coverage"], 4),
                "kw_missing": kw_after["missing"]})
    if not skip_faithfulness:
        row.update(metrics.faithfulness(pair["resume"], out["resume_md"]))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflows", nargs="*", default=list(WORKFLOWS))
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="only the first N pairs")
    ap.add_argument("--skip-faithfulness", action="store_true")
    args = ap.parse_args()

    unavailable = [workflow for workflow in args.workflows if workflow not in WORKFLOWS]
    if unavailable:
        states = {
            workflow: WORKFLOW_SPECS[workflow].status
            for workflow in unavailable if workflow in WORKFLOW_SPECS
        }
        ap.error(f"workflows are unavailable or unknown: {unavailable}; known states: {states}")

    pairs = load_pairs()[: args.limit]
    done = done_keys()
    os.makedirs(os.path.dirname(config.RESULTS_FILE), exist_ok=True)

    for wf in args.workflows:
        expected_model = WORKFLOW_SPECS[wf].model
        for pair in pairs:
            for rep in range(args.repeats):
                if (wf, pair["pair_id"], rep, expected_model) in done:
                    continue
                tracker = Tracker()
                print(f"[{wf}] {pair['pair_id']} repeat {rep} ...", end=" ", flush=True)
                t0 = time.time()
                try:
                    out = WORKFLOWS[wf](pair, tracker)
                    used_model = actual_model(tracker)
                    if used_model != expected_model:
                        raise RuntimeError(
                            f"{wf} routed to {used_model}, expected {expected_model}")
                    row = score(pair, out, tracker, args.skip_faithfulness)
                    row["error"] = None
                except Exception as e:
                    out = {"resume_md": ""}
                    row = {**tracker.totals(), "error": f"{type(e).__name__}: {e}"}
                    traceback.print_exc()
                row.update({"workflow": wf, "pair_id": pair["pair_id"], "repeat": rep,
                            "model": expected_model, "wall_seconds": round(time.time() - t0, 1),
                            "resume_md": out["resume_md"], "trace": tracker.events})
                with open(config.RESULTS_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print("ok" if not row["error"] else "ERROR", f"({row['wall_seconds']}s)")


if __name__ == "__main__":
    main()
