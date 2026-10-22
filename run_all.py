"""Runs every workflow on every pair, scores it, and appends one JSON line per run.

    python run_all.py                         # everything, 2 repeats
    python run_all.py --workflows w2_baseline # just one workflow
    python run_all.py --limit 2 --repeats 1   # quick test on 2 pairs

Safe to stop and restart: runs already in results.jsonl with the same model and config are skipped.
Changing config.py or the shared prompts changes the config hash, so those runs are redone.
Workflows run one after another (not in parallel) so they don't fight over the GPU,
otherwise the timing numbers would be meaningless.
"""
import argparse
import hashlib
import json
import os
import time
import traceback

import config
from common import prompts
from common.data import load_pairs
from common.llm import Tracker
from eval import metrics
from workflows import WORKFLOWS


def config_hash():
    settings = {k: getattr(config, k) for k in dir(config) if k.isupper()}
    settings["prompts"] = [
        prompts.BASE_RULES,
        prompts.CREATOR_SYSTEM,
        prompts.STRUCTURED_CREATOR_SYSTEM,
    ]
    return hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:12]


def done_keys():
    if not os.path.exists(config.RESULTS_FILE):
        return set()
    with open(config.RESULTS_FILE, encoding="utf-8") as f:
        return {(r["workflow"], r["pair_id"], r["repeat"], r.get("model"), r.get("config_hash"))
                for r in map(json.loads, f)}


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

    pairs = load_pairs()[: args.limit]
    done = done_keys()
    chash = config_hash()
    os.makedirs(os.path.dirname(config.RESULTS_FILE), exist_ok=True)

    for wf in args.workflows:
        for pair in pairs:
            for rep in range(args.repeats):
                if (wf, pair["pair_id"], rep, config.MODEL, chash) in done:
                    continue
                tracker = Tracker()
                print(f"[{wf}] {pair['pair_id']} repeat {rep} ...", end=" ", flush=True)
                t0 = time.time()
                try:
                    out = WORKFLOWS[wf](pair, tracker)
                    if out.get("errors") or not out["resume_md"].strip():
                        # the workflow stopped with a clear error (e.g. TOOL_BUDGET_EXCEEDED): a failed run
                        row = {**tracker.totals(), "hit_budget": out.get("hit_budget", False), "finished": False,
                               "error": "; ".join(out.get("errors") or ["EMPTY_OUTPUT: no resume produced"])}
                    else:
                        row = score(pair, out, tracker, args.skip_faithfulness)
                        row["error"] = None
                except Exception as e:
                    out = {"resume_md": ""}
                    row = {**tracker.totals(), "error": f"{type(e).__name__}: {e}"}
                    traceback.print_exc()
                row.update({"workflow": wf, "pair_id": pair["pair_id"], "repeat": rep,
                            "model": config.MODEL, "config_hash": chash,
                            "wall_seconds": round(time.time() - t0, 1),
                            "resume_md": out["resume_md"], "trace": tracker.events})
                with open(config.RESULTS_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print("ok" if not row["error"] else "ERROR", f"({row['wall_seconds']}s)")


if __name__ == "__main__":
    main()
