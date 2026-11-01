"""Runs every workflow on every pair, scores it, and appends one JSON line per run.

    python run_all.py                         # every test pair, 1 repeat
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
from pathlib import Path
import re
import time
import traceback

import config
from common import prompts
from common.data import dataset_hash, load_manifest, load_pairs
from common.llm import Tracker
from eval import metrics
from eval.operational import evaluate_operational
from workflows import WORKFLOWS


def config_hash():
    """Generation-only identity. Offline evaluator settings must never invalidate writer outputs."""
    generation_keys = (
        "MODEL", "TEMPERATURE", "SEED", "NUM_CTX", "THINK", "MAX_STEPS",
        "MAX_CRITIC_ROUNDS", "MAX_TOOL_CALLS", "PDF_MIN_CHARS_PER_PAGE",
        "JD_MIN_CHARS", "FETCH_TIMEOUT",
    )
    settings = {key: getattr(config, key) for key in generation_keys}
    settings["prompts"] = [
        prompts.BASE_RULES,
        prompts.CREATOR_SYSTEM,
        prompts.STRUCTURED_CREATOR_SYSTEM,
    ]
    return hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:12]


generation_config_hash = config_hash


def done_keys():
    if not os.path.exists(config.RESULTS_FILE):
        return set()
    with open(config.RESULTS_FILE, encoding="utf-8") as f:
        return {(r["workflow"], r["pair_id"], r["repeat"], r.get("model"),
                 r.get("config_hash"), r.get("dataset_hash"))
                for r in map(json.loads, f)}


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _run_dir(manifest, chash, workflow, pair_id, repeat):
    safe = re.compile(r"^[A-Za-z0-9_.-]+$")
    components = (manifest["version"], chash, workflow, pair_id)
    if any(not safe.fullmatch(str(value)) for value in components):
        raise ValueError(f"unsafe result path component in {components}")
    root = Path(config.RESULTS_FILE).parent / "runs"
    return root / manifest["version"] / chash / workflow / pair_id / f"repeat_{repeat}"


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
    row.update(metrics.protected_field_metrics(pair["resume"], out["resume_md"]))
    row.update(metrics.fabrication_bait_metrics(
        pair["resume"], out["resume_md"], pair.get("keywords", []),
    ))
    if not skip_faithfulness:
        row.update(metrics.faithfulness(pair["resume"], out["resume_md"]))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflows", nargs="*", default=list(WORKFLOWS))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="only the first N pairs")
    ap.add_argument("--skip-faithfulness", action="store_true")
    ap.add_argument("--split", default="test", help="dataset split to run (default: test)")
    ap.add_argument("--categories", nargs="*", help="optional category IDs to include")
    args = ap.parse_args()

    pairs = load_pairs(split=args.split, categories=args.categories)[: args.limit]
    manifest = load_manifest()
    dhash = dataset_hash()
    done = done_keys()
    chash = config_hash()
    os.makedirs(os.path.dirname(config.RESULTS_FILE), exist_ok=True)

    for wf in args.workflows:
        for pair in pairs:
            for rep in range(args.repeats):
                if (wf, pair["pair_id"], rep, config.MODEL, chash, dhash) in done:
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
                row.update(evaluate_operational(
                    wf,
                    pair,
                    out,
                    tracker.totals(),
                    tracker.events,
                    scored_row=row,
                    error=row.get("error"),
                ))
                run_dir = _run_dir(manifest, chash, wf, pair["pair_id"], rep)
                resume_path = run_dir / "resume.md"
                metrics_path = run_dir / "metrics.json"
                trace_path = run_dir / "trace.json"
                _atomic_text(resume_path, out.get("resume_md", ""))
                _atomic_text(trace_path, json.dumps(tracker.events, indent=2, ensure_ascii=False) + "\n")
                row.update({
                    "workflow": wf,
                    "pair_id": pair["pair_id"],
                    "resume_id": pair["resume_id"],
                    "jd_id": pair["jd_id"],
                    "category": pair["category"],
                    "role_family": pair["role_family"],
                    "seniority": pair["seniority"],
                    "fit_level": pair["fit_level"],
                    "split": pair["split"],
                    "repeat": rep,
                    "model": config.MODEL,
                    "config_hash": chash,
                    "dataset_version": manifest["version"],
                    "dataset_hash": dhash,
                    "wall_seconds": round(time.time() - t0, 1),
                    "resume_path": str(resume_path),
                    "metrics_path": str(metrics_path),
                    "trace_path": str(trace_path),
                    "native_output_path": out.get("output_path"),
                })
                _atomic_text(metrics_path, json.dumps(row, indent=2, ensure_ascii=False) + "\n")
                with open(config.RESULTS_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print("ok" if not row["error"] else "ERROR", f"({row['wall_seconds']}s)")


if __name__ == "__main__":
    main()
