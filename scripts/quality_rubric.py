"""Score every generated resume with the quality-only rubric (v3) and count good resumes.

    python scripts/quality_rubric.py                      # judge everything not yet scored, then summarise
    python scripts/quality_rubric.py --summary-only --nli-threshold 0.85 --rubric-threshold 4

A resume is good when its NLI claim faithfulness (original vs generated) is at least
--nli-threshold AND its rubric overall score is at least --rubric-threshold. Failed generations and
unscored judgments are not good. Per-resume scores are written to resumes.csv so thresholds can be
changed later without new judge calls. Resumable: finished judgments are kept.
"""
import argparse
import csv
import glob
import json
import os
import pathlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data import load_pairs  # noqa: E402
from eval import quality_rubric as qr  # noqa: E402
from eval.judge import hosted_client, hosted_request_options  # noqa: E402

WORKFLOWS = ["w1_single_pass", "w2_baseline", "w3_critic", "w4_evidence", "w5_fine_tuned",
             "w6_best_of_n", "w7_plan_execute"]
DEFAULT_NLI_ROOT = ("results/evaluations/3bb1861e2944d4b4e792b78d475572a70ef762c03ac03fd3cca8acc9141a6f81/"
                    "2b526ff0b92d/1cd268431e1c38d1")


def load_env():
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def newest_generations():
    """Newest config hash per workflow and its generation rows, keyed by pair."""
    rows = [json.loads(l) for l in (ROOT / "results/results.jsonl").read_text().splitlines() if l.strip()]
    newest = {}
    for r in rows:
        if r.get("workflow") in WORKFLOWS and r.get("config_hash"):
            newest[r["workflow"]] = r["config_hash"]
    out = {wf: {} for wf in WORKFLOWS}
    for r in rows:
        if r.get("workflow") in newest and r.get("config_hash") == newest[r["workflow"]]:
            out[r["workflow"]][r["pair_id"]] = r
    return newest, out


def judge_all(args, out_dir, pairs, generations):
    client = hosted_client(args.provider)
    options = hosted_request_options(args.provider)
    jobs = []
    for wf in args.workflows:
        for pair_id, row in sorted(generations[wf].items()):
            path = out_dir / "records" / wf / f"{pair_id}.json"
            previous = None
            if path.exists():
                existing = json.loads(path.read_text())
                retry = args.retry_unscored and existing.get("failure_kind") == "invalid_output"
                if existing.get("status") == "scored" or existing.get("failure_kind") == "generation_unavailable" \
                        or (existing.get("failure_kind") == "invalid_output" and not retry):
                    continue
                previous = existing if retry else None
            jobs.append((wf, pair_id, row, path, previous))
    print(f"{len(jobs)} resumes to judge", flush=True)
    lock, done = threading.Lock(), [0]

    def work(job):
        wf, pair_id, row, path, previous = job
        pair = pairs[pair_id]
        base = {"workflow": wf, "pair_id": pair_id, "config_hash": row["config_hash"],
                "rubric_version": qr.QUALITY_RUBRIC_VERSION}
        if row.get("error") or not row.get("resume_path") or not (ROOT / row["resume_path"]).exists():
            record = {**base, "status": "generation_unavailable", "failure_kind": "generation_unavailable",
                      "scores": None, "mean": None}
        else:
            tailored = (ROOT / row["resume_path"]).read_text(encoding="utf-8")
            attempts = args.retry_unscored or args.max_attempts
            record = {**base, **qr.score(pair["resume"], pair["jd"], tailored, client, args.model,
                                         args.reasoning_effort, options, attempts)}
            if previous:
                # Unscored after the first pass: retried with extra attempts; earlier attempts kept for audit.
                record["retried"] = True
                record["attempts"] = previous.get("attempts", []) + record["attempts"]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)
        with lock:
            done[0] += 1
            print(f"[{done[0]}/{len(jobs)}] {wf} {pair_id}: {record['status']} {record.get('overall')}", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, jobs))


def summarise(args, out_dir, generations):
    nli = {}
    for f in glob.glob(str(ROOT / args.nli_root / "records/*/*/repeat_0.json")):
        r = json.loads(pathlib.Path(f).read_text())
        nli[(r["workflow"], r["pair_id"])] = r
    rows, summary = [], []
    for wf in WORKFLOWS:
        counts = {"workflow": wf, "attempts": len(generations[wf]), "generated": 0, "nli_pass": 0,
                  "rubric_scored": 0, "rubric_pass": 0, "good": 0}
        for pair_id, gen in sorted(generations[wf].items()):
            rec_path = out_dir / "records" / wf / f"{pair_id}.json"
            rubric = json.loads(rec_path.read_text()) if rec_path.exists() else {}
            n = nli.get((wf, pair_id), {})
            generated = not gen.get("error") and n.get("claim_faithfulness") is not None
            faith = n.get("claim_faithfulness") if generated else None
            overall = rubric.get("overall")
            nli_pass = faith is not None and faith >= args.nli_threshold - 1e-9
            rubric_pass = overall is not None and overall >= args.rubric_threshold - 1e-9
            row = {"workflow": wf, "pair_id": pair_id, "category": gen.get("category"),
                   "fit_level": gen.get("fit_level"), "generated": generated,
                   "claim_faithfulness": faith,
                   "unsupported_claims": (n["n_claims"] - n["supported_claims"]) if generated else None,
                   "fabrication_bait": n.get("fabrication_bait_violation") if generated else None,
                   "protected_field_violation": n.get("protected_field_violation") if generated else None,
                   "rubric_status": rubric.get("status", "not_run"), "rubric_overall": overall,
                   **{f"rubric_{d}": (rubric.get("scores") or {}).get(d) for d in qr.DIMENSIONS},
                   "nli_pass": nli_pass, "rubric_pass": rubric_pass, "good": nli_pass and rubric_pass}
            rows.append(row)
            counts["generated"] += generated
            counts["nli_pass"] += nli_pass
            counts["rubric_scored"] += overall is not None
            counts["rubric_pass"] += rubric_pass
            counts["good"] += row["good"]
        counts["good_rate"] = round(counts["good"] / counts["attempts"], 3) if counts["attempts"] else None
        summary.append(counts)
    with open(out_dir / "resumes.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta = {"nli_threshold": args.nli_threshold, "rubric_threshold": args.rubric_threshold,
            "nli_root": args.nli_root, "summary": summary}
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nGood resume = NLI faithfulness >= {args.nli_threshold} AND rubric overall >= {args.rubric_threshold}")
    print(f"{'workflow':16} {'gen':>4} {'NLI pass':>9} {'rubric scored':>14} {'rubric pass':>12} {'GOOD':>6} {'good/100':>9}")
    for c in summary:
        print(f"{c['workflow']:16} {c['generated']:>4} {c['nli_pass']:>9} {c['rubric_scored']:>14} "
              f"{c['rubric_pass']:>12} {c['good']:>6} {c['good_rate']:>9}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workflows", nargs="+", default=WORKFLOWS, choices=WORKFLOWS)
    ap.add_argument("--provider", default="deepinfra")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--nli-root", default=DEFAULT_NLI_ROOT)
    ap.add_argument("--nli-threshold", type=float, default=0.8)
    ap.add_argument("--rubric-threshold", type=float, default=4.0)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--retry-unscored", type=int, default=0, metavar="N",
                    help="re-judge resumes left unscored by invalid judge output, with up to N attempts")
    ap.add_argument("--pair-ids-file", help="judge only these pairs (trial run)")
    args = ap.parse_args()
    os.chdir(ROOT)
    load_env()
    ident = qr.identity(args.provider, args.model, args.reasoning_effort, args.max_attempts)
    _, generations = newest_generations()
    if args.pair_ids_file:
        keep = {l.strip() for l in open(args.pair_ids_file) if l.strip() and not l.startswith("#")}
        generations = {wf: {p: r for p, r in rows.items() if p in keep} for wf, rows in generations.items()}
    dataset_hash = next(iter(next(iter(generations.values())).values()))["dataset_hash"]
    out_dir = ROOT / "results/evaluations" / dataset_hash / "quality-rubric" / ident["identity_hash"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "identity.json").write_text(json.dumps(ident, indent=2) + "\n")
    if not args.summary_only:
        pairs = {p["pair_id"]: p for p in load_pairs()}
        judge_all(args, out_dir, pairs, generations)
    summarise(args, out_dir, generations)
    print(f"\nResults: {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
