"""Runs every workflow on every pair, scores it, and appends one JSON line per run.

    python run_all.py                         # every test pair, 1 repeat
    python run_all.py --workflows w2_baseline # just one workflow
    python run_all.py --limit 2 --repeats 1   # quick test on 2 pairs

Safe to stop and restart: runs already in results.jsonl with the same model and generation
identity are skipped. Changing generation settings, shared code/prompts, or workflow source changes
that identity; evaluator-only changes are handled by offline rescoring.
Workflows run one after another (not in parallel) so they don't fight over the GPU,
otherwise the timing numbers would be meaningless.
"""
import argparse
import hashlib
import importlib.util
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
from workflows import WORKFLOWS, WORKFLOW_MODELS


def _generation_source_paths(workflow=None):
    """Return generation code that can change a saved writer output."""
    root = Path(__file__).resolve().parent
    paths = list((root / "common").glob("*.py"))
    workflow_names = [workflow] if workflow else sorted(WORKFLOWS)
    for name in workflow_names:
        module_name = WORKFLOWS[name].__module__
        spec = importlib.util.find_spec(module_name)
        if spec and spec.origin:
            paths.append(Path(spec.origin))
    return sorted(set(paths), key=lambda path: str(path))


def _generation_source_fingerprint(workflow=None):
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for path in _generation_source_paths(workflow):
        try:
            label = str(path.resolve().relative_to(root))
        except ValueError:
            label = str(path.resolve())
        digest.update(label.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def config_hash(workflow=None):
    """Generation-only identity; evaluator changes do not invalidate writer outputs."""
    generation_keys = (
        "BASE_MODEL", "FINE_TUNED_MODEL", "TEMPERATURE", "SEED", "NUM_CTX", "THINK", "MAX_STEPS",
        "MAX_CRITIC_ROUNDS", "MAX_TOOL_CALLS", "PDF_MIN_CHARS_PER_PAGE",
        "W3_GENERATION_VERSION", "W3_STRUCTURED_RETRIES",
        "W3_DRAFT_MAX_TOKENS", "W3_CRITIC_MAX_TOKENS",
        "W6_NUM_CANDIDATES",
        "JD_MIN_CHARS", "FETCH_TIMEOUT",
    )
    settings = {key: getattr(config, key) for key in generation_keys}
    settings["prompts"] = [
        prompts.BASE_RULES,
        prompts.CREATOR_SYSTEM,
        prompts.STRUCTURED_CREATOR_SYSTEM,
    ]
    settings["workflow"] = workflow
    settings["generation_source"] = _generation_source_fingerprint(workflow)
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
           "hit_budget": out.get("hit_budget", False),
           "finished": out.get("finished", bool(out.get("resume_md", "").strip())),
           "words": len(out["resume_md"].split()), "evaluation_failures": {}}

    def evaluate_component(name, function):
        try:
            row.update(function())
        except Exception as exc:
            row["evaluation_failures"][name] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

    evaluate_component(
        "relevance",
        lambda: metrics.jd_similarity(pair["resume"], out["resume_md"], pair["jd"]),
    )

    def keyword_scores():
        before = metrics.keyword_coverage(pair["resume"], pair["keywords"])
        after = metrics.keyword_coverage(out["resume_md"], pair["keywords"])
        return {
            "kw_before": round(before["coverage"], 4),
            "kw_after": round(after["coverage"], 4),
            "kw_missing": after["missing"],
        }

    evaluate_component("keywords", keyword_scores)
    evaluate_component(
        "protected_fields",
        lambda: metrics.protected_field_metrics(pair["resume"], out["resume_md"]),
    )
    evaluate_component(
        "fabrication_bait",
        lambda: metrics.fabrication_bait_metrics(
            pair["resume"], out["resume_md"], pair.get("keywords", []),
        ),
    )
    if not skip_faithfulness:
        evaluate_component(
            "faithfulness", lambda: metrics.faithfulness(pair["resume"], out["resume_md"])
        )
    return row


def score_generated_output(pair, out, tracker, skip_faithfulness=False):
    """Score a successful generation without converting evaluator errors into run errors."""
    try:
        row = score(pair, out, tracker, skip_faithfulness)
    except Exception as exc:
        # Preserve the generation and trace even if an unexpected failure escapes
        # the component-level isolation in score().
        row = {
            **tracker.totals(),
            "hit_budget": out.get("hit_budget", False),
            "finished": out.get("finished", True),
            "words": len(out["resume_md"].split()),
            "evaluation_failures": {
                "online_score": f"{type(exc).__name__}: {exc}"
            },
        }
        traceback.print_exc()
    row["error"] = None
    return row


def add_operational_scores(workflow, pair, out, tracker, row):
    """Add operational metrics without losing artifacts if that evaluator fails."""
    try:
        row.update(evaluate_operational(
            workflow,
            pair,
            out,
            tracker.totals(),
            tracker.events,
            scored_row=row,
            error=row.get("error"),
        ))
    except Exception as exc:
        row.setdefault("evaluation_failures", {})["operational"] = (
            f"{type(exc).__name__}: {exc}"
        )
        traceback.print_exc()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflows", nargs="*", default=list(WORKFLOWS))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="only the first N pairs")
    ap.add_argument("--skip-faithfulness", action="store_true")
    ap.add_argument("--outputs-only", action="store_true",
                    help="save resumes and runtime traces without offline scoring")
    ap.add_argument("--split", default="test", help="dataset split to run (default: test)")
    ap.add_argument("--categories", nargs="*", help="optional category IDs to include")
    args = ap.parse_args()

    pairs = load_pairs(split=args.split, categories=args.categories)[: args.limit]
    manifest = load_manifest()
    dhash = dataset_hash()
    done = done_keys()
    os.makedirs(os.path.dirname(config.RESULTS_FILE), exist_ok=True)

    for wf in args.workflows:
        expected_model = WORKFLOW_MODELS[wf]
        chash = config_hash(wf)
        for pair in pairs:
            for rep in range(args.repeats):
                if (wf, pair["pair_id"], rep, expected_model, chash, dhash) in done:
                    continue
                tracker = Tracker()
                print(f"[{wf}] {pair['pair_id']} repeat {rep} ...", end=" ", flush=True)
                t0 = time.time()
                generation_started = time.time()
                evaluation_seconds = 0.0
                out = {"resume_md": "", "finished": False, "hit_budget": False}
                try:
                    out = WORKFLOWS[wf](pair, tracker)
                except Exception as e:
                    generation_seconds = time.time() - generation_started
                    row = {**tracker.totals(), "error": f"{type(e).__name__}: {e}"}
                    traceback.print_exc()
                else:
                    generation_seconds = time.time() - generation_started
                    if out.get("errors") or not out.get("resume_md", "").strip():
                        # A clear workflow error is a generation failure, not an evaluator failure.
                        row = {
                            **tracker.totals(),
                            "hit_budget": out.get("hit_budget", False),
                            "finished": False,
                            "error": "; ".join(
                                out.get("errors") or ["EMPTY_OUTPUT: no resume produced"]
                            ),
                        }
                    else:
                        if args.outputs_only:
                            row = {
                                **tracker.totals(),
                                "hit_budget": out.get("hit_budget", False),
                                "finished": out.get("finished", False),
                                "evaluation_skipped": True,
                                "error": None,
                            }
                        else:
                            evaluation_started = time.time()
                            row = score_generated_output(
                                pair, out, tracker, args.skip_faithfulness
                            )
                            evaluation_seconds = time.time() - evaluation_started
                if not args.outputs_only:
                    add_operational_scores(wf, pair, out, tracker, row)
                run_dir = _run_dir(manifest, chash, wf, pair["pair_id"], rep)
                resume_path = run_dir / "resume.md"
                metrics_path = run_dir / "metrics.json"
                trace_path = run_dir / "trace.json"
                _atomic_text(resume_path, out.get("resume_md", ""))
                _atomic_text(trace_path, json.dumps(tracker.events, indent=2, ensure_ascii=False) + "\n")
                actual_model = next(
                    (event.get("model") for event in tracker.events
                     if event.get("type") == "llm" and event.get("model")),
                    expected_model,
                )
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
                    "model": actual_model,
                    "config_hash": chash,
                    "dataset_version": manifest["version"],
                    "dataset_hash": dhash,
                    "wall_seconds": round(time.time() - t0, 1),
                    "generation_seconds": round(generation_seconds, 3),
                    "evaluation_seconds": round(evaluation_seconds, 3),
                    "resume_path": str(resume_path),
                    "metrics_path": str(metrics_path),
                    "trace_path": str(trace_path),
                    "native_output_path": out.get("output_path"),
                    # Technique-specific diagnostics remain separate from the shared scorecard.
                    "workflow_details": out.get("details", {}),
                })
                _atomic_text(metrics_path, json.dumps(row, indent=2, ensure_ascii=False) + "\n")
                with open(config.RESULTS_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                status = (
                    "ERROR" if row["error"] else
                    "EVAL_ERROR" if row.get("evaluation_failures") else
                    "ok"
                )
                print(status, f"({row['wall_seconds']}s)")


if __name__ == "__main__":
    main()
