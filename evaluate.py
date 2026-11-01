"""One resumable entry point for offline quality, robustness and review evaluation.

Example:
    python evaluate.py all --workflows w1_single_pass --judge-model gemma4:e4b --resume
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import config
from common.data import dataset_hash, load_pairs
from common.llm import Tracker
from eval import metrics
from eval.human_review import write_review_artifacts
from eval.judge import JudgeConfig, evaluate_judge, installed_model, judge_identity
from eval.operational import evaluate_operational
from eval.reporting import write_reports
from eval.robustness import (
    VARIANTS,
    perturb_pair,
    robustness_manifest,
    select_stratified_pairs,
)
from scripts.validate_dataset import validate_dataset
from workflows import WORKFLOWS


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _generation_rows(workflows):
    rows = [
        json.loads(line)
        for line in Path(config.RESULTS_FILE).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [row for row in rows if row.get("workflow") in workflows]


def _validate_generation_rows(rows, pairs, expected_dataset_hash):
    pair_map = {pair["pair_id"]: pair for pair in pairs}
    errors = []
    seen = set()
    for row in rows:
        key = (row["workflow"], row["pair_id"], row.get("repeat", 0))
        if key in seen:
            errors.append(f"duplicate generation row {key}")
        seen.add(key)
        if row["pair_id"] not in pair_map:
            errors.append(f"unknown pair {row['pair_id']}")
        if row.get("dataset_hash") != expected_dataset_hash:
            errors.append(f"{key}: dataset hash mismatch")
        for field in ("resume_path", "metrics_path", "trace_path"):
            if not Path(row[field]).exists():
                errors.append(f"{key}: missing {field}={row[field]}")
    if errors:
        raise RuntimeError("RUN_COMPATIBILITY_FAILED:\n" + "\n".join(errors))


def _evaluation_identity(judge_config, model_info, nli_model):
    nli = {
        **nli_model,
        "version": config.NLI_EVALUATOR_VERSION,
        "threshold": config.ENTAIL_THRESHOLD,
        "evidence_limit": config.NLI_EVIDENCE_LIMIT,
        "evidence_normalization": "markdown-nfkc-v1",
    }
    identity = {
        "schema_version": config.EVALUATOR_SCHEMA_VERSION,
        "judge": judge_identity(judge_config, model_info),
        "nli": nli,
        "faithfulness_components": {
            "protected_fields": config.PROTECTED_FIELD_EVALUATOR_VERSION,
            "fabrication_bait": config.FABRICATION_BAIT_EVALUATOR_VERSION,
        },
        "embedding_model": config.EMBED_MODEL,
        "task_gate": {
            "faithfulness": config.TASK_FAITHFULNESS_THRESHOLD,
            "min_words": config.TASK_MIN_WORDS,
            "max_words": config.TASK_MAX_WORDS,
        },
        "timing_note": (
            "seconds is generation model/tool time; historical wall_seconds may include online "
            "embedding or NLI work and is reported without attempting to estimate a split"
        ),
    }
    identity["evaluator_hash"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()[:16]
    return identity


def _quality(pair, tailored_resume):
    result = {"words": len(tailored_resume.split())}
    result.update(metrics.jd_similarity(pair["resume"], tailored_resume, pair["jd"]))
    before = metrics.keyword_coverage(pair["resume"], pair["keywords"])
    after = metrics.keyword_coverage(tailored_resume, pair["keywords"])
    result.update({
        "kw_before": round(before["coverage"], 4),
        "kw_after": round(after["coverage"], 4),
        "kw_missing": after["missing"],
    })
    return result


def _record_path(root, row):
    return (
        root / "records" / row["workflow"] / row["pair_id"]
        / f"repeat_{row.get('repeat', 0)}.json"
    )


def _evaluate_output(
    generation_row,
    pair,
    tailored_resume,
    root,
    judge_config,
    skip_nli=False,
    skip_judge=False,
    force=False,
    trace=None,
    totals=None,
):
    failures = {}
    record = {
        **generation_row,
        "resume_md": tailored_resume,
        "original_resume": pair["resume"],
        "job_description": pair["jd"],
        "evaluation_failures": failures,
    }
    try:
        record.update(_quality(pair, tailored_resume))
    except Exception as exc:
        failures["embedding_or_keyword"] = f"{type(exc).__name__}: {exc}"

    try:
        record.update(metrics.protected_field_metrics(pair["resume"], tailored_resume))
    except Exception as exc:
        failures["protected_field_violation"] = f"{type(exc).__name__}: {exc}"
        record["protected_field_violation"] = None

    try:
        record.update(metrics.fabrication_bait_metrics(
            pair["resume"], tailored_resume, pair.get("keywords", []),
        ))
    except Exception as exc:
        failures["fabrication_bait_violation"] = f"{type(exc).__name__}: {exc}"
        record["fabrication_bait_violation"] = None

    if not skip_nli:
        started = time.perf_counter()
        try:
            record.update(metrics.faithfulness(
                pair["resume"],
                tailored_resume,
                cache_dir=root / "cache" / "nli",
                force=force,
            ))
        except Exception as exc:
            failures["faithfulness"] = f"{type(exc).__name__}: {exc}"
            record["faithfulness"] = None
        record["nli_seconds"] = round(time.perf_counter() - started, 4)

    if not skip_judge:
        judgment = evaluate_judge(
            pair["resume"],
            pair["jd"],
            tailored_resume,
            root / "cache" / "judge",
            judge_config=judge_config,
            force=force,
        )
        record["judge_status"] = judgment["status"]
        record["judge_cache_key"] = judgment["cache_key"]
        record["judge_record"] = judgment
        if judgment["status"] == "scored":
            record["judge_overall_mean"] = judgment["result"]["overall_mean"]
            record["judge_dimensions"] = judgment["result"]["dimensions"]
            for name, dimension in judgment["result"]["dimensions"].items():
                record[f"judge_{name}"] = dimension["score"]
        else:
            record["judge_overall_mean"] = None
            failures["judge_overall_mean"] = judgment["failure_reason"]
        record["judge_seconds"] = round(
            sum(attempt.get("seconds", 0) for attempt in judgment["attempts"]), 4
        )

    operational_output = {
        "resume_md": tailored_resume,
        "finished": not bool(generation_row.get("error")) and bool(tailored_resume.strip()),
        "hit_budget": generation_row.get("hit_budget", False),
        "errors": [generation_row["error"]] if generation_row.get("error") else [],
    }
    record.update(evaluate_operational(
        generation_row["workflow"],
        pair,
        operational_output,
        totals or generation_row,
        trace if trace is not None else _json(generation_row["trace_path"]),
        scored_row=record,
        error=generation_row.get("error"),
    ))
    return record


def _write_index(root, rows, name="evaluations.jsonl"):
    rows = sorted(rows, key=lambda row: (
        row["workflow"], row["pair_id"], row.get("repeat", 0), row.get("variant", "")
    ))
    _atomic_text(
        root / name,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def evaluate_saved(
    rows,
    pair_map,
    root,
    judge_config,
    skip_nli=False,
    skip_judge=False,
    limit=None,
    force=False,
):
    evaluated = []
    for generation_row in rows[:limit]:
        output_path = _record_path(root, generation_row)
        if output_path.exists() and not force:
            existing = _json(output_path)
            nli_complete = skip_nli or existing.get("nli_evaluator_version") == config.NLI_EVALUATOR_VERSION
            components_complete = (
                existing.get("protected_field_evaluator_version")
                == config.PROTECTED_FIELD_EVALUATOR_VERSION
                and existing.get("fabrication_bait_evaluator_version")
                == config.FABRICATION_BAIT_EVALUATOR_VERSION
            )
            judge_complete = skip_judge or existing.get("judge_status") in {"scored", "unscored"}
            if nli_complete and components_complete and judge_complete:
                evaluated.append(existing)
                continue
        pair = pair_map[generation_row["pair_id"]]
        tailored = Path(generation_row["resume_path"]).read_text(encoding="utf-8")
        record = _evaluate_output(
            generation_row, pair, tailored, root, judge_config,
            skip_nli=skip_nli, skip_judge=skip_judge, force=force,
        )
        record["evaluator_hash"] = root.name
        _atomic_text(output_path, json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        evaluated.append(record)
        print(
            f"[evaluate] {record['workflow']} {record['pair_id']}: "
            f"faithfulness={record.get('faithfulness')} judge={record.get('judge_overall_mean')}",
            flush=True,
        )
    _write_index(root, evaluated)
    return evaluated


def _write_manifest(path, manifest):
    if path.exists():
        existing = _json(path)
        if existing != manifest:
            raise RuntimeError("ROBUSTNESS_MANIFEST_CHANGED")
    else:
        _atomic_text(path, json.dumps(manifest, indent=2) + "\n")


def evaluate_robustness(
    workflows,
    base_rows,
    pairs,
    root,
    judge_config,
    skip_nli=False,
    skip_judge=False,
    force=False,
    limit=None,
):
    manifest = robustness_manifest(pairs, dataset_hash(), config.ROBUSTNESS_SEED)
    _write_manifest(root / "robustness" / "sample_manifest.json", manifest)
    selected = select_stratified_pairs(pairs, config.ROBUSTNESS_SEED)
    if limit is not None:
        selected = selected[:limit]
    pair_map = {pair["pair_id"]: pair for pair in pairs}
    base_by_key = {
        (row["workflow"], row["pair_id"]): row for row in base_rows
    }
    results = []
    for workflow in workflows:
        for base_pair in selected:
            base_generation = base_by_key.get((workflow, base_pair["pair_id"]))
            if not base_generation:
                raise RuntimeError(f"missing original generation for {workflow}/{base_pair['pair_id']}")
            for variant in VARIANTS:
                variant_dir = (
                    root / "robustness" / "runs" / workflow / base_pair["pair_id"] / variant
                )
                record_path = variant_dir / "evaluation.json"
                if record_path.exists() and not force:
                    existing = _json(record_path)
                    nli_complete = (
                        skip_nli
                        or existing.get("nli_evaluator_version") == config.NLI_EVALUATOR_VERSION
                    )
                    components_complete = (
                        existing.get("protected_field_evaluator_version")
                        == config.PROTECTED_FIELD_EVALUATOR_VERSION
                        and existing.get("fabrication_bait_evaluator_version")
                        == config.FABRICATION_BAIT_EVALUATOR_VERSION
                    )
                    judge_complete = (
                        skip_judge or existing.get("judge_status") in {"scored", "unscored"}
                    )
                    if nli_complete and components_complete and judge_complete:
                        results.append(existing)
                        continue
                if variant == "original":
                    variant_pair = base_pair
                    tailored = Path(base_generation["resume_path"]).read_text(encoding="utf-8")
                    generation_row = dict(base_generation)
                    trace = _json(base_generation["trace_path"])
                    totals = generation_row
                else:
                    resume, jd = perturb_pair(base_pair, variant)
                    _atomic_text(variant_dir / "input_resume.md", resume)
                    _atomic_text(variant_dir / "input_jd.md", jd)
                    variant_pair = {**base_pair, "resume": resume, "jd": jd}
                    saved_resume = variant_dir / "resume.md"
                    saved_trace = variant_dir / "trace.json"
                    if (
                        not force
                        and saved_resume.exists()
                        and saved_trace.exists()
                        and record_path.exists()
                    ):
                        prior = _json(record_path)
                        tailored = saved_resume.read_text(encoding="utf-8")
                        generation_row = dict(prior)
                        trace, totals = _json(saved_trace), prior
                        record = _evaluate_output(
                            generation_row,
                            variant_pair,
                            tailored,
                            root,
                            judge_config,
                            skip_nli=skip_nli,
                            skip_judge=skip_judge,
                            force=force,
                            trace=trace,
                            totals=totals,
                        )
                        record.update({
                            "base_pair_id": base_pair["pair_id"],
                            "variant": variant,
                            "pair_id": f"{base_pair['pair_id']}__{variant}",
                            "transformation_version": "meaning-preserving-v1",
                            "evaluator_hash": root.name,
                        })
                        _atomic_text(
                            record_path,
                            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                        )
                        results.append(record)
                        continue
                    tracker = Tracker()
                    started = time.perf_counter()
                    try:
                        output = WORKFLOWS[workflow](variant_pair, tracker)
                        tailored = output.get("resume_md", "")
                        error = "; ".join(output.get("errors") or []) or None
                    except Exception as exc:
                        tailored = ""
                        output = {}
                        error = f"{type(exc).__name__}: {exc}"
                    _atomic_text(variant_dir / "resume.md", tailored)
                    _atomic_text(
                        variant_dir / "trace.json",
                        json.dumps(tracker.events, indent=2, ensure_ascii=False) + "\n",
                    )
                    generation_row = {
                        **base_generation,
                        **tracker.totals(),
                        "error": error,
                        "resume_path": str(variant_dir / "resume.md"),
                        "trace_path": str(variant_dir / "trace.json"),
                        "wall_seconds": round(time.perf_counter() - started, 4),
                        "finished": not error and bool(tailored.strip()),
                    }
                    trace, totals = tracker.events, tracker.totals()
                record = _evaluate_output(
                    generation_row,
                    variant_pair,
                    tailored,
                    root,
                    judge_config,
                    skip_nli=skip_nli,
                    skip_judge=skip_judge,
                    force=force,
                    trace=trace,
                    totals=totals,
                )
                record.update({
                    "base_pair_id": base_pair["pair_id"],
                    "variant": variant,
                    "pair_id": f"{base_pair['pair_id']}__{variant}",
                    "transformation_version": "meaning-preserving-v1",
                    "evaluator_hash": root.name,
                })
                _atomic_text(record_path, json.dumps(record, indent=2, ensure_ascii=False) + "\n")
                results.append(record)
                print(
                    f"[robustness] {workflow} {base_pair['pair_id']} {variant}: "
                    f"{'ok' if not record.get('error') else 'ERROR'}",
                    flush=True,
                )
    _write_index(root, results, "robustness/evaluations.jsonl")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("all", "score", "robustness", "report"), default="all")
    parser.add_argument("--workflows", nargs="+", default=["w1_single_pass"])
    parser.add_argument("--judge-model", default=config.JUDGE_MODEL)
    parser.add_argument("--resume", action="store_true", help="reuse compatible checkpoints (default)")
    parser.add_argument("--force", action="store_true", help="ignore evaluation caches/checkpoints")
    parser.add_argument("--skip-faithfulness", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--skip-reports", action="store_true")
    parser.add_argument("--skip-human-review", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--robustness-limit", type=int)
    args = parser.parse_args(argv)

    unknown = sorted(set(args.workflows) - set(WORKFLOWS))
    if unknown:
        raise SystemExit(f"Unknown workflows: {unknown}; registered={sorted(WORKFLOWS)}")
    validation = validate_dataset()
    pairs = load_pairs(split="test")
    pair_map = {pair["pair_id"]: pair for pair in pairs}
    generation_rows = _generation_rows(args.workflows)
    _validate_generation_rows(generation_rows, pairs, validation["dataset_hash"])

    judge_config = JudgeConfig(model=args.judge_model)
    model_info = installed_model(judge_config.model)
    nli_model = metrics.nli_model_identity()
    identity = _evaluation_identity(judge_config, model_info, nli_model)
    generation_hashes = sorted({row["config_hash"] for row in generation_rows})
    generation_scope = hashlib.sha256(
        json.dumps(generation_hashes).encode()
    ).hexdigest()[:12]
    root = (
        Path(config.RESULTS_FILE).parent / "evaluations" / validation["dataset_hash"]
        / generation_scope / identity["evaluator_hash"]
    )
    _atomic_text(root / "evaluator_identity.json", json.dumps({
        **identity,
        "dataset": validation,
        "generation_hashes": generation_hashes,
        "judge_model_info": model_info,
        "judge_config": asdict(judge_config),
    }, indent=2) + "\n")

    if not args.skip_faithfulness and args.command in {"all", "score", "robustness"}:
        preflight = metrics.nli_preflight()
        _atomic_text(root / "nli_preflight.json", json.dumps(preflight, indent=2) + "\n")

    limit = args.limit
    robustness_limit = args.robustness_limit
    if args.pilot:
        limit = limit or 3
        robustness_limit = robustness_limit or 2

    evaluation_rows = []
    if args.command in {"all", "score"}:
        evaluation_rows = evaluate_saved(
            generation_rows,
            pair_map,
            root,
            judge_config,
            skip_nli=args.skip_faithfulness,
            skip_judge=args.skip_judge,
            limit=limit,
            force=args.force,
        )
    elif (root / "evaluations.jsonl").exists():
        evaluation_rows = [
            json.loads(line)
            for line in (root / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()
        ]

    robustness_rows = []
    if args.command in {"all", "robustness"} and not args.skip_robustness:
        robustness_rows = evaluate_robustness(
            args.workflows,
            generation_rows,
            pairs,
            root,
            judge_config,
            skip_nli=args.skip_faithfulness,
            skip_judge=args.skip_judge,
            force=args.force,
            limit=robustness_limit,
        )
    elif (root / "robustness" / "evaluations.jsonl").exists():
        robustness_rows = [
            json.loads(line)
            for line in (root / "robustness" / "evaluations.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]

    if args.command in {"all", "report"} and not args.skip_reports:
        if not evaluation_rows and (root / "evaluations.jsonl").exists():
            evaluation_rows = [
                json.loads(line)
                for line in (root / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()
            ]
        write_reports(root / "summaries", evaluation_rows, robustness_rows)
    if args.command in {"all", "report"} and not args.skip_human_review and evaluation_rows:
        write_review_artifacts(root / "human_review", evaluation_rows)

    print(json.dumps({
        "output_root": str(root),
        "evaluated": len(evaluation_rows),
        "robustness_rows": len(robustness_rows),
        "evaluator_hash": identity["evaluator_hash"],
    }, indent=2))
    return root


if __name__ == "__main__":
    main()
