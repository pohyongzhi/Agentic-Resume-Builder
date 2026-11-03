"""One resumable entry point for offline quality, robustness, injection and review evaluation.

Example:
    python evaluate.py all --workflows w1_single_pass --judge-model gemma4:e4b --resume

Extension points (see docs/evaluation/workstreams/integration/HANDOVER.md):

- Per-output metrics implement the ``OutputComponent`` protocol and are added in
  ``build_output_components``. Their identity joins the evaluator hash and their
  ``is_complete`` check joins checkpoint completeness for scoring and robustness records.
- Run-level stages (robustness, pairwise, optional injection) are ``Stage`` entries in ``STAGES``. Each one
  automatically gets a CLI command, ``--skip-<name>``, ``--<name>-limit`` and a JSONL index that
  ``report`` reloads; ``eval.reporting.STAGE_REPORTERS`` writes its summary files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import config
from common.data import dataset_hash, load_pairs
from common.llm import Tracker
from eval import metrics
from eval.human_review import write_review_artifacts
from eval.injection import (
    injection_identity,
    injection_manifest,
    run_injection_attack,
    write_injection_manifest,
)
from eval.judge import JudgeConfig, configured_model, evaluate_judge, judge_identity
from eval.operational import evaluate_operational
from eval.pairwise import (
    FAILURE_GENERATION,
    FAILURE_INVALID,
    build_generation_pairs,
    evaluate_generation_pair,
    pairwise_identity,
)
from eval.reporting import write_reports
from eval.requirement_coverage import (
    RequirementCoverageConfig,
    coverage_identity,
    evaluate_requirement_coverage,
)
from eval.robustness import (
    FAMILIES as ROBUSTNESS_FAMILIES,
    ORIGINAL,
    VARIANT_FAMILIES,
    build_jd_pool,
    build_variant,
    classify_generation,
    out_of_scope_expected_behaviour,
    robustness_identity,
    robustness_manifest,
    select_variants,
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


class OutputComponent(Protocol):
    """A per-output metric evaluated alongside NLI, judge and deterministic checks."""

    name: str

    def identity(self) -> dict[str, Any]:
        """Cache identity: inputs are hashed by the component; this covers versions/models."""

    def is_complete(self, record: dict[str, Any]) -> bool:
        """Whether a saved record already carries a final result for this identity."""

    def evaluate(self, record, generation_row, pair, tailored_resume, root, force) -> None:
        """Update ``record`` in place; put failures in ``record['evaluation_failures']``."""


# A coverage result in one of these states is final for its identity and is not recomputed on
# resume. Exceptions outside the evaluator (for example a missing client) are retried.
COVERAGE_FINAL_STATUSES = {"scored", "unscored", "no_supportable_requirements", "generation_failed"}


class RequirementCoverageComponent:
    """Supportable-requirement coverage; failures stay unscored and are never zero."""

    name = "requirement_coverage"

    def __init__(self, enabled=True, coverage_config=None, model_info=None, client=None):
        self.enabled = enabled
        self.config = coverage_config or RequirementCoverageConfig()
        self.model_info = model_info or {
            "provider": self.config.provider, "tag": self.config.model, "digest": None,
        }
        self.client = client
        self._identity = None

    def identity(self):
        if not self.enabled:
            return {"status": "skipped"}
        if self._identity is None:
            self._identity = {
                **coverage_identity(self.config, self.model_info),
                "status": "enabled",
            }
        return self._identity

    def is_complete(self, record):
        if not self.enabled:
            return True
        return (
            record.get("requirement_coverage_evaluator_hash") == self.identity()["evaluator_hash"]
            and record.get("requirement_coverage_status") in COVERAGE_FINAL_STATUSES
        )

    def evaluate(self, record, generation_row, pair, tailored_resume, root, force):
        if not self.enabled:
            return
        failures = record.setdefault("evaluation_failures", {})
        failures.pop(self.name, None)
        record["requirement_coverage_evaluator_version"] = self.config.evaluator_version
        record["requirement_coverage_evaluator_hash"] = self.identity()["evaluator_hash"]
        record["requirement_coverage"] = None
        if generation_row.get("error") or not tailored_resume.strip():
            record["requirement_coverage_status"] = "generation_failed"
            return
        started = time.perf_counter()
        try:
            result = evaluate_requirement_coverage(
                pair["resume"],
                tailored_resume,
                pair.get("keywords", []),
                Path(root) / "cache" / "requirement_coverage",
                evaluator_config=self.config,
                client=self.client,
                model_info=self.model_info,
                force=force,
            )
        except Exception as exc:
            record["requirement_coverage_status"] = "evaluation_error"
            failures[self.name] = f"{type(exc).__name__}: {exc}"
        else:
            record.update({
                "requirement_coverage": result["coverage"],
                "requirement_coverage_status": result["status"],
                "requirement_coverage_supportable_count": result.get(
                    "supportable_requirement_count"
                ),
                "requirement_coverage_covered_count": result.get("covered_requirement_count"),
                "requirement_coverage_uncovered": result.get(
                    "uncovered_supportable_requirements", []
                ),
                "requirement_coverage_record": result,
            })
            if result["status"] == "unscored":
                failures[self.name] = result["failure_reason"]
                if result.get("failure_kind") == "evaluator_call_failed":
                    # Not final: the evaluator was unreachable, so resume retries it.
                    record["requirement_coverage_status"] = "evaluation_error"
        record["requirement_coverage_seconds"] = round(time.perf_counter() - started, 4)


def build_output_components(args, judge_config, judge_model_info):
    """Construct every registered per-output component for one run.

    Model metadata is resolved once here so every row carries the same evaluator identity.
    New per-output metrics are appended to the returned tuple.
    """
    coverage_enabled = not getattr(args, "skip_coverage", False)
    coverage_config = RequirementCoverageConfig(
        provider=getattr(args, "coverage_provider", None) or judge_config.provider,
        model=getattr(args, "coverage_model", None) or judge_config.model,
    )
    coverage_model_info = None
    if coverage_enabled:
        same_model = (
            (coverage_config.provider, coverage_config.model)
            == (judge_config.provider, judge_config.model)
            and judge_model_info.get("status") != "skipped"
        )
        coverage_model_info = (
            judge_model_info if same_model else configured_model(coverage_config)
        )
    return (
        RequirementCoverageComponent(
            enabled=coverage_enabled,
            coverage_config=coverage_config,
            model_info=coverage_model_info,
        ),
    )


def _evaluation_identity(judge_config, model_info, nli_model, components=()):
    nli = {
        **nli_model,
        "version": config.NLI_EVALUATOR_VERSION,
        "threshold": config.ENTAIL_THRESHOLD,
        "evidence_limit": config.NLI_EVIDENCE_LIMIT,
        "evidence_normalization": "markdown-nfkc-v1",
    }
    judge = judge_identity(judge_config, model_info)
    judge["status"] = model_info.get("status", "enabled")
    identity = {
        "schema_version": config.EVALUATOR_SCHEMA_VERSION,
        "judge": judge,
        "nli": nli,
        "faithfulness_components": {
            "protected_fields": config.PROTECTED_FIELD_EVALUATOR_VERSION,
            "fabrication_bait": config.FABRICATION_BAIT_EVALUATOR_VERSION,
        },
        "output_components": {
            component.name: component.identity() for component in components
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
    components=(),
):
    failures = {}
    record = {
        **generation_row,
        "resume_md": tailored_resume,
        "original_resume": pair["resume"],
        "job_description": pair["jd"],
        "evaluation_failures": failures,
    }
    saved_quality_fields = {
        "words", "sim_before", "sim_after", "sim_gain",
        "kw_before", "kw_after", "kw_missing",
    }
    if not saved_quality_fields <= generation_row.keys():
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

    for component in components:
        try:
            component.evaluate(record, generation_row, pair, tailored_resume, root, force)
        except Exception as exc:
            failures[component.name] = f"{type(exc).__name__}: {exc}"

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
    try:
        record.update(evaluate_operational(
            generation_row["workflow"],
            pair,
            operational_output,
            totals or generation_row,
            trace if trace is not None else _json(generation_row["trace_path"]),
            scored_row=record,
            error=generation_row.get("error"),
        ))
    except Exception as exc:
        failures["operational"] = f"{type(exc).__name__}: {exc}"
    return record


def _write_index(root, rows, name="evaluations.jsonl"):
    rows = sorted(rows, key=lambda row: (
        row["workflow"], row["pair_id"], row.get("repeat", 0), row.get("variant", ""),
        row.get("channel", ""),
    ))
    _atomic_text(
        root / name,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def _read_index(path):
    path = Path(path)
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _row_key(row):
    return (
        row.get("workflow"), row.get("pair_id"), row.get("repeat", 0), row.get("variant", ""),
        row.get("channel", ""), row.get("challenger_workflow", ""),
    )


def _merged_index_rows(path, new_rows):
    """Keep earlier index rows that this (subset) run did not touch, replaced by new rows.

    The root is keyed by dataset, generation scope and evaluator identity, so earlier rows in
    the same index always belong to the same saved generations and evaluator.
    """
    merged = {_row_key(row): row for row in _read_index(path)}
    merged.update({_row_key(row): row for row in new_rows})
    return list(merged.values())


def select_pairs(rows, pair_ids):
    """Restrict generation rows to a fixed pair subset; ``None`` keeps every row."""
    if pair_ids is None:
        return rows
    return [row for row in rows if row["pair_id"] in pair_ids]


def _requested_pair_ids(args, pair_map):
    requested = list(getattr(args, "pair_ids", None) or [])
    if getattr(args, "pair_ids_file", None):
        requested += [
            line.strip()
            for line in Path(args.pair_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not requested:
        return None
    unknown = sorted(set(requested) - set(pair_map))
    if unknown:
        raise SystemExit(f"Unknown pair IDs: {unknown}")
    return set(requested)


def _deterministic_components_complete(existing):
    return (
        existing.get("protected_field_evaluator_version")
        == config.PROTECTED_FIELD_EVALUATOR_VERSION
        and existing.get("fabrication_bait_evaluator_version")
        == config.FABRICATION_BAIT_EVALUATOR_VERSION
    )


def _record_complete(existing, skip_nli=False, skip_judge=False, components=()):
    """Checkpoint completeness shared by saved-output scoring and robustness."""
    nli_complete = (
        skip_nli or existing.get("nli_evaluator_version") == config.NLI_EVALUATOR_VERSION
    )
    judge_complete = skip_judge or existing.get("judge_status") in {"scored", "unscored"}
    return (
        nli_complete
        and _deterministic_components_complete(existing)
        and judge_complete
        and all(component.is_complete(existing) for component in components)
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
    components=(),
):
    evaluated = []
    for generation_row in rows[:limit]:
        output_path = _record_path(root, generation_row)
        if output_path.exists() and not force:
            existing = _json(output_path)
            if _record_complete(existing, skip_nli, skip_judge, components):
                evaluated.append(existing)
                continue
        pair = pair_map[generation_row["pair_id"]]
        tailored = Path(generation_row["resume_path"]).read_text(encoding="utf-8")
        record = _evaluate_output(
            generation_row, pair, tailored, root, judge_config,
            skip_nli=skip_nli, skip_judge=skip_judge, force=force, components=components,
        )
        record["evaluator_hash"] = root.name
        _atomic_text(output_path, json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        evaluated.append(record)
        print(
            f"[evaluate] {record['workflow']} {record['pair_id']}: "
            f"faithfulness={record.get('faithfulness')} judge={record.get('judge_overall_mean')} "
            f"coverage={record.get('requirement_coverage')}",
            flush=True,
        )
    # A subset run (--limit/--pair-ids) updates its rows without dropping other saved rows.
    _write_index(root, _merged_index_rows(root / "evaluations.jsonl", evaluated))
    return evaluated


def _write_manifest(path, manifest):
    if path.exists():
        existing = _json(path)
        if existing != manifest:
            raise RuntimeError("ROBUSTNESS_MANIFEST_CHANGED")
    else:
        _atomic_text(path, json.dumps(manifest, indent=2) + "\n")


def robustness_root(root):
    """v2 robustness artifacts live under the variant/sample identity; v1 files stay as history."""
    return Path(root) / "robustness" / robustness_identity()["identity_hash"]


# Base-generation fields copied to a variant generation row. Saved quality and task fields are
# deliberately not copied, so every variant output is scored on its own text.
_GENERATION_IDENTITY_FIELDS = (
    "workflow", "pair_id", "resume_id", "jd_id", "category", "role_family", "seniority",
    "fit_level", "split", "repeat", "model", "config_hash", "dataset_version", "dataset_hash",
)


def _input_hash(spec):
    return hashlib.sha256(
        json.dumps([spec["resume"], spec["jd"]], ensure_ascii=False).encode()
    ).hexdigest()[:16]


def _current_generation_hash(workflow):
    """The generation identity variants are produced with (None if it cannot be computed)."""
    try:
        from run_all import config_hash as _generation_hash

        return _generation_hash(workflow)
    except Exception:
        return None


def _robustness_generation(
    workflow_fn, base_pair, base_generation, spec, run_dir, force, current_hash=None,
):
    """Generate once per variant input; ``generation.json`` (written last) marks a reusable run."""
    generation_path = run_dir / "generation.json"
    input_hash = _input_hash(spec)
    if generation_path.exists() and not force:
        saved = _json(generation_path)
        if (
            saved.get("variant_version") == config.ROBUSTNESS_VARIANT_VERSION
            and saved.get("input_hash") == input_hash
        ):
            tailored = (run_dir / "resume.md").read_text(encoding="utf-8")
            return saved, tailored, _json(run_dir / "trace.json")
    input_resume = run_dir / f"input_resume{spec['resume_suffix']}"
    _atomic_text(input_resume, spec["resume"])
    _atomic_text(run_dir / "input_jd.md", spec["jd"])
    variant_pair = {
        **base_pair,
        "pair_id": f"{base_pair['pair_id']}__{spec['variant']}",
        "resume_path": str(input_resume),
        "resume": spec["resume"],
        "jd": spec["jd"],
        "keywords": spec["presented_keywords"],
        # The presented JD text is the input; never refetch the original posting.
        "jd_url": "",
        "job_url": "",
    }
    tracker = Tracker()
    started = time.perf_counter()
    crashed = False
    output = {}
    try:
        output = workflow_fn(variant_pair, tracker) or {}
        tailored = output.get("resume_md", "") or ""
        error = "; ".join(output.get("errors") or []) or (
            None if tailored.strip() else "EMPTY_OUTPUT"
        )
    except Exception as exc:
        tailored = ""
        error = f"{type(exc).__name__}: {exc}"
        crashed = True
    _atomic_text(run_dir / "resume.md", tailored)
    _atomic_text(
        run_dir / "trace.json", json.dumps(tracker.events, indent=2, ensure_ascii=False) + "\n",
    )
    generation_row = {
        **{key: base_generation[key] for key in _GENERATION_IDENTITY_FIELDS if key in base_generation},
        **tracker.totals(),
        "error": error,
        "hit_budget": output.get("hit_budget", False),
        "finished": not error and bool(tailored.strip()),
        "generation_outcome": classify_generation(tailored, error, crashed),
        "resume_path": str(run_dir / "resume.md"),
        "trace_path": str(run_dir / "trace.json"),
        "input_resume_path": str(input_resume),
        "input_jd_path": str(run_dir / "input_jd.md"),
        "input_hash": input_hash,
        "variant_version": config.ROBUSTNESS_VARIANT_VERSION,
        # Variants run with the current workflow code; the original is the saved generation.
        "config_hash": current_hash,
        "original_config_hash": base_generation.get("config_hash"),
        "generation_identity_matches_original": (
            current_hash is not None and current_hash == base_generation.get("config_hash")
        ),
        "wall_seconds": round(time.perf_counter() - started, 4),
    }
    _atomic_text(generation_path, json.dumps(generation_row, indent=2, ensure_ascii=False) + "\n")
    return generation_row, tailored, tracker.events


def _robustness_record(
    workflow, workflow_fn, variant, base_pair, base_generation, jd_pool, stage_root, root,
    judge_config, skip_nli, skip_judge, force, components, seed, current_hash=None,
):
    spec = build_variant(base_pair, variant, seed, jd_pool)
    in_scope = spec["scope"] == "in_scope"
    # Out-of-scope runs are checked for expected behaviour only: no quality judge or coverage.
    variant_components = components if in_scope else ()
    variant_skip_judge = skip_judge or not in_scope
    run_dir = stage_root / "runs" / workflow / base_pair["pair_id"] / variant
    record_path = run_dir / "evaluation.json"
    if record_path.exists() and not force:
        existing = _json(record_path)
        if existing.get("variant_version") == config.ROBUSTNESS_VARIANT_VERSION and _record_complete(
            existing, skip_nli, variant_skip_judge, variant_components,
        ):
            return existing
    if variant == ORIGINAL:
        tailored = Path(base_generation["resume_path"]).read_text(encoding="utf-8")
        generation_row = {
            **base_generation,
            "generation_outcome": classify_generation(tailored, base_generation.get("error"), False),
        }
        trace = _json(base_generation["trace_path"])
        totals = generation_row
    else:
        generation_row, tailored, trace = _robustness_generation(
            workflow_fn, base_pair, base_generation, spec, run_dir, force, current_hash,
        )
        totals = generation_row
    # Faithfulness, protected fields and bait are always scored against the clean original
    # resume (the true facts). In-scope variants also use the clean JD and keywords.
    scoring_pair = {**base_pair, "jd": spec["scoring_jd"], "keywords": spec["scoring_keywords"]}
    record = _evaluate_output(
        generation_row, scoring_pair, tailored, root, judge_config,
        skip_nli=skip_nli, skip_judge=variant_skip_judge, force=force,
        trace=trace, totals=totals, components=variant_components,
    )
    record.update({
        "base_pair_id": base_pair["pair_id"],
        "variant": variant,
        "variant_family": spec["family"],
        "variant_scope": spec["scope"],
        "variant_metadata": spec["metadata"],
        "pair_id": f"{base_pair['pair_id']}__{variant}",
        "variant_version": config.ROBUSTNESS_VARIANT_VERSION,
        "transformation_version": config.ROBUSTNESS_VARIANT_VERSION,
        "robustness_identity_hash": stage_root.name,
        "scoring_reference": "clean_original_resume",
        "scoring_jd": "clean_original_jd" if in_scope else "presented_jd",
        "evaluator_hash": root.name,
    })
    if not in_scope:
        record.update(out_of_scope_expected_behaviour(record))
    _atomic_text(record_path, json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(
        f"[robustness] {workflow} {base_pair['pair_id']} {variant}: "
        f"{record.get('generation_outcome')}",
        flush=True,
    )
    return record


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
    components=(),
    variants=None,
    workflow_registry=None,
    dataset_hash_value=None,
):
    """Run the seeded perturbation variants on the ordered stratified sample.

    Layout under ``robustness/<identity_hash>/``: ``identity.json``, ``sample_manifest.json``
    (frozen before any generation), ``runs/<workflow>/<pair_id>/<variant>/`` with
    ``input_resume.*``, ``input_jd.md``, ``resume.md``, ``trace.json``, ``generation.json`` and
    ``evaluation.json``, and ``evaluations.jsonl`` (merged across variant/limit subsets).
    ``original`` reuses the saved repeat-0 generation. A pair without a saved original is
    recorded as ``missing_original`` and none of its variants are generated.
    """
    workflow_registry = workflow_registry or WORKFLOWS
    seed = config.ROBUSTNESS_SEED
    stage_root = robustness_root(root)
    manifest = robustness_manifest(pairs, dataset_hash_value or dataset_hash(), seed)
    _write_manifest(stage_root / "sample_manifest.json", manifest)
    _atomic_text(stage_root / "identity.json", json.dumps(robustness_identity(), indent=2) + "\n")
    selected_variants = select_variants(variants)
    pair_map = {pair["pair_id"]: pair for pair in pairs}
    selected_ids = manifest["pair_ids"][:limit] if limit is not None else manifest["pair_ids"]
    jd_pool = build_jd_pool(pairs)
    base_by_key = {
        (row["workflow"], row["pair_id"]): row for row in base_rows if row.get("repeat", 0) == 0
    }
    results = []
    for workflow in workflows:
        current_hash = _current_generation_hash(workflow)
        saved_hashes = sorted({
            str(row.get("config_hash")) for row in base_rows if row["workflow"] == workflow
        })
        if current_hash is None or saved_hashes != [current_hash]:
            print(
                f"[robustness] warning: {workflow} variants are generated with config hash "
                f"{current_hash}, saved originals use {saved_hashes}; deltas then mix code "
                "drift with input perturbation (see generation_identity_matches_original)",
                flush=True,
            )
        for pair_id in selected_ids:
            base_pair = pair_map[pair_id]
            base_generation = base_by_key.get((workflow, pair_id))
            if base_generation is None:
                print(f"[robustness] {workflow} {pair_id}: no saved original; skipped", flush=True)
                results.append({
                    "workflow": workflow,
                    "base_pair_id": pair_id,
                    "pair_id": f"{pair_id}__{ORIGINAL}",
                    "variant": ORIGINAL,
                    "variant_family": "baseline",
                    "variant_scope": "in_scope",
                    "category": base_pair.get("category", ""),
                    "fit_level": base_pair.get("fit_level", ""),
                    "error": "original generation unavailable",
                    "generation_outcome": "missing_original",
                    "evaluation_failures": {},
                })
                continue
            for variant in selected_variants:
                results.append(_robustness_record(
                    workflow, workflow_registry.get(workflow), variant, base_pair,
                    base_generation, jd_pool, stage_root, root, judge_config,
                    skip_nli, skip_judge, force, components, seed, current_hash,
                ))
    index = _merged_index_rows(stage_root / "evaluations.jsonl", results)
    _write_index(stage_root, index)
    return index


def injection_root(root):
    """Injection artifacts live under the attack identity so attack changes never mix trees."""
    return Path(root) / "injection" / injection_identity()["identity_hash"]


def _injection_diagnostics(pair, output, root, skip_nli=False, force=False):
    """Protected-field, bait and faithfulness checks against the clean (unattacked) source."""
    failures = {}
    diagnostics = {"evaluation_failures": failures}
    try:
        diagnostics.update(metrics.protected_field_metrics(pair["resume"], output))
    except Exception as exc:
        failures["protected_field_violation"] = f"{type(exc).__name__}: {exc}"
        diagnostics["protected_field_violation"] = None
    try:
        diagnostics.update(metrics.fabrication_bait_metrics(
            pair["resume"], output, pair.get("keywords", []),
        ))
    except Exception as exc:
        failures["fabrication_bait_violation"] = f"{type(exc).__name__}: {exc}"
        diagnostics["fabrication_bait_violation"] = None
    if not skip_nli:
        try:
            diagnostics.update(metrics.faithfulness(
                pair["resume"], output, cache_dir=Path(root) / "cache" / "nli", force=force,
            ))
        except Exception as exc:
            failures["faithfulness"] = f"{type(exc).__name__}: {exc}"
            diagnostics["faithfulness"] = None
    return diagnostics


def _injection_record_complete(existing, attack, manifest_hash, skip_nli):
    if existing.get("canary") != attack["canary"] or existing.get("manifest_hash") != manifest_hash:
        return False
    if not existing.get("generation_completed"):
        return True
    return _deterministic_components_complete(existing) and (
        skip_nli or existing.get("nli_evaluator_version") == config.NLI_EVALUATOR_VERSION
    )


def evaluate_injection(
    workflows,
    pairs,
    root,
    dataset_hash_value,
    skip_nli=False,
    force=False,
    limit=None,
    workflow_registry=None,
    generation_config_hashes=None,
):
    """Run the prompt-injection benchmark with a manifest frozen before any generation.

    Layout under ``injection/<identity_hash>/``: ``sample_manifest.json``,
    ``runs/<workflow>/<pair_id>/<channel>/{resume.md,trace.json,generation.json,evaluation.json}``
    and ``evaluations.jsonl``. ``generation.json`` is written last and marks a reusable
    generation; ``evaluation.json`` marks reusable diagnostics. Generation failures keep
    ``injection_success=None`` and are excluded from the success-rate denominator.
    """
    workflow_registry = workflow_registry or WORKFLOWS
    stage_root = injection_root(root)
    identity = injection_identity()
    manifest = injection_manifest(
        pairs, dataset_hash_value, config.INJECTION_SEED, identity["attack_version"],
    )
    # Written (or verified) before any generation so the sample cannot drift mid-run.
    write_injection_manifest(stage_root / "sample_manifest.json", manifest)
    _atomic_text(stage_root / "identity.json", json.dumps(identity, indent=2) + "\n")

    generation_config_hashes = generation_config_hashes or {}
    current_hashes = {}
    for workflow in workflows:
        if workflow in generation_config_hashes:
            from run_all import config_hash as _generation_hash

            current_hashes[workflow] = _generation_hash(workflow)
            if current_hashes[workflow] not in generation_config_hashes[workflow]:
                raise RuntimeError(
                    f"INJECTION_GENERATION_IDENTITY_MISMATCH: {workflow} code/config is "
                    f"{current_hashes[workflow]}, saved generations are "
                    f"{sorted(generation_config_hashes[workflow])}"
                )

    pair_map = {pair["pair_id"]: pair for pair in pairs}
    selected_ids = manifest["pair_ids"][:limit] if limit is not None else manifest["pair_ids"]
    attacks = {(attack["pair_id"], attack["channel"]): attack for attack in manifest["attacks"]}
    results = []
    for workflow in workflows:
        for pair_id in selected_ids:
            pair = pair_map[pair_id]
            for channel in manifest["channels"]:
                attack = attacks[(pair_id, channel)]
                run_dir = stage_root / "runs" / workflow / pair_id / channel
                evaluation_path = run_dir / "evaluation.json"
                generation_path = run_dir / "generation.json"
                if evaluation_path.exists() and not force:
                    existing = _json(evaluation_path)
                    if _injection_record_complete(
                        existing, attack, manifest["manifest_hash"], skip_nli
                    ):
                        results.append(existing)
                        continue
                generation = None
                if generation_path.exists() and not force:
                    saved = _json(generation_path)
                    if saved.get("canary") == attack["canary"]:
                        generation = saved
                        output = (run_dir / "resume.md").read_text(encoding="utf-8")
                if generation is None:
                    tracker = Tracker()
                    generation = run_injection_attack(
                        workflow,
                        workflow_registry[workflow],
                        pair,
                        dataset_hash_value,
                        channel,
                        tracker,
                        identity["attack_version"],
                    )
                    output = generation.pop("resume_md")
                    _atomic_text(run_dir / "resume.md", output)
                    _atomic_text(
                        run_dir / "trace.json",
                        json.dumps(tracker.events, indent=2, ensure_ascii=False) + "\n",
                    )
                    generation.update({
                        **tracker.totals(),
                        "manifest_hash": manifest["manifest_hash"],
                        "injection_identity_hash": identity["identity_hash"],
                        "config_hash": current_hashes.get(workflow),
                        "dataset_hash": dataset_hash_value,
                        "resume_path": str(run_dir / "resume.md"),
                        "trace_path": str(run_dir / "trace.json"),
                    })
                    _atomic_text(
                        generation_path,
                        json.dumps(generation, indent=2, ensure_ascii=False) + "\n",
                    )
                record = {
                    **generation,
                    "category": pair.get("category", ""),
                    "fit_level": pair.get("fit_level", ""),
                    # Shared reporting treats ``error`` as a generation failure.
                    "error": None if generation["generation_completed"] else (
                        generation.get("generation_error") or "generation failed"
                    ),
                    "evaluation_failures": {},
                }
                if generation["generation_completed"]:
                    record.update(_injection_diagnostics(pair, output, root, skip_nli, force))
                _atomic_text(
                    evaluation_path, json.dumps(record, indent=2, ensure_ascii=False) + "\n"
                )
                results.append(record)
                print(
                    f"[injection] {workflow} {pair_id} {channel}: "
                    f"success={record['injection_success']}",
                    flush=True,
                )
    _write_index(stage_root, results)
    return results


@dataclass
class EvaluationRun:
    """Shared state handed to every run-level stage."""

    args: argparse.Namespace
    root: Path
    pairs: list
    pair_map: dict
    generation_rows: list
    dataset_hash: str
    judge_config: JudgeConfig
    components: tuple = ()
    evaluation_rows: list = field(default_factory=list)
    judge_model_info: dict | None = None
    pair_ids: set | None = None

    def resolved_judge_model(self):
        """Judge model metadata resolved once per run, even when single-output judging is skipped."""
        if self.judge_model_info is None or self.judge_model_info.get("status") == "skipped":
            self.judge_model_info = configured_model(self.judge_config)
        return self.judge_model_info


@dataclass(frozen=True)
class Stage:
    """A run-level evaluation stage.

    ``run(run, limit)`` computes rows (resumably) and ``index(root)`` is the file it writes,
    which ``report`` reloads (with ``loader``, JSONL by default) when the stage is not executed
    in this invocation. Rows are passed to ``eval.reporting.STAGE_REPORTERS[name]`` if one is
    registered. ``in_all=False`` keeps a stage out of the ``all`` command.
    """

    name: str
    run: Callable[[EvaluationRun, int | None], list]
    index: Callable[[Path], Path]
    pilot_limit: int
    help: str
    loader: Callable[[Path], list] | None = None
    in_all: bool = True

    def load(self, root):
        path = self.index(root)
        return (self.loader or _read_index)(path)


def _run_robustness(run, limit):
    return evaluate_robustness(
        run.args.workflows,
        run.generation_rows,
        run.pairs,
        run.root,
        run.judge_config,
        skip_nli=run.args.skip_faithfulness,
        skip_judge=run.args.skip_judge,
        force=run.args.force,
        limit=limit,
        components=run.components,
        variants=robustness_variant_selection(run.args),
        dataset_hash_value=run.dataset_hash,
    )


def robustness_variant_selection(args):
    """Variants chosen with ``--robustness-variants`` and/or ``--robustness-families``."""
    return select_variants(
        getattr(args, "robustness_variants", None), getattr(args, "robustness_families", None),
    )


def _run_injection(run, limit):
    saved_hashes = {}
    for row in run.generation_rows:
        saved_hashes.setdefault(row["workflow"], set()).add(row["config_hash"])
    return evaluate_injection(
        run.args.workflows,
        run.pairs,
        run.root,
        run.dataset_hash,
        skip_nli=run.args.skip_faithfulness,
        force=run.args.force,
        limit=limit,
        generation_config_hashes=saved_hashes,
    )


def _parse_config_hashes(values):
    parsed = {}
    for value in values or ():
        workflow, separator, config_hash = value.partition("=")
        if not separator or not workflow or not config_hash:
            raise SystemExit(f"--config-hash expects WORKFLOW=HASH, got {value!r}")
        parsed[workflow] = config_hash
    return parsed


def pairwise_root(root, identity):
    return Path(root) / "pairwise" / identity["evaluator_hash"]


def _pairwise_record_complete(existing, identity):
    """Scored and invalid-output records are final; judge failures are retried on resume."""
    if (existing.get("evaluator") or {}).get("evaluator_hash") != identity["evaluator_hash"]:
        return False
    return existing.get("status") == "scored" or (
        existing.get("status") == "unscored" and existing.get("failure_kind") == FAILURE_INVALID
    )


def evaluate_pairwise_stage(
    generation_rows,
    pair_map,
    root,
    judge_config,
    model_info,
    baseline_workflow=config.PAIRWISE_BASELINE_WORKFLOW,
    challenger_workflows=None,
    config_hashes=None,
    limit=None,
    force=False,
    client=None,
    pair_ids=None,
):
    """Judge every challenger/baseline generation pair in both positions.

    Layout under ``pairwise/<evaluator_hash>/``: ``identity.json``,
    ``records/<challenger>/<pair_id>/repeat_<n>.json`` and ``evaluations.jsonl``; the
    evaluator's own cache is ``pairwise_cache/``. ``generation_unavailable`` records are
    recomputed each run (no judge call), ``judge_failed`` records are retried.
    """
    identity = pairwise_identity(judge_config, model_info)
    stage_root = pairwise_root(root, identity)
    _atomic_text(stage_root / "identity.json", json.dumps(identity, indent=2) + "\n")
    pairs = build_generation_pairs(
        generation_rows,
        baseline_workflow=baseline_workflow,
        challenger_workflows=challenger_workflows,
        config_hashes=config_hashes,
    )
    if pair_ids is not None:
        pairs = [pair for pair in pairs if pair["pair_id"] in pair_ids]
    if limit is not None:
        keep = sorted({pair["pair_id"] for pair in pairs})[:limit]
        pairs = [pair for pair in pairs if pair["pair_id"] in keep]
    records = []
    for pair in pairs:
        record_path = (
            stage_root / "records" / pair["challenger_workflow"] / pair["pair_id"]
            / f"repeat_{pair.get('repeat', 0)}.json"
        )
        if record_path.exists() and not force:
            existing = _json(record_path)
            if _pairwise_record_complete(existing, identity):
                records.append(existing)
                continue
        source = pair_map.get(pair["pair_id"])
        if source is None:
            raise RuntimeError(f"unknown pair {pair['pair_id']}")
        record = evaluate_generation_pair(
            pair,
            source["resume"],
            source["jd"],
            root / "pairwise_cache",
            judge_config=judge_config,
            model_info=model_info,
            client=client,
            force=force,
        )
        record.update({
            "category": source.get("category", ""),
            "fit_level": source.get("fit_level", ""),
            "workflow": pair["challenger_workflow"],
        })
        if record["status"] != FAILURE_GENERATION:
            _atomic_text(record_path, json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        records.append(record)
        print(
            f"[pairwise] {pair['challenger_workflow']} vs {pair['baseline_workflow']} "
            f"{pair['pair_id']}: {record['status']} winner={record.get('consensus_winner')}",
            flush=True,
        )
    index = _merged_index_rows(
        stage_root / "evaluations.jsonl", [_compact_pairwise(record) for record in records],
    )
    _write_index(stage_root, index)
    # Reports cover every comparison saved under this identity, not only this subset.
    return index


_ORDER_SUMMARY_FIELDS = (
    "status", "failure_kind", "role_at_a", "winner_position", "normalized_winner",
)


def _compact_pairwise(record):
    """Index row: drops prompts and raw attempts but keeps what ``summarise_pairwise`` reads."""
    compact = {
        key: value for key, value in record.items() if key not in {"orders", "system_prompt"}
    }
    if record.get("orders"):
        compact["orders"] = {
            name: {field: (order or {}).get(field) for field in _ORDER_SUMMARY_FIELDS}
            for name, order in record["orders"].items()
        }
    return compact


def _pairwise_index(root):
    """Latest pairwise index under ``root`` (the identity is only known after model lookup)."""
    candidates = sorted(
        (Path(root) / "pairwise").glob("*/evaluations.jsonl"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else Path(root) / "pairwise" / "evaluations.jsonl"


def _run_pairwise(run, limit):
    args = run.args
    baseline = args.pairwise_baseline
    selected = set(args.workflows)
    challengers = args.pairwise_challengers or sorted(selected - {baseline})
    missing = sorted(({baseline} | set(challengers)) - selected)
    if missing or not challengers:
        message = (
            f"pairwise needs the baseline and at least one challenger in --workflows; "
            f"missing={missing or ['<challenger>']}"
        )
        if args.command == "pairwise":
            raise SystemExit(message)
        print(f"[pairwise] skipped: {message}", flush=True)
        return _read_index(_pairwise_index(run.root))
    return evaluate_pairwise_stage(
        run.generation_rows,
        run.pair_map,
        run.root,
        run.judge_config,
        run.resolved_judge_model(),
        baseline_workflow=baseline,
        challenger_workflows=challengers,
        config_hashes=_parse_config_hashes(args.config_hash),
        limit=limit,
        force=args.force,
        pair_ids=run.pair_ids,
    )


def review_root(root):
    return Path(root) / "human_review"


def review_forms_started(review_dir):
    """True when any blinded review form already carries a reviewer label."""
    import csv

    for name in ("rubric_ratings.csv", "claim_support_ratings.csv"):
        path = Path(review_dir) / name
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            if any((row.get("reviewer_id") or "").strip() for row in csv.DictReader(handle)):
                return True
    return False


STAGES = {
    "robustness": Stage(
        name="robustness",
        run=_run_robustness,
        index=lambda root: robustness_root(root) / "evaluations.jsonl",
        pilot_limit=2,
        help="seeded noisy, format and out-of-scope input variants on the stratified sample",
    ),
    "injection": Stage(
        name="injection",
        run=_run_injection,
        index=lambda root: injection_root(root) / "evaluations.jsonl",
        pilot_limit=2,
        help="resume/JD prompt-injection canary attacks on the stratified sample (optional)",
        in_all=False,
    ),
    "pairwise": Stage(
        name="pairwise",
        run=_run_pairwise,
        index=_pairwise_index,
        pilot_limit=2,
        help="position-swapped pairwise judging of challengers against the baseline",
    ),
}


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("all", "score", *STAGES, "report"), default="all",
    )
    parser.add_argument("--workflows", nargs="+", default=["w1_single_pass"])
    parser.add_argument("--judge-provider", choices=("ollama", "groq"), default=config.JUDGE_PROVIDER)
    parser.add_argument("--judge-model", default=config.JUDGE_MODEL)
    parser.add_argument(
        "--coverage-provider", choices=("ollama", "groq"),
        help="requirement-coverage evaluator provider (default: judge provider)",
    )
    parser.add_argument(
        "--coverage-model", help="requirement-coverage evaluator model (default: judge model)",
    )
    parser.add_argument("--resume", action="store_true", help="reuse compatible checkpoints (default)")
    parser.add_argument("--force", action="store_true", help="ignore evaluation caches/checkpoints")
    parser.add_argument("--skip-faithfulness", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-coverage", action="store_true")
    parser.add_argument("--pairwise-baseline", default=config.PAIRWISE_BASELINE_WORKFLOW)
    parser.add_argument("--pairwise-challengers", nargs="+")
    parser.add_argument(
        "--config-hash", action="append", metavar="WORKFLOW=HASH",
        help="use only saved generations with this config hash for WORKFLOW (repeatable)",
    )
    parser.add_argument(
        "--pair-ids", nargs="+", metavar="PAIR_ID",
        help="score and pairwise-judge only these dataset pairs (resumable subset)",
    )
    parser.add_argument(
        "--pair-ids-file", help="file with one pair ID per line; combined with --pair-ids",
    )

    parser.add_argument(
        "--robustness-variants", nargs="+", metavar="VARIANT",
        choices=[name for name in VARIANT_FAMILIES if name != ORIGINAL],
        help="robustness variants to run (default: all); original is added for in-scope ones",
    )
    parser.add_argument(
        "--robustness-families", nargs="+", choices=ROBUSTNESS_FAMILIES,
        help="robustness variant families to run; combined with --robustness-variants",
    )
    parser.add_argument("--skip-reports", action="store_true")
    parser.add_argument("--skip-human-review", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--limit", type=int)
    for stage in STAGES.values():
        parser.add_argument(f"--skip-{stage.name}", action="store_true", help=f"skip {stage.help}")
        parser.add_argument(f"--{stage.name}-limit", type=int, help=f"pairs for {stage.help}")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    unknown = sorted(set(args.workflows) - set(WORKFLOWS))
    if unknown:
        raise SystemExit(f"Unknown workflows: {unknown}; registered={sorted(WORKFLOWS)}")
    validation = validate_dataset()
    pairs = load_pairs(split="test")
    pair_map = {pair["pair_id"]: pair for pair in pairs}
    generation_rows = _generation_rows(args.workflows)
    config_hashes = _parse_config_hashes(args.config_hash)
    unknown_hash_workflows = sorted(set(config_hashes) - set(args.workflows))
    if unknown_hash_workflows:
        raise SystemExit(f"--config-hash for unselected workflows: {unknown_hash_workflows}")
    generation_rows = [
        row for row in generation_rows
        if row["workflow"] not in config_hashes
        or row.get("config_hash") == config_hashes[row["workflow"]]
    ]
    _validate_generation_rows(generation_rows, pairs, validation["dataset_hash"])
    pair_ids = _requested_pair_ids(args, pair_map)

    judge_config = JudgeConfig(provider=args.judge_provider, model=args.judge_model)
    model_info = (
        {"provider": judge_config.provider, "tag": judge_config.model, "status": "skipped"}
        if args.skip_judge
        else configured_model(judge_config)
    )
    components = build_output_components(args, judge_config, model_info)
    nli_model = metrics.nli_model_identity()
    identity = _evaluation_identity(judge_config, model_info, nli_model, components)
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
        "output_component_model_info": {
            component.name: getattr(component, "model_info", None)
            for component in components
            if getattr(component, "enabled", True)
        },
    }, indent=2) + "\n")

    if not args.skip_faithfulness and args.command in {"all", "score", "robustness", "injection"}:
        preflight = metrics.nli_preflight()
        _atomic_text(root / "nli_preflight.json", json.dumps(preflight, indent=2) + "\n")

    limit = args.limit
    stage_limits = {name: getattr(args, f"{name}_limit") for name in STAGES}
    if args.pilot:
        limit = limit or 3
        stage_limits = {
            name: value or STAGES[name].pilot_limit for name, value in stage_limits.items()
        }

    run = EvaluationRun(
        args=args,
        root=root,
        pairs=pairs,
        pair_map=pair_map,
        generation_rows=generation_rows,
        dataset_hash=validation["dataset_hash"],
        judge_config=judge_config,
        components=components,
        judge_model_info=model_info,
        pair_ids=pair_ids,
    )
    if args.command in {"all", "score"}:
        evaluate_saved(
            select_pairs(generation_rows, pair_ids),
            pair_map,
            root,
            judge_config,
            skip_nli=args.skip_faithfulness,
            skip_judge=args.skip_judge,
            limit=limit,
            force=args.force,
            components=components,
        )
    # Reports always cover every saved record in this root, not just this invocation's subset.
    run.evaluation_rows = _read_index(root / "evaluations.jsonl")

    stage_rows = {}
    for stage in STAGES.values():
        requested = args.command == stage.name or (args.command == "all" and stage.in_all)
        if requested and not getattr(args, f"skip_{stage.name}"):
            stage_rows[stage.name] = stage.run(run, stage_limits[stage.name])
        else:
            stage_rows[stage.name] = stage.load(root)

    if args.command in {"all", "report"} and not args.skip_reports:
        write_reports(root / "summaries", run.evaluation_rows, stage_rows=stage_rows)
    if args.command in {"all", "report"} and not args.skip_human_review and run.evaluation_rows:
        if review_forms_started(review_root(root)):
            # Never overwrite review forms once a reviewer has started filling them in.
            print("[human_review] completed forms found; packet left unchanged", flush=True)
        else:
            write_review_artifacts(review_root(root), run.evaluation_rows)

    print(json.dumps({
        "output_root": str(root),
        "evaluated": len(run.evaluation_rows),
        **{f"{name}_rows": len(rows) for name, rows in stage_rows.items()},
        "evaluator_hash": identity["evaluator_hash"],
    }, indent=2))
    return root


if __name__ == "__main__":
    main()
