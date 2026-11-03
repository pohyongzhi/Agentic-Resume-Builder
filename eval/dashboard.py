"""Read-only loading and aggregation of saved evaluation results for the Results dashboard.

Nothing here generates, evaluates or calls a model. Every function reads files written by
``evaluate.py`` / ``eval.reporting`` under
``results/evaluations/<dataset_hash>/<generation_scope>/<evaluator_hash>/`` and returns plain
Python data. Stage summaries in ``summaries/`` are preferred; when a summary file is missing the
same table is rebuilt in memory from the saved index with the ``eval.reporting`` functions that
write it, so the numbers match a ``report`` run. Missing stages are reported, never guessed.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import config

WORKFLOW_ORDER = (
    "w1_single_pass",
    "w2_baseline",
    "w3_critic",
    "w4_evidence",
    "w5_fine_tuned",
    "w6_best_of_n",
    "w7_plan_execute",
)
BASELINE = config.PAIRWISE_BASELINE_WORKFLOW

# Plain-language descriptions, written from the workflow modules in workflows/.
WORKFLOW_INFO = {
    "w1_single_pass": {
        "code": "W1",
        "title": "Single pass",
        "role": "Lower bound",
        "summary": (
            "The original resume and the job description go to the model in one prompt with the "
            "shared writing rules. Whatever comes back is the output: no tools, no checks, no retry."
        ),
        "steps": ["Resume + JD", "One model call", "Markdown resume"],
    },
    "w2_baseline": {
        "code": "W2",
        "title": "Agentic baseline",
        "role": "Shared control",
        "summary": (
            "A resume parser and a JD analyser run in parallel. A creator writes a schema-checked "
            "JSON draft, and a harness checks protected facts (name, contact, employers, titles, "
            "dates, degrees) against the parsed resume. Rejected drafts go back with the errors, "
            f"up to {config.MAX_TOOL_CALLS} tool calls; a valid draft is rendered with a fixed "
            "template."
        ),
        "steps": ["Parse resume ‖ Analyse JD", "Draft JSON", "Validate facts ↺", "Render"],
    },
    "w3_critic": {
        "code": "W3",
        "title": "Multi-critic",
        "role": "Review and revise",
        "summary": (
            "Same extraction and first draft as W2. A fact-checker, a recruiter and a formatting "
            "reviewer critique the draft in parallel; the creator revises from their combined "
            f"feedback, at most {config.MAX_CRITIC_ROUNDS} times. The final draft must still pass "
            "the W2 harness before it is rendered."
        ),
        "steps": ["Parse ‖ Analyse", "Draft", "3 critics ‖", "Revise ↺", "Validate + render"],
    },
    "w4_evidence": {
        "code": "W4",
        "title": "Evidence-grounded",
        "role": "Cite every claim",
        "summary": (
            "The resume is split into numbered source spans. Every summary sentence and bullet "
            "must cite the spans it is based on. Citation and rule checks, then a model verifier, "
            "flag unsupported claims; the draft is revised up to twice and anything still "
            "unsupported is dropped before validation and rendering."
        ),
        "steps": ["Parse ‖ Analyse", "Split spans", "Cited draft", "Check + verify ↺",
                  "Drop unsupported", "Render"],
    },
    "w5_fine_tuned": {
        "code": "W5",
        "title": "Fine-tuned",
        "role": "Same prompt, tuned model",
        "summary": (
            "The same single call as W1, but made with the project's fine-tuned small generator "
            "(agentic-resume-v4-1). It isolates what fine-tuning adds without any agent loop."
        ),
        "steps": ["Resume + JD", "One call (fine-tuned)", "Markdown resume"],
    },
    "w6_best_of_n": {
        "code": "W6",
        "title": "Best-of-N",
        "role": "Generate, then select",
        "summary": (
            f"{config.W6_NUM_CANDIDATES} drafts are written with different strategy prompts "
            "(skills first, impact first, JD-wording alignment). Structurally invalid drafts are "
            "rejected, a judge scores the rest on a four-part rubric in both presentation orders, "
            "and the highest-scoring valid draft is kept."
        ),
        "steps": [f"{config.W6_NUM_CANDIDATES} strategy drafts", "Validate", "Judge ×2 orders",
                  "Pick best"],
    },
    "w7_plan_execute": {
        "code": "W7",
        "title": "Plan and execute",
        "role": "Planner + parallel writers",
        "summary": (
            "A planner chooses the sections and their order, maps JD requirements to resume "
            "evidence and says what each section should lead with or cut. One writer per section "
            "drafts in parallel, and an assembler merges them. There are no grounding checks."
        ),
        "steps": ["Plan", "Section writers ‖", "Assemble"],
    },
}


def short_name(workflow: str) -> str:
    info = WORKFLOW_INFO.get(workflow)
    return info["code"] if info else workflow


def display_name(workflow: str) -> str:
    info = WORKFLOW_INFO.get(workflow)
    return f"{info['code']} {info['title']}" if info else workflow


def order_workflows(workflows) -> list[str]:
    """Known workflows in W1-W7 order, then any unknown names alphabetically."""
    present = set(workflows)
    known = [name for name in WORKFLOW_ORDER if name in present]
    return known + sorted(present - set(WORKFLOW_ORDER))


# --- Discovery ---------------------------------------------------------------------------------


def default_evaluations_dir() -> Path:
    """``results/evaluations``; ``ARB_EVALUATIONS_DIR`` points the dashboard elsewhere."""
    override = os.environ.get("ARB_EVALUATIONS_DIR")
    return Path(override) if override else Path(config.RESULTS_FILE).parent / "evaluations"


def _latest_index(directory: Path) -> Path | None:
    candidates = sorted(
        (path for path in directory.glob("*/evaluations.jsonl") if path.is_file()),
        key=lambda path: (path.stat().st_mtime, str(path)),
    )
    return candidates[-1] if candidates else None


def pairwise_index_path(root: Path) -> Path | None:
    """Newest ``pairwise/<hash>/evaluations.jsonl`` (as ``evaluate._pairwise_index`` picks)."""
    return _latest_index(Path(root) / "pairwise")


def robustness_index_path(root: Path) -> Path | None:
    """Newest v2 ``robustness/<hash>/evaluations.jsonl``, else the v1 ``robustness/`` index."""
    directory = Path(root) / "robustness"
    latest = _latest_index(directory)
    if latest:
        return latest
    legacy = directory / "evaluations.jsonl"
    return legacy if legacy.is_file() else None


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _signature(root: Path) -> tuple:
    paths = [root / "evaluations.jsonl", pairwise_index_path(root), robustness_index_path(root)]
    summaries = root / "summaries"
    if summaries.is_dir():
        paths.extend(sorted(summaries.glob("*.json")))
    return tuple((str(path), _mtime(path)) for path in paths if path)


@dataclass(frozen=True)
class EvaluationRoot:
    path: Path
    dataset_hash: str
    generation_scope: str
    evaluator_hash: str
    modified: float
    outputs: int
    workflows: tuple[str, ...]
    stages: tuple[str, ...]

    @property
    def label(self) -> str:
        when = datetime.fromtimestamp(self.modified).strftime("%Y-%m-%d %H:%M")
        codes = ", ".join(short_name(name) for name in self.workflows) or "no workflows"
        stages = " + ".join(self.stages) if self.stages else "no stages"
        return (
            f"{when} · {self.outputs} outputs · {codes} · {stages} · "
            f"{self.generation_scope}/{self.evaluator_hash}"
        )


def _count_index(path: Path) -> tuple[int, tuple[str, ...]]:
    count, workflows = 0, set()
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                count += 1
                workflows.add(row.get("workflow", ""))
    except OSError:
        return 0, ()
    workflows.discard("")
    return count, tuple(order_workflows(workflows))


def root_stages(root: Path) -> tuple[str, ...]:
    root = Path(root)
    stages = []
    if (root / "evaluations.jsonl").is_file():
        stages.append("score")
    if pairwise_index_path(root) or (root / "summaries" / "pairwise.json").is_file():
        stages.append("pairwise")
    if robustness_index_path(root) or (root / "summaries" / "robustness_families.json").is_file():
        stages.append("robustness")
    return tuple(stages)


def discover_roots(base: str | Path | None = None) -> list[EvaluationRoot]:
    """Every evaluation root under ``base``, newest first (by its index or summaries)."""
    base = Path(base) if base is not None else default_evaluations_dir()
    if not base.is_dir():
        return []
    roots = []
    for path in base.glob("*/*/*"):
        if not path.is_dir():
            continue
        index = path / "evaluations.jsonl"
        summaries = path / "summaries"
        if not index.is_file() and not summaries.is_dir():
            continue
        modified = max(
            [_mtime(index)] + [_mtime(item) for item in summaries.glob("*.json")]
            if summaries.is_dir() else [_mtime(index)]
        )
        outputs, workflows = _count_index(index) if index.is_file() else (0, ())
        roots.append(EvaluationRoot(
            path=path,
            dataset_hash=path.parent.parent.name,
            generation_scope=path.parent.name,
            evaluator_hash=path.name,
            modified=modified,
            outputs=outputs,
            workflows=workflows,
            stages=root_stages(path),
        ))
    roots.sort(key=lambda root: (root.modified, str(root.path)), reverse=True)
    return roots


# --- Loading -----------------------------------------------------------------------------------


def read_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    """Rows of a JSONL file; blank or truncated lines (a run still writing) are skipped."""
    if not path:
        return []
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@dataclass
class EvaluationBundle:
    root: Path
    rows: list[dict[str, Any]] = field(default_factory=list)
    summaries: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    pairwise_rows: list[dict[str, Any]] = field(default_factory=list)
    robustness_rows: list[dict[str, Any]] = field(default_factory=list)
    identity: dict[str, Any] = field(default_factory=dict)
    # Tables rebuilt in memory when a summary file is missing (computed once per bundle).
    derived: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def workflows(self) -> list[str]:
        names = {row.get("workflow") for row in self.rows if row.get("workflow")}
        if not names:
            names = {
                row.get("workflow") for row in self.summaries.get("overall", [])
                if row.get("workflow")
            }
        return order_workflows(names)

    @property
    def has_pairwise(self) -> bool:
        return bool(self.pairwise_rows or self.summaries.get("pairwise"))

    @property
    def has_robustness(self) -> bool:
        return bool(
            self.robustness_rows
            or self.summaries.get("robustness_families")
            or self.summaries.get("robustness_out_of_scope_families")
        )


_CACHE: dict[str, tuple[tuple, EvaluationBundle]] = {}
_CACHE_LIMIT = 4


def load_bundle(root: str | Path) -> EvaluationBundle:
    """Load one evaluation root (cached until any of its index or summary files change)."""
    root = Path(root)
    key = str(root.resolve())
    signature = _signature(root)
    cached = _CACHE.get(key)
    if cached and cached[0] == signature:
        return cached[1]
    summaries = {}
    summary_dir = root / "summaries"
    if summary_dir.is_dir():
        for path in sorted(summary_dir.glob("*.json")):
            table = _read_json(path)
            if isinstance(table, list):
                summaries[path.stem] = table
    bundle = EvaluationBundle(
        root=root,
        rows=read_jsonl(root / "evaluations.jsonl"),
        summaries=summaries,
        pairwise_rows=read_jsonl(pairwise_index_path(root)),
        robustness_rows=read_jsonl(robustness_index_path(root)),
        identity=_read_json(root / "evaluator_identity.json") or {},
    )
    if len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = (signature, bundle)
    return bundle


def clear_cache():
    _CACHE.clear()


# --- Per-output metric summaries (leaderboard) --------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    better: str          # "higher" or "lower"
    kind: str            # "rate" (0-1 shown as %), "score" (0-1), "scale" (1-5), "number"
    tier: str = "quality"


LEADERBOARD_METRICS = (
    MetricSpec("task_success", "Task success", "higher", "rate"),
    MetricSpec("claim_faithfulness", "Claim faithfulness", "higher", "score"),
    MetricSpec("unsupported_claim_rate", "Unsupported claims", "lower", "rate"),
    MetricSpec("protected_field_violation", "Protected-field violations", "lower", "rate"),
    MetricSpec("fabrication_bait_violation", "Fabrication bait", "lower", "rate"),
    MetricSpec("requirement_coverage", "Requirement coverage", "higher", "score"),
    MetricSpec("kw_after", "Keyword coverage", "higher", "score", "tier2"),
    MetricSpec("sim_after", "JD similarity", "higher", "score", "tier2"),
    MetricSpec("judge_overall_mean", "Rubric judge (1-5)", "higher", "scale", "tier2"),
    MetricSpec("total_tokens", "Tokens / output", "lower", "number", "cost"),
    MetricSpec("historical_run_wall_seconds", "Wall time (s)", "lower", "number", "cost"),
)
METRIC_SPECS = {spec.key: spec for spec in LEADERBOARD_METRICS}
# Older records score only ``faithfulness``; it is the same claim-support ratio.
METRIC_FALLBACKS = {
    "claim_faithfulness": "faithfulness",
    "historical_run_wall_seconds": "generation_model_seconds",
}
_SUMMARY_METRICS = tuple(
    dict.fromkeys(
        [spec.key for spec in LEADERBOARD_METRICS] + list(METRIC_FALLBACKS.values())
    )
)


def filter_options(bundle: EvaluationBundle) -> tuple[list[str], list[str]]:
    """Categories and fit levels present in the scored rows (or the stratified summaries)."""
    categories = {row.get("category") for row in bundle.rows if row.get("category")}
    fit_levels = {row.get("fit_level") for row in bundle.rows if row.get("fit_level")}
    if not bundle.rows:
        categories = {r.get("category") for r in bundle.summaries.get("by_category", [])}
        fit_levels = {r.get("fit_level") for r in bundle.summaries.get("by_fit_level", [])}
    fit_rank = {"strong": 0, "moderate": 1, "weak": 2}
    return (
        sorted(value for value in categories if value),
        sorted((value for value in fit_levels if value), key=lambda v: (fit_rank.get(v, 9), v)),
    )


def _index_summary(table, filters):
    output: dict[str, dict[str, dict]] = {}
    for row in table:
        if any(row.get(key) != value for key, value in filters.items()):
            continue
        output.setdefault(row.get("workflow", ""), {})[row.get("metric")] = row
    return output


def metric_summaries(
    bundle: EvaluationBundle, category: str | None = None, fit_level: str | None = None,
) -> tuple[dict[str, dict[str, dict]], str]:
    """``{workflow: {metric: summary row}}`` for the filter, plus where it came from.

    Uses ``summaries/overall|by_category|by_fit_level.json`` when they cover the filter; both
    filters at once (or missing summaries) are rebuilt from ``evaluations.jsonl`` with
    ``eval.reporting.summarise`` so eligible/scored/unscored keep their meaning.
    """
    category = category or None
    fit_level = fit_level or None
    name, filters = "overall", {}
    if category and not fit_level:
        name, filters = "by_category", {"category": category}
    elif fit_level and not category:
        name, filters = "by_fit_level", {"fit_level": fit_level}
    elif category and fit_level:
        name = None
    if name and bundle.summaries.get(name):
        indexed = _index_summary(bundle.summaries[name], filters)
        if indexed:
            return indexed, f"summaries/{name}.json"
    if not bundle.rows:
        return {}, "none"
    from eval.reporting import summarise

    rows = [
        row for row in bundle.rows
        if (not category or row.get("category") == category)
        and (not fit_level or row.get("fit_level") == fit_level)
    ]
    return _index_summary(summarise(rows, ("workflow",), _SUMMARY_METRICS), {}), "evaluations.jsonl"


def metric_cell(summary: dict[str, dict], metric: str) -> dict[str, Any]:
    """One leaderboard cell: value (None when unscored), its denominators and display text."""
    row = summary.get(metric)
    fallback = METRIC_FALLBACKS.get(metric)
    if (not row or not row.get("scored")) and fallback and summary.get(fallback):
        if summary[fallback].get("scored"):
            row = summary[fallback]
    if not row:
        return {"value": None, "eligible": 0, "scored": 0, "unscored": 0, "status": "missing",
                "ci": (None, None), "numerator": None}
    value = row.get("mean")
    scored = int(row.get("scored") or 0)
    status = "scored" if scored and value is not None else (
        "unscored" if int(row.get("eligible") or 0) else "not_applicable"
    )
    return {
        "value": float(value) if status == "scored" else None,
        "eligible": int(row.get("eligible") or 0),
        "scored": scored,
        "unscored": int(row.get("unscored") or 0),
        "status": status,
        "ci": (row.get("ci_95_low"), row.get("ci_95_high")),
        "numerator": row.get("numerator"),
    }


def leaderboard(bundle, category=None, fit_level=None, metrics=LEADERBOARD_METRICS):
    """Rows per workflow (W1-W7 order) with a ``metric_cell`` per metric, plus the source."""
    summaries, source = metric_summaries(bundle, category, fit_level)
    rows = []
    for workflow in order_workflows(summaries):
        rows.append({
            "workflow": workflow,
            "cells": {spec.key: metric_cell(summaries[workflow], spec.key) for spec in metrics},
        })
    return rows, source


def metric_is_present(board: list[dict], metric: str) -> bool:
    return any(row["cells"][metric]["scored"] for row in board)


def tradeoff_points(board: list[dict], x_metric: str, y_metric: str) -> list[dict[str, Any]]:
    """Scatter points for workflows scored on both metrics (unscored ones are left out)."""
    points = []
    for row in board:
        x, y = row["cells"][x_metric], row["cells"][y_metric]
        if x["value"] is None or y["value"] is None:
            continue
        points.append({
            "workflow": row["workflow"], "x": x["value"], "y": y["value"],
            "x_n": x["scored"], "y_n": y["scored"],
        })
    return points


HEADLINE_METRICS = (
    ("task_success", "Highest task success"),
    ("claim_faithfulness", "Most faithful"),
    ("requirement_coverage", "Best requirement coverage"),
    ("total_tokens", "Cheapest per output"),
)


def headline_findings(board: list[dict]) -> list[dict[str, Any]]:
    """Best scored workflow per headline metric, with W2's value for comparison.

    Ties keep W1-W7 order. A metric nobody scored yields ``workflow=None``.
    """
    findings = []
    for metric, title in HEADLINE_METRICS:
        spec = METRIC_SPECS[metric]
        scored = [row for row in board if row["cells"][metric]["value"] is not None]
        baseline = next((row for row in board if row["workflow"] == BASELINE), None)
        if not scored:
            findings.append({"metric": metric, "title": title, "workflow": None})
            continue
        sign = 1 if spec.better == "higher" else -1
        best = max(scored, key=lambda row: sign * row["cells"][metric]["value"])
        value = best["cells"][metric]["value"]
        ties = [
            row["workflow"] for row in scored
            if row is not best and abs(row["cells"][metric]["value"] - value) < 1e-9
        ]
        findings.append({
            "metric": metric,
            "title": title,
            "workflow": best["workflow"],
            "value": best["cells"][metric]["value"],
            "scored": best["cells"][metric]["scored"],
            "ties": ties,
            "baseline": baseline["cells"][metric]["value"] if baseline else None,
        })
    return findings


# --- Workflow profiles (explainer cards) -------------------------------------------------------


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def workflow_profiles(bundle: EvaluationBundle) -> dict[str, dict[str, Any]]:
    """Measured averages per workflow from the saved per-output rows."""
    grouped: dict[str, list[dict]] = {}
    for row in bundle.rows:
        grouped.setdefault(row.get("workflow", ""), []).append(row)
    profiles = {}
    for workflow, rows in grouped.items():
        if not workflow:
            continue
        task = [row.get("task_success") for row in rows]
        task_scored = [value for value in task if value is not None]
        tokens = [
            row.get("total_tokens")
            if row.get("total_tokens") is not None
            else (
                (row.get("input_tokens") or 0) + (row.get("output_tokens") or 0)
                if row.get("input_tokens") is not None else None
            )
            for row in rows
        ]
        profiles[workflow] = {
            "outputs": len(rows),
            "generation_failures": sum(bool(row.get("error")) for row in rows),
            "llm_calls": _mean(row.get("llm_calls") for row in rows),
            "tool_calls": _mean(row.get("tool_calls") for row in rows),
            "tokens": _mean(tokens),
            "wall_seconds": _mean(
                row.get("wall_seconds") if row.get("wall_seconds") is not None
                else row.get("seconds")
                for row in rows
            ),
            "task_success": sum(bool(value) for value in task_scored),
            "task_scored": len(task_scored),
            "models": sorted({row.get("model") for row in rows if row.get("model")}),
        }
    return profiles


# --- Pairwise ----------------------------------------------------------------------------------


def _decode(value):
    if isinstance(value, str) and value[:1] in "{[":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def pairwise_table(bundle: EvaluationBundle, baseline: str = BASELINE) -> list[dict[str, Any]]:
    """``summaries/pairwise.json`` rows against ``baseline`` (rebuilt from the index if absent)."""
    table = bundle.summaries.get("pairwise")
    if not table and bundle.pairwise_rows:
        if "pairwise" not in bundle.derived:
            from eval.reporting import pairwise_summary

            bundle.derived["pairwise"] = pairwise_summary(
                bundle.pairwise_rows, ("baseline_workflow", "challenger_workflow")
            )
        table = bundle.derived["pairwise"]
    rows = [
        {key: _decode(value) for key, value in row.items()}
        for row in (table or [])
        if row.get("baseline_workflow", baseline) == baseline
    ]
    order = {name: index for index, name in enumerate(order_workflows(
        row.get("challenger_workflow", "") for row in rows
    ))}
    return sorted(rows, key=lambda row: order.get(row.get("challenger_workflow", ""), 99))


PAIRWISE_OUTCOME_LABELS = {
    "challenger": "win",
    "baseline": "loss",
    "tie": "tie",
}


def pairwise_outcome(record: dict[str, Any] | None) -> str:
    """Per-pair result for the challenger: win / loss / tie / inconsistent / unscored / unavailable."""
    if not record:
        return "not judged"
    status = record.get("status")
    if status == config.PAIRWISE_FAILURE_GENERATION:
        return "unavailable"
    if status != "scored":
        return "unscored"
    if not record.get("position_swap_consistent"):
        return "inconsistent"
    return PAIRWISE_OUTCOME_LABELS.get(record.get("consensus_winner"), "unscored")


# --- Robustness --------------------------------------------------------------------------------


def _robustness_tables(bundle):
    """(families, out_of_scope_families, variants, out_of_scope_variants, deltas)."""
    if "robustness" not in bundle.derived:
        bundle.derived["robustness"] = _build_robustness_tables(bundle)
    return bundle.derived["robustness"]


def _build_robustness_tables(bundle):
    s = bundle.summaries
    if s.get("robustness_families") is not None or s.get("robustness_out_of_scope_families") is not None:
        return (
            s.get("robustness_families") or [],
            s.get("robustness_out_of_scope_families") or [],
            s.get("robustness_variants") or [],
            s.get("robustness_out_of_scope") or [],
            s.get("robustness_deltas") or [],
        )
    rows = bundle.robustness_rows
    if not rows:
        return [], [], [], [], []
    from eval.reporting import (
        robustness_delta_table,
        robustness_family_tables,
        robustness_variant_tables,
    )

    try:
        families, oos_families = robustness_family_tables(rows)
        variants, oos_variants = robustness_variant_tables(rows)
        deltas = robustness_delta_table(rows)
    except (KeyError, TypeError, ValueError):
        return [], [], [], [], []
    return families, oos_families, variants, oos_variants, deltas


ROBUSTNESS_FAMILIES = (
    ("noisy", "Noisy input", "stability"),
    ("format", "Format changes", "stability"),
    ("out_of_scope", "Out of scope", "expected_behaviour"),
)


def robustness_family_panels(bundle: EvaluationBundle) -> list[dict[str, Any]]:
    """One panel per variant family with a rate and its denominator per workflow.

    In-scope families report ``stability_score = stable / scored``; out of scope reports
    ``expected_behaviour_pass_rate = passes / scored``. Unscored workflows keep ``value=None``.
    """
    families, oos_families, *_ = _robustness_tables(bundle)
    panels = []
    for family, title, measure in ROBUSTNESS_FAMILIES:
        source = oos_families if measure == "expected_behaviour" else families
        items = []
        for row in source:
            if row.get("family") != family:
                continue
            if measure == "stability":
                value, numerator = row.get("stability_score"), row.get("stable")
                note = (
                    f"{row.get('failed', 0)} failed of {row.get('eligible', 0)} eligible"
                    if row.get("failed") else ""
                )
            else:
                value = row.get("expected_behaviour_pass_rate")
                numerator = row.get("expected_behaviour_passes")
                note = f"{row.get('crashed', 0)} crashed" if row.get("crashed") else ""
            items.append({
                "workflow": row.get("workflow", ""),
                "value": None if value is None else float(value),
                "numerator": numerator,
                "denominator": row.get("scored", 0),
                "runs": row.get("runs", 0),
                "variants": row.get("variant", ""),
                "note": note,
            })
        rank = {name: index for index, name in enumerate(order_workflows(
            item["workflow"] for item in items
        ))}
        items.sort(key=lambda item: rank[item["workflow"]])
        panels.append({"family": family, "title": title, "measure": measure, "items": items})
    return panels


DELTA_METRIC_LABELS = {
    "faithfulness": "Claim faithfulness",
    "unsupported_claim_rate": "Unsupported claims",
    "task_success": "Task success",
    "requirement_coverage": "Requirement coverage",
    "protected_field_violation": "Protected-field violations",
    "fabrication_bait_violation": "Fabrication bait",
    "kw_after": "Keyword coverage",
    "sim_after": "JD similarity",
    "judge_overall_mean": "Rubric judge",
}


def robustness_delta_matrix(bundle: EvaluationBundle, metric: str = "faithfulness") -> dict:
    """Workflow x in-scope variant mean deltas (variant minus original) for one metric."""
    *_, deltas = _robustness_tables(bundle)
    rows = [row for row in deltas if row.get("metric") == metric]
    variants = []
    for row in rows:
        key = (row.get("family", ""), row.get("variant", ""))
        if key not in variants:
            variants.append(key)
    family_rank = {"noisy": 0, "format": 1}
    variants.sort(key=lambda key: (family_rank.get(key[0], 9), key[1]))
    cells = {(row.get("workflow"), row.get("variant")): row for row in rows}
    direction = next((row.get("direction") for row in rows), "higher_is_better")
    return {
        "metric": metric,
        "direction": direction,
        "variants": variants,
        "workflows": order_workflows({row.get("workflow") for row in rows}),
        "cells": cells,
    }


def robustness_delta_metrics(bundle: EvaluationBundle) -> list[str]:
    *_, deltas = _robustness_tables(bundle)
    present = {row.get("metric") for row in deltas if row.get("scored")}
    return [metric for metric in DELTA_METRIC_LABELS if metric in present]


def out_of_scope_variant_rows(bundle: EvaluationBundle) -> list[dict[str, Any]]:
    *_, oos_variants, _ = _robustness_tables(bundle)
    rank = {name: index for index, name in enumerate(WORKFLOW_ORDER)}
    return sorted(
        oos_variants, key=lambda row: (row.get("variant", ""), rank.get(row.get("workflow"), 99))
    )


# --- Output viewer -----------------------------------------------------------------------------


def _primary_rows(rows):
    """One row per (workflow, pair): the lowest repeat."""
    chosen: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row.get("workflow", ""), row.get("pair_id", ""))
        if key not in chosen or (row.get("repeat") or 0) < (chosen[key].get("repeat") or 0):
            chosen[key] = row
    return chosen


def pair_choices(bundle: EvaluationBundle) -> list[tuple[str, str]]:
    """``(label, pair_id)`` for every pair with at least one saved output."""
    pairs: dict[str, dict] = {}
    for row in bundle.rows:
        pair_id = row.get("pair_id")
        if pair_id:
            entry = pairs.setdefault(pair_id, {"row": row, "workflows": set()})
            entry["workflows"].add(row.get("workflow"))
    choices = []
    for pair_id in sorted(pairs):
        row = pairs[pair_id]["row"]
        parts = [pair_id]
        if row.get("category"):
            parts.append(row["category"].replace("_", " "))
        if row.get("fit_level"):
            parts.append(f"{row['fit_level']} fit")
        parts.append(f"{len(pairs[pair_id]['workflows'])} outputs")
        choices.append((" · ".join(parts), pair_id))
    return choices


def _source_texts(pair_id: str, row: dict) -> tuple[str, str]:
    resume, jd = row.get("original_resume") or "", row.get("job_description") or ""
    if resume and jd:
        return resume, jd
    try:
        from common.data import load_pairs

        for pair in load_pairs():
            if pair["pair_id"] == pair_id:
                return resume or pair.get("resume", ""), jd or pair.get("jd", "")
    except (OSError, ValueError, KeyError):
        pass
    return resume, jd


def _output_text(row: dict) -> str:
    text = row.get("resume_md")
    if text is None and row.get("resume_path"):
        try:
            text = Path(row["resume_path"]).read_text(encoding="utf-8")
        except OSError:
            text = None
    return text or ""


def pair_view(bundle: EvaluationBundle, pair_id: str) -> dict[str, Any] | None:
    """Original inputs plus each workflow's output and key scores for one pair."""
    primary = _primary_rows(row for row in bundle.rows if row.get("pair_id") == pair_id)
    if not primary:
        return None
    pairwise = {}
    for record in bundle.pairwise_rows:
        if record.get("pair_id") != pair_id or record.get("baseline_workflow", BASELINE) != BASELINE:
            continue
        key = record.get("challenger_workflow")
        if key not in pairwise or (record.get("repeat") or 0) < (pairwise[key].get("repeat") or 0):
            pairwise[key] = record
    first = next(iter(primary.values()))
    resume, jd = _source_texts(pair_id, first)
    outputs = {}
    for (workflow, _), row in primary.items():
        faithfulness = row.get("claim_faithfulness")
        if faithfulness is None:
            faithfulness = row.get("faithfulness")
        outputs[workflow] = {
            "resume_md": _output_text(row),
            "error": row.get("error"),
            "claim_faithfulness": faithfulness,
            "unsupported_claim_rate": row.get("unsupported_claim_rate"),
            "requirement_coverage": row.get("requirement_coverage"),
            "requirement_coverage_status": row.get("requirement_coverage_status"),
            "protected_field_violation_count": row.get("protected_field_violation_count"),
            "fabrication_bait_match_count": row.get("fabrication_bait_match_count"),
            "task_success": row.get("task_success"),
            "total_tokens": row.get("total_tokens"),
            "wall_seconds": row.get("wall_seconds", row.get("seconds")),
            "pairwise": (
                "baseline" if workflow == BASELINE
                else pairwise_outcome(pairwise.get(workflow)) if bundle.pairwise_rows
                else None
            ),
        }
    return {
        "pair_id": pair_id,
        "category": first.get("category", ""),
        "fit_level": first.get("fit_level", ""),
        "role_family": first.get("role_family", ""),
        "seniority": first.get("seniority", ""),
        "original_resume": resume,
        "job_description": jd,
        "outputs": {name: outputs[name] for name in order_workflows(outputs)},
    }
