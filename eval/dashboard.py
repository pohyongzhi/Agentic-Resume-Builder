"""Read-only loading and aggregation of saved evaluation results for the Results dashboard.

Nothing here generates, evaluates or calls a model. Every function reads files written by
``evaluate.py`` / ``eval.reporting`` under
``results/evaluations/<dataset_hash>/<generation_scope>/<evaluator_hash>/`` and returns plain
Python data. Stage summaries in ``summaries/`` are preferred; when a summary file is missing the
same table is rebuilt in memory from the saved index with the ``eval.reporting`` functions that
write it, so the numbers match a ``report`` run. Missing stages are reported, never guessed.

Results follow the final criteria: a resume is good when its NLI claim faithfulness against the
original resume reaches the truth gate and its rubric v3 overall score (``eval.quality_rubric``)
reaches the writing gate; a workflow's result is good resumes out of every attempt. Rubric v3
records are read with ``eval.dashboard_charts.load_rubric`` from the ``quality-rubric`` folder next
to the evaluation root, so this tab and the "Results (charts)" tab count the same resumes.
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


# Rubric v3 dimensions (``eval.quality_rubric.DIMENSIONS``), each scored 1-5.
RUBRIC_DIMENSION_LABELS = {
    "quantified_evidence": "Quantified evidence",
    "clarity": "Clarity",
    "concision": "Concision",
    "logical_order": "Logical order",
}
# Column groups of the leaderboard, in display order.
METRIC_TIERS = {
    "final": "Final result",
    "truth": "Truth: NLI and guardrails",
    "writing": "Writing: rubric v3",
    "cost": "Cost",
}
# Only final-criteria metrics: rubric v1 (``judge_*``), requirement coverage, keyword coverage,
# JD similarity, pairwise and task success are superseded and deliberately left out.
LEADERBOARD_METRICS = (
    MetricSpec("good_rate", "Good resumes", "higher", "rate", "final"),
    MetricSpec("nli_pass_rate", "NLI pass rate", "higher", "rate", "truth"),
    MetricSpec("claim_faithfulness", "Claim faithfulness", "higher", "score", "truth"),
    MetricSpec("unsupported_claim_rate", "Unsupported claims", "lower", "rate", "truth"),
    MetricSpec("fabrication_bait_violation", "Fabrication bait", "lower", "rate", "truth"),
    MetricSpec("protected_field_violation", "Protected-field violations", "lower", "rate", "truth"),
    MetricSpec("rubric_pass_rate", "Rubric pass rate", "higher", "rate", "writing"),
    MetricSpec("rubric_overall", "Rubric v3 overall", "higher", "scale", "writing"),
    *(
        MetricSpec(f"rubric_{name}", label, "higher", "scale", "writing")
        for name, label in RUBRIC_DIMENSION_LABELS.items()
    ),
    MetricSpec("total_tokens", "Tokens / output", "lower", "number", "cost"),
    MetricSpec("historical_run_wall_seconds", "Wall time (s)", "lower", "number", "cost"),
)
METRIC_SPECS = {spec.key: spec for spec in LEADERBOARD_METRICS}
# Computed here from the per-output rows and the rubric v3 records (not in ``summaries/``).
FINAL_METRICS = (
    "good_rate", "nli_pass_rate", "rubric_pass_rate", "rubric_overall",
    *(f"rubric_{name}" for name in RUBRIC_DIMENSION_LABELS),
)
# Older records score only ``faithfulness``; it is the same claim-support ratio.
METRIC_FALLBACKS = {
    "claim_faithfulness": "faithfulness",
    "historical_run_wall_seconds": "generation_model_seconds",
}
_SUMMARY_METRICS = tuple(
    dict.fromkeys(
        [spec.key for spec in LEADERBOARD_METRICS if spec.key not in FINAL_METRICS]
        + list(METRIC_FALLBACKS.values())
    )
)
# Verdict of one resume under the final criteria (``eval.dashboard_charts.verdict``).
VERDICT_LABELS = {
    "good": "good",
    "nli_fail": "failed truth",
    "rubric_fail": "failed writing",
    "unscored": "unscored",
    "rejected": "no resume",
}
_EPSILON = 1e-9  # the same tolerance ``eval.dashboard_charts`` applies at the gates


# --- Final criteria: rubric v3 and good resumes -------------------------------------------------


def _rubric_signature(root: Path) -> tuple:
    """Modification times that change when a rubric v3 run is added, extended or re-summarised."""
    from eval import dashboard_charts as dc

    paths = []
    for base in dc._rubric_candidates(Path(root)):
        for run in sorted(base.glob("*")):
            paths += [run / "identity.json", run / "summary.json",
                      *sorted((run / "records").glob("*"))]
    return tuple((str(path), _mtime(path)) for path in paths)


def rubric_scores(bundle: EvaluationBundle) -> tuple[dict, dict]:
    """Rubric v3 records by ``(workflow, pair_id)`` and the run's metadata for this bundle.

    The run is chosen and read by ``eval.dashboard_charts.load_rubric`` (the rubric v3 run with
    the most records for the dataset); ``meta["gates"]`` holds the truth and writing gates
    (``{"nli": 0.8, "rubric": 4.0}`` unless the run's summary says otherwise). Reading the
    records takes a moment, so they are kept on the bundle until the run's files change.
    """
    from eval import dashboard_charts as dc

    signature = _rubric_signature(bundle.root)
    cached = bundle.derived.get("rubric")
    if cached and cached[0] == signature:
        return cached[1], cached[2]
    records, meta = dc.load_rubric(bundle.root)
    bundle.derived["rubric"] = (signature, records, meta)
    return records, meta


def final_gates(bundle: EvaluationBundle) -> dict[str, float]:
    """``{"nli": truth gate, "rubric": writing gate}`` used for this bundle."""
    return dict(rubric_scores(bundle)[1]["gates"])


def faithfulness_value(row: dict) -> float | None:
    """Claim faithfulness of one output (older records carry only ``faithfulness``)."""
    value = row.get("claim_faithfulness")
    return row.get("faithfulness") if value is None else value


def _rubric_value(record: dict | None, metric: str = "overall"):
    """Rubric v3 ``overall`` or one dimension score of a record; None when not scored."""
    if not record:
        return None
    value = record.get("overall") if metric == "overall" else (record.get("scores") or {}).get(metric)
    return None if value is None or isinstance(value, bool) else float(value)


def resume_verdict(row: dict, record: dict | None, gates: dict) -> str:
    """good / nli_fail / rubric_fail / unscored / rejected, as ``eval.dashboard_charts.verdict``."""
    from eval import dashboard_charts as dc

    return dc.verdict({**row, "claim_faithfulness": faithfulness_value(row)}, record, gates)


def _passes(value, gate) -> bool:
    return value >= gate - _EPSILON


def good_resume_counts(rows, records: dict, gates: dict) -> dict[str, Any]:
    """Verdict counts for one workflow's outputs (one row per pair).

    As in the final count, an unscored resume is not good. ``good_scored`` is False, and the
    good-resume rate unknown rather than 0, when no rubric v3 run exists or when resumes are
    waiting on a gate (unscored) while none of the workflow's resumes got a writing-gate verdict.
    """
    rows = list(rows)
    verdicts = {kind: 0 for kind in VERDICT_LABELS}
    for row in rows:
        record = records.get((row.get("workflow", ""), row.get("pair_id", "")))
        verdicts[resume_verdict(row, record, gates)] += 1
    decided = verdicts["good"] + verdicts["rubric_fail"]
    return {
        "attempts": len(rows),
        "verdicts": verdicts,
        "good": verdicts["good"],
        "good_scored": bool(records and rows) and not (verdicts["unscored"] and not decided),
    }


def _summary_row(metric, eligible, values=None, numerator=None, scored=None):
    """A row shaped like ``eval.reporting.summarise`` output: a rate (``numerator``/``scored``,
    Wilson interval) or a mean of ``values`` (bootstrap interval with the report's seed)."""
    from eval.reporting import _bootstrap_ci, _wilson_ci

    if numerator is not None:
        mean = numerator / scored if scored else None
        low, high = _wilson_ci(numerator, scored)
    else:
        values = [float(value) for value in values or () if value is not None]
        scored = len(values)
        mean = sum(values) / scored if scored else None
        low, high = _bootstrap_ci(
            values, config.REPORT_BOOTSTRAP_SEED, config.REPORT_BOOTSTRAP_SAMPLES
        )
    return {
        "metric": metric,
        "eligible": eligible,
        "scored": scored,
        "unscored": eligible - scored,
        "numerator": numerator if scored else None,
        "denominator": scored if numerator is not None else None,
        "mean": round(mean, 6) if mean is not None else None,
        "ci_95_low": round(low, 6) if low is not None else None,
        "ci_95_high": round(high, 6) if high is not None else None,
    }


def final_summaries(
    bundle: EvaluationBundle, category: str | None = None, fit_level: str | None = None,
) -> dict[str, dict[str, dict]]:
    """``{workflow: {metric: summary row}}`` for ``FINAL_METRICS``, from one output per pair.

    * ``good_rate``: good resumes / attempts (a failed generation or an unscored resume is not
      good), with the verdict counts in ``verdicts``; unknown (``mean`` None) when
      ``good_resume_counts`` cannot decide it (no rubric v3 run, or no writing-gate verdict).
    * ``nli_pass_rate``: claim faithfulness >= the truth gate, over outputs NLI scored.
    * ``rubric_pass_rate``: rubric v3 overall >= the writing gate, over outputs the judge scored.
    * ``rubric_overall`` and ``rubric_<dimension>``: means over outputs the judge scored.

    Failed generations are not eligible for the per-resume measures; unscored outputs are
    counted as unscored, never as 0. Empty when the bundle has no per-output rows.
    """
    records, meta = rubric_scores(bundle)
    gates = meta["gates"]
    primary = _primary_rows(
        row for row in bundle.rows
        if (not category or row.get("category") == category)
        and (not fit_level or row.get("fit_level") == fit_level)
    )
    grouped: dict[str, list[dict]] = {}
    for (workflow, _), row in primary.items():
        if workflow:
            grouped.setdefault(workflow, []).append(row)
    output = {}
    for workflow, rows in grouped.items():
        counts = good_resume_counts(rows, records, gates)
        done = [row for row in rows if not row.get("error")]
        judged = [records.get((workflow, row.get("pair_id", ""))) for row in done]
        faith = [faithfulness_value(row) for row in done]
        faith = [value for value in faith if value is not None]
        overall = [_rubric_value(record) for record in judged]
        overall = [value for value in overall if value is not None]
        attempts = counts["attempts"]
        good = (
            _summary_row("good_rate", attempts, numerator=counts["good"], scored=attempts)
            if counts["good_scored"]
            else _summary_row("good_rate", attempts, values=[])
        )
        good["verdicts"] = counts["verdicts"]
        metrics = {
            "good_rate": good,
            "nli_pass_rate": _summary_row(
                "nli_pass_rate", len(done),
                numerator=sum(_passes(value, gates["nli"]) for value in faith), scored=len(faith),
            ),
            "rubric_pass_rate": _summary_row(
                "rubric_pass_rate", len(done),
                numerator=sum(_passes(value, gates["rubric"]) for value in overall),
                scored=len(overall),
            ),
            "rubric_overall": _summary_row("rubric_overall", len(done), values=overall),
        }
        for name in RUBRIC_DIMENSION_LABELS:
            metrics[f"rubric_{name}"] = _summary_row(
                f"rubric_{name}", len(done),
                values=[_rubric_value(record, name) for record in judged],
            )
        output[workflow] = metrics
    return output


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
    ``eval.reporting.summarise`` so eligible/scored/unscored keep their meaning. The final
    criteria (``FINAL_METRICS``) are added from ``final_summaries``. Kept on the bundle per
    filter until the rubric v3 records are reloaded.
    """
    category = category or None
    fit_level = fit_level or None
    records, meta = rubric_scores(bundle)
    key = ("summaries", category, fit_level)
    cached = bundle.derived.get(key)
    if cached and cached[0] is records:
        return cached[1]
    summaries, source = _saved_summaries(bundle, category, fit_level)
    final = final_summaries(bundle, category, fit_level)
    for workflow, metrics in final.items():
        summaries.setdefault(workflow, {}).update(metrics)
    if records and final:
        source += f" + rubric v3 run {Path(meta['path']).name}"
    bundle.derived[key] = (records, (summaries, source))
    return summaries, source


def _saved_summaries(bundle, category, fit_level):
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
                "ci": (None, None), "numerator": None, "verdicts": None}
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
        "verdicts": row.get("verdicts"),  # good_rate only: counts per verdict
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
    ("good_rate", "Most good resumes"),
    ("claim_faithfulness", "Most faithful"),
    ("rubric_overall", "Best written (rubric v3)"),
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
    """Measured averages per workflow from the saved per-output rows, plus its good resumes.

    ``good``/``attempts`` count one output per pair under the final criteria; ``good_scored`` is
    False when ``good_resume_counts`` cannot decide the rate.
    """
    grouped: dict[str, list[dict]] = {}
    for row in bundle.rows:
        grouped.setdefault(row.get("workflow", ""), []).append(row)
    records, meta = rubric_scores(bundle)
    primary: dict[str, list[dict]] = {}
    for (workflow, _), row in _primary_rows(bundle.rows).items():
        primary.setdefault(workflow, []).append(row)
    profiles = {}
    for workflow, rows in grouped.items():
        if not workflow:
            continue
        good = good_resume_counts(primary.get(workflow, []), records, meta["gates"])
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
            "good": good["good"],
            "attempts": good["attempts"],
            "good_scored": good["good_scored"],
            "verdicts": good["verdicts"],
            "models": sorted({row.get("model") for row in rows if row.get("model")}),
        }
    return profiles


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


# Per-variant deltas offered in the robustness view: the truth measures of the final criteria.
# ``eval.reporting.robustness_delta_table`` also writes superseded metrics (task success,
# requirement coverage, keyword coverage, JD similarity, rubric v1); they are not shown.
DELTA_METRIC_LABELS = {
    "faithfulness": "Claim faithfulness",
    "unsupported_claim_rate": "Unsupported claims",
    "protected_field_violation": "Protected-field violations",
    "fabrication_bait_violation": "Fabrication bait",
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
    """Original inputs plus each workflow's output, its final-criteria scores and verdict.

    Per output: claim faithfulness and whether it passes the truth gate (``nli_pass``), the
    rubric v3 overall and dimension scores and whether the overall passes the writing gate
    (``rubric_pass``), and the ``verdict`` (``VERDICT_LABELS``). Unscored values stay None.
    """
    primary = _primary_rows(row for row in bundle.rows if row.get("pair_id") == pair_id)
    if not primary:
        return None
    records, meta = rubric_scores(bundle)
    gates = meta["gates"]
    first = next(iter(primary.values()))
    resume, jd = _source_texts(pair_id, first)
    outputs = {}
    for (workflow, _), row in primary.items():
        record = records.get((workflow, pair_id))
        failed = bool(row.get("error"))
        faithfulness = None if failed else faithfulness_value(row)
        overall = None if failed else _rubric_value(record)
        outputs[workflow] = {
            "resume_md": _output_text(row),
            "error": row.get("error"),
            "claim_faithfulness": faithfulness,
            "nli_pass": None if faithfulness is None else _passes(faithfulness, gates["nli"]),
            "unsupported_claim_rate": row.get("unsupported_claim_rate"),
            "protected_field_violation_count": row.get("protected_field_violation_count"),
            "fabrication_bait_match_count": row.get("fabrication_bait_match_count"),
            "rubric_overall": overall,
            "rubric_pass": None if overall is None else _passes(overall, gates["rubric"]),
            "rubric_scores": {
                name: None if failed else _rubric_value(record, name)
                for name in RUBRIC_DIMENSION_LABELS
            },
            "rubric_status": (record or {}).get("status"),
            "verdict": resume_verdict(row, record, gates),
            "total_tokens": row.get("total_tokens"),
            "wall_seconds": row.get("wall_seconds", row.get("seconds")),
        }
    return {
        "pair_id": pair_id,
        "category": first.get("category", ""),
        "fit_level": first.get("fit_level", ""),
        "role_family": first.get("role_family", ""),
        "seniority": first.get("seniority", ""),
        "original_resume": resume,
        "job_description": jd,
        "gates": dict(gates),
        "has_rubric": bool(records),
        "outputs": {name: outputs[name] for name in order_workflows(outputs)},
    }
