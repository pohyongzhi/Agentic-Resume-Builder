"""Chart payload for the "Results (charts)" tab, built from one saved evaluation root.

Read-only, like ``eval.dashboard``: it reads the per-output rows that ``evaluate.py`` writes to
``evaluations.jsonl`` (loaded with ``eval.dashboard.load_bundle``; when no index exists it falls
back to ``records/<workflow>/<pair>/repeat_*.json``) and returns plain JSON-ready data:

* ``rows``: one compact row per output (``COLUMNS``), each with a single ``outcome``;
* ``summary``: per-workflow rates and means with 1,000-resample bootstrap 95% intervals;
* ``headlines`` and ``text``: chart titles and page text computed from the same numbers, so they
  stay true after a rerun.

Values an evaluator did not score stay ``None`` (JSON ``null``); they are never counted as 0.
"""
from __future__ import annotations

import json
import math
import random
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import config
from eval import dashboard as dd

COLUMNS = ("wf", "cat", "fit", "outcome", "faith", "words", "wall", "calls", "bait", "edu",
           "unsupported")
OUTCOMES = ("pass", "unfaithful", "sections", "length", "rejected", "other")
UNFAITHFUL_REASONS = (
    "faithfulness_below_threshold", "fabrication_bait_violation", "protected_field_violation",
)
FAILURE_PHRASES = {
    "unfaithful": "unfaithful content",
    "length": "the wrong length",
    "sections": "a missing required section",
    "rejected": "a rejected draft",
    "other": "another failed check",
}
CATEGORY_LABELS = {
    "business_technology": "Business tech",
    "cloud_devops": "Cloud & DevOps",
    "cybersecurity": "Cybersecurity",
    "data_ai": "Data & AI",
    "software_engineering": "Software eng.",
}
CATEGORY_PREFIXES = {"bt": "business_technology", "cd": "cloud_devops", "cy": "cybersecurity",
                     "da": "data_ai", "se": "software_engineering"}
FIT_ORDER = ("strong", "moderate", "weak")
EDUCATION_HEADING = re.compile(r"^#+\s*education", re.I | re.M)
DEGREE = re.compile(r"(Bachelor|Master|Diploma|Degree|B\.Sc|BSc|Associate)", re.I)
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 42


# --- workflows ---------------------------------------------------------------------------------


def registry_order() -> list[str]:
    """Workflow names in registry order (``workflows.WORKFLOWS``)."""
    try:
        from workflows import WORKFLOWS

        return list(WORKFLOWS)
    except Exception:  # the dashboard still works if the workflow stack cannot import
        return list(dd.WORKFLOW_ORDER)


def order_workflows(names) -> list[str]:
    present = set(names)
    registry = registry_order()
    known = [name for name in registry if name in present]
    return known + sorted(present - set(registry))


def _title(workflow: str) -> str:
    info = dd.WORKFLOW_INFO.get(workflow)
    return info["title"] if info else workflow.replace("_", " ")


# --- per-output compaction ---------------------------------------------------------------------


def outcome(row: dict) -> str:
    """The single most serious outcome of one output."""
    if row.get("error"):
        return "rejected"
    if row.get("task_success"):
        return "pass"
    reasons = set(row.get("task_failure_reasons") or ())
    if reasons & set(UNFAITHFUL_REASONS):
        return "unfaithful"
    if "missing_required_sections" in reasons:
        return "sections"
    if "word_count_out_of_range" in reasons:
        return "length"
    return "other"


def _category(row: dict) -> str:
    category = row.get("category") or CATEGORY_PREFIXES.get(str(row.get("pair_id", ""))[:2])
    if not category:
        return "Unknown"
    return CATEGORY_LABELS.get(category, category.replace("_", " ").capitalize())


def _number(value, digits=None):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    if digits is None:
        return int(value) if value.is_integer() else value
    return round(value, digits)


def _flag(value):
    return None if value is None else int(bool(value))


def _words(row: dict, text: str):
    words = _number(row.get("words"))
    if words is None and text:
        words = len(text.split())
    return words


def _unsupported(row: dict):
    claims, supported = _number(row.get("n_claims")), _number(row.get("supported_claims"))
    if claims is not None and supported is not None:
        return int(claims - supported)
    listed = row.get("unsupported")
    return len(listed) if isinstance(listed, list) and row.get("claim_faithfulness") is not None \
        else None


def has_education(text: str) -> bool:
    return bool(text and EDUCATION_HEADING.search(text) and DEGREE.search(text))


def compact_row(row: dict, code: str) -> list:
    kind = outcome(row)
    base = [code, _category(row), row.get("fit_level"), kind]
    if kind == "rejected":
        return base + [None] * (len(COLUMNS) - len(base))
    text = row.get("resume_md") or ""
    wall = row.get("wall_seconds")
    if wall is None:
        wall = row.get("seconds")
    return base + [
        _number(row.get("claim_faithfulness"), 4),
        _words(row, text),
        _number(wall, 1),
        _number(row.get("llm_calls")),
        _flag(row.get("fabrication_bait_violation")),
        int(has_education(text)),
        _unsupported(row),
    ]


# --- statistics --------------------------------------------------------------------------------


def boot_ci(values, n=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    """``[mean, low, high]`` with a percentile bootstrap 95% interval; ``[None]*3`` if empty."""
    values = [value for value in values if value is not None]
    if not values:
        return [None, None, None]
    rng = random.Random(seed)
    k = len(values)
    samples = sorted(
        statistics.fmean(values[rng.randrange(k)] for _ in range(k)) for _ in range(n)
    )
    return [round(statistics.fmean(values), 4), round(samples[int(0.025 * n)], 4),
            round(samples[int(0.975 * n) - 1], 4)]


def _median(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    middle = statistics.median(values)
    return int(middle) if float(middle).is_integer() else round(middle, 1)


def _summary(code, name, rows, raw):
    done = [row for row in rows if row["outcome"] != "rejected"]
    scored_task = [row for row in raw if not row.get("error") and row.get("task_success") is not None]
    bait = [row["bait"] for row in done if row["bait"] is not None]
    return {
        "short": code, "name": name, "total": len(rows), "done": len(done),
        "rejected": len(rows) - len(done),
        "faith": boot_ci([row["faith"] for row in done]),
        "faith_n": sum(row["faith"] is not None for row in done),
        "bait": boot_ci([float(value) for value in bait]), "bait_n": len(bait),
        "edu": boot_ci([float(row["edu"]) for row in done]),
        "length_ok": boot_ci([
            float(config.TASK_MIN_WORDS <= row["words"] <= config.TASK_MAX_WORDS)
            for row in done if row["words"] is not None
        ]),
        "task": boot_ci([float(bool(row["task_success"])) for row in scored_task]),
        "wall_median": _median([row["wall"] for row in done]),
        "calls_median": _median([row["calls"] for row in done]),
    }


# --- loading -----------------------------------------------------------------------------------


def _record_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "records").glob("*/*/repeat_*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(row, dict):
            row.setdefault("workflow", path.parent.parent.name)
            row.setdefault("pair_id", path.parent.name)
            rows.append(row)
    return rows


def load_rows(root) -> list[dict]:
    """One row per (workflow, pair), the lowest repeat, from the index or the saved records."""
    root = Path(root)
    rows = dd.load_bundle(root).rows if (root / "evaluations.jsonl").is_file() else []
    if not rows:
        rows = _record_rows(root)
    rows = [row for row in rows if row.get("workflow") and row.get("pair_id")]
    return list(dd._primary_rows(rows).values())


def generation_date(root: Path, rows) -> str | None:
    """Date the newest saved generation was written, if its run files are still on disk."""
    stamps = [row.get(key) for row in rows for key in ("generated_at", "created_at")]
    stamps = [str(stamp)[:10] for stamp in stamps if stamp]
    if stamps:
        return max(stamps)
    try:
        base = Path(root).resolve().parents[4]  # <base>/results/evaluations/<d>/<g>/<e>
    except IndexError:
        return None
    newest = 0.0
    for row in rows:
        relative = row.get("metrics_path") or row.get("resume_path")
        if not relative:
            continue
        path = Path(relative)
        path = path if path.is_absolute() else base / path
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    return datetime.fromtimestamp(newest).strftime("%Y-%m-%d") if newest else None


def _identity_nli_model(root: Path) -> str | None:
    identity = dd.load_bundle(root).identity if (root / "evaluations.jsonl").is_file() else {}
    if not identity:
        try:
            identity = json.loads((root / "evaluator_identity.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            identity = {}
    nli = identity.get("nli") if isinstance(identity, dict) else None
    return nli.get("model") if isinstance(nli, dict) else None


# --- text --------------------------------------------------------------------------------------


def join_codes(codes) -> str:
    codes = list(codes)
    if len(codes) <= 1:
        return "".join(codes)
    return ", ".join(codes[:-1]) + " and " + codes[-1]


def _f2(value) -> str:
    return f"{value:.2f}"


def _floor2(value) -> float:
    return math.floor(value * 100 + 1e-9) / 100


def _ceil2(value) -> float:
    return math.ceil(value * 100 - 1e-9) / 100


def _ties(values: dict, target) -> list[str]:
    return [code for code, value in values.items() if value == target]


def headlines(codes, rows, summary) -> dict[str, str]:
    """A title per chart, computed from the data; neutral when the data cannot support one."""
    by_code = {item["short"]: item for item in summary}
    done = [row for row in rows if row["outcome"] != "rejected"]
    out = {}

    # Outcome.
    passes = {code: sum(r["outcome"] == "pass" for r in rows if r["wf"] == code) for code in codes}
    shares = {code: passes[code] / by_code[code]["total"] for code in codes if by_code[code]["total"]}
    failures = Counter(row["outcome"] for row in rows if row["outcome"] != "pass")
    failure = (
        f"; the most common failure is {FAILURE_PHRASES[failures.most_common(1)[0][0]]}"
        if failures else ""
    )
    if not shares:
        out["outcome"] = "Outcome of each workflow's resumes"
    elif max(shares.values()) == 0:
        out["outcome"] = "No workflow's resumes pass every check" + failure
    else:
        best = _ties(shares, max(shares.values()))
        verb = "passes" if len(best) == 1 else "pass"
        counts = (
            f"{passes[best[0]]} of {by_code[best[0]]['total']} resumes" if len(best) == 1
            else f"{max(shares.values()):.0%} of their resumes"
        )
        out["outcome"] = f"{join_codes(best)} {verb} most often ({counts}){failure}"

    # Faithfulness distribution.
    means = {code: by_code[code]["faith"][0] for code in codes if by_code[code]["faith"][0] is not None}
    low = _ties(means, min(means.values())) if means else []
    rest = [means[code] for code in means if code not in low]
    if len(means) < 2 or not rest:
        out["faith"] = "Claim faithfulness of each workflow's resumes"
    else:
        verb = "is" if len(low) == 1 else "are"
        out["faith"] = (
            f"{join_codes(low)} {verb} the least faithful (mean {_f2(means[low[0]])}); "
            f"every other{' scored' if len(means) < len(codes) else ''} workflow averages "
            f"{_f2(_floor2(min(rest)))} or higher"
        )

    # Invented qualifications.
    rates = {code: by_code[code]["bait"][0] for code in codes if by_code[code]["bait"][0] is not None}
    if not rates:
        out["bait"] = "Invented qualifications by workflow"
    elif max(rates.values()) == 0:
        out["bait"] = "No workflow invents a qualification in any resume"
    else:
        top = _ties(rates, max(rates.values()))
        verb = "invents" if len(top) == 1 else "invent"
        whose = "its" if len(top) == 1 else "their"
        out["bait"] = f"{join_codes(top)} {verb} a qualification in {max(rates.values()):.0%} of {whose} resumes"

    # Category heatmap.
    spreads = []
    for code in codes:
        cats = {}
        for row in done:
            if row["wf"] == code and row["faith"] is not None:
                cats.setdefault(row["cat"], []).append(row["faith"])
        cat_means = [statistics.fmean(values) for values in cats.values()]
        if len(cat_means) >= 2:
            spreads.append(max(cat_means) - min(cat_means))
    out["heat"] = (
        f"Job category shifts mean faithfulness by at most {_f2(_ceil2(max(spreads)))} for any workflow"
        if spreads else "Mean claim faithfulness by job category"
    )

    out["fit"] = "Claim faithfulness by how well the candidate fits the job"

    # Time.
    walls = {code: by_code[code]["wall_median"] for code in codes
             if by_code[code]["wall_median"] is not None and by_code[code]["wall_median"] > 0}
    if len(walls) < 2:
        out["time"] = "Median time per resume"
    else:
        slow, fast = _ties(walls, max(walls.values())), _ties(walls, min(walls.values()))
        ratio = max(walls.values()) / min(walls.values())
        if ratio < 1.05:
            out["time"] = "Every workflow takes about the same time per resume"
        else:
            times = f"{ratio:.0f}" if ratio >= 10 else f"{ratio:.1f}"
            verb = "takes" if len(slow) == 1 else "take"
            out["time"] = (
                f"{join_codes(slow)} {verb} about {times} times as long as "
                f"{' or '.join(fast) if len(fast) <= 2 else join_codes(fast)} per resume"
            )

    # Model calls.
    calls = {code: by_code[code]["calls_median"] for code in codes
             if by_code[code]["calls_median"] is not None}
    if not calls:
        out["calls"] = "Median model calls per resume"
    elif len(set(calls.values())) == 1:
        out["calls"] = f"Every workflow makes a median of {calls[next(iter(calls))]:g} model calls per resume"
    else:
        most = _ties(calls, max(calls.values()))
        verb = "makes" if len(most) == 1 else "make"
        out["calls"] = (
            f"{join_codes(most)} {verb} the most model calls: a median of "
            f"{max(calls.values()):g} per resume"
        )

    # Faithfulness against time (the dashed frontier in the scatter).
    points = [(by_code[code]["wall_median"], by_code[code]["faith"][0], code) for code in codes
              if by_code[code]["wall_median"] and by_code[code]["faith"][0] is not None]
    if len(points) < 2:
        out["scatter"] = "Claim faithfulness against time per resume"
    else:
        front, best = [], -1.0
        for wall, faith, code in sorted(points, key=lambda p: (p[0], -p[1])):
            if faith > best:
                front.append(code)
                best = faith
        if len(front) == 1:
            out["scatter"] = f"{front[0]} is both the fastest and the most faithful workflow"
        else:
            out["scatter"] = (
                f"Nothing beats {' or '.join(front) if len(front) == 2 else join_codes(front)} "
                "on both speed and faithfulness"
            )

    # Length.
    under = {}
    for code in codes:
        words = [row["words"] for row in done if row["wf"] == code and row["words"] is not None]
        if words:
            under[code] = sum(w < config.TASK_MIN_WORDS for w in words) / len(words)
    short = [code for code, share in under.items() if share > 0.5]
    if not under:
        out["words"] = "Word count of each workflow's resumes"
    elif short:
        verb = "mostly writes" if len(short) == 1 else "mostly write"
        out["words"] = (
            f"{join_codes(short)} {verb} resumes under the {config.TASK_MIN_WORDS}-word minimum"
        )
    else:
        out["words"] = (
            f"Most resumes from every workflow reach the {config.TASK_MIN_WORDS}-word minimum"
        )
    return out


def _models_text(raw, short_of) -> str:
    models = {}
    for row in raw:
        if row.get("model"):
            models.setdefault(row["workflow"], Counter())[row["model"]] += 1
    if not models:
        return ""
    per_workflow = {wf: counts.most_common(1)[0][0] for wf, counts in models.items()}
    common = Counter(per_workflow.values()).most_common(1)[0][0]
    others = [f"{short_of[wf]}: {model}" for wf, model in per_workflow.items() if model != common]
    return common + (f" ({'; '.join(others)})" if others else "")


# --- payload -----------------------------------------------------------------------------------


_CACHE: dict[str, tuple[tuple, dict[str, Any] | None]] = {}


def build_payload(root) -> dict[str, Any] | None:
    """The chart page's data for one evaluation root, or ``None`` when it has no outputs.

    Cached per root until its index or summaries change (the bootstrap takes about a second).
    """
    root = Path(root)
    if not root.is_dir():
        return None
    key, signature = str(root.resolve()), dd._signature(root)
    cached = _CACHE.get(key)
    if cached and cached[0] == signature and signature:
        return cached[1]
    payload = _build_payload(root)
    if len(_CACHE) >= 4:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = (signature, payload)
    return payload


def clear_cache():
    _CACHE.clear()


def _build_payload(root: Path) -> dict[str, Any] | None:
    raw = load_rows(root)
    if not raw:
        return None
    workflows = order_workflows({row["workflow"] for row in raw})
    short_of = {wf: dd.short_name(wf) for wf in workflows}
    codes = [short_of[wf] for wf in workflows]
    raw.sort(key=lambda row: (workflows.index(row["workflow"]), str(row["pair_id"])))
    rows = [compact_row(row, short_of[row["workflow"]]) for row in raw]
    records = [dict(zip(COLUMNS, row)) for row in rows]
    summary = [
        _summary(short_of[wf], _title(wf), [r for r in records if r["wf"] == short_of[wf]],
                 [r for r in raw if r["workflow"] == wf])
        for wf in workflows
    ]
    pairs = len({row["pair_id"] for row in raw})
    originals = {row["pair_id"]: len(row["original_resume"].split())
                 for row in raw if row.get("original_resume")}
    dataset = next((row.get("dataset_hash") for row in raw if row.get("dataset_hash")), None)
    dataset = dataset or root.parent.parent.name
    generated = generation_date(root, raw)
    models = _models_text(raw, short_of)
    nli = _identity_nli_model(root)

    caveat = (
        f"Evaluation run: dataset {dataset[:12]} · evaluator {root.name} · {len(raw)} outputs "
        f"({pairs} pairs × {len(workflows)} workflows)"
        + (f" · generated {generated}" if generated else "") + "."
    )
    footer = (
        "Claim faithfulness: share of a resume's claims entailed by the original resume "
        f"(cross-encoder NLI{', ' + nli if nli else ''}). Invented qualification: a job-description "
        "skill or certification absent from the original appears in the output. Intervals: "
        f"{BOOTSTRAP_SAMPLES:,} bootstrap resamples over pairs."
        + (f" Generated {generated}" if generated else "")
        + (f"{' with' if generated else ' Models:'} {models}" if models else "")
        + ("." if generated or models else "")
        + " Requirement coverage and pairwise comparison against "
        f"{dd.short_name(dd.BASELINE)} are not included yet."
    )
    return {
        "workflows": [{"short": short_of[wf], "name": _title(wf), "key": wf} for wf in workflows],
        "baseline": dd.short_name(dd.BASELINE),
        "summary": summary,
        "cols": list(COLUMNS),
        "rows": rows,
        "fits": [fit for fit in FIT_ORDER if any(r["fit"] == fit for r in records)]
        + sorted({r["fit"] for r in records if r["fit"] and r["fit"] not in FIT_ORDER}),
        "thresholds": {"faith": config.TASK_FAITHFULNESS_THRESHOLD,
                       "min_words": config.TASK_MIN_WORDS, "max_words": config.TASK_MAX_WORDS},
        "original_words": round(statistics.fmean(originals.values())) if originals else None,
        "run": {"dataset_hash": dataset, "generation_scope": root.parent.name,
                "evaluator_hash": root.name, "outputs": len(raw), "pairs": pairs,
                "generated": generated, "models": models, "nli_model": nli},
        "headlines": headlines(codes, records, summary),
        "text": {
            "kicker": (f"IS469 · Resume Agent Lab · {pairs} resume and job-description pairs × "
                       f"{len(workflows)} workflows"),
            "intro": (f"Every workflow rewrote the same {pairs} resumes for their job descriptions "
                      "using a small local model. These charts use only automatic checks: claim "
                      "faithfulness against the original resume, invented qualifications, "
                      "required sections, length, time and model calls."),
            "caveat": caveat,
            "footer": footer,
        },
    }
