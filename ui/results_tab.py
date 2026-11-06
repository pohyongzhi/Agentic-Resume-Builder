"""The read-only "Results" tab: explains W1-W7 and shows saved evaluation results.

Results follow the final criteria: a resume is good when its NLI claim faithfulness reaches the
truth gate and its rubric v3 overall score reaches the writing gate, and a workflow's result is
good resumes out of every attempt. Superseded measures (rubric v1, requirement coverage, keyword
coverage, JD similarity, pairwise against W2, task success) are not shown.

Layout only; data comes from ``eval.dashboard`` and charts from ``ui.charts``. No handler here
generates, evaluates or calls a model. ``render_*`` functions are plain and testable; the
``build_results_tab`` function wires them into Gradio components inside an existing Blocks.
"""
from __future__ import annotations

from html import escape
from pathlib import Path

import gradio as gr

import config
from eval import dashboard as dd
from eval import dashboard_charts as dc
from ui import charts

ALL = ""
COST_CHOICES = [("Tokens per output", "total_tokens"), ("Wall time per output", "historical_run_wall_seconds")]
QUALITY_CHOICES = [("Good-resume rate", "good_rate"), ("Claim faithfulness", "claim_faithfulness"),
                   ("Rubric v3 overall", "rubric_overall")]
DEFAULT_QUALITY = "good_rate"
SLOT_WORKFLOWS = dd.WORKFLOW_ORDER
# Chart components: no default prose styling, plus the hover/focus tooltip layer.
CHART_HTML = {"apply_default_css": False, "js_on_load": charts.TOOLTIP_JS}


def _models():
    try:
        from workflows import WORKFLOW_MODELS

        return dict(WORKFLOW_MODELS)
    except Exception:  # the explainer still renders if the workflow stack cannot import
        return {
            name: config.FINE_TUNED_MODEL if name == "w5_fine_tuned" else config.BASE_MODEL
            for name in dd.WORKFLOW_ORDER
        }


def _bundle(path):
    if not path:
        return None
    path = Path(path)
    if not path.is_dir():
        return None
    return dd.load_bundle(path)


NO_RUN = "No saved evaluation found. Run evaluate.py, then press Refresh."


# --- status ------------------------------------------------------------------------------------


def render_status(path) -> str:
    bundle = _bundle(path)
    if bundle is None:
        return charts.empty(NO_RUN)
    root = Path(path)
    stages = dd.root_stages(root)

    def mark(name):
        return f"<b>{name}</b> ✓" if name in stages else f"{name} — not run"

    dataset = root.parent.parent.name
    workflows = ", ".join(dd.short_name(name) for name in bundle.workflows) or "none"
    summaries = "summaries ✓" if bundle.summaries else "no summaries (rebuilt in memory)"
    records, meta = dd.rubric_scores(bundle)
    rubric = (
        f"<b>rubric v3</b> ✓ <code>{escape(Path(meta['path']).name)}</code>"
        if records else "rubric v3 — not found"
    )
    return charts.wrap(
        '<div class="status-line">'
        f"Evaluator <code>{escape(root.name)}</code> · generation scope "
        f"<code>{escape(root.parent.name)}</code> · dataset <code>{escape(dataset[:12])}</code>"
        f"<br><b>{len(bundle.rows)}</b> scored outputs · workflows: <b>{escape(workflows)}</b>"
        f" · {mark('NLI')} · {rubric} · {summaries}"
        "</div>"
    )


def _criteria(gates) -> str:
    """The final criteria in one sentence, with this run's gates."""
    return (
        f"A resume is <b>good</b> when its NLI claim faithfulness against the original resume is "
        f"at least {gates['nli']:.2f} (truth gate) and its rubric v3 overall score is at least "
        f"{gates['rubric']:g} out of 5 (writing gate). A workflow's result is its good resumes out "
        "of every attempt; a failed generation is not good."
    )


def _verdict_text(verdicts) -> str:
    """``"94 good · 1 failed truth · 5 failed writing"`` (zero counts left out)."""
    return " · ".join(
        f"{verdicts[kind]} {label}" for kind, label in dd.VERDICT_LABELS.items()
        if verdicts.get(kind)
    )


# --- 1. workflow explainer ---------------------------------------------------------------------


def _stat(value, label):
    return f"<div><b>{value}</b><span>{label}</span></div>"


def render_explainer(path, models=None) -> str:
    models = models or _models()
    bundle = _bundle(path)
    profiles = dd.workflow_profiles(bundle) if bundle else {}
    cards = []
    for workflow in dd.WORKFLOW_ORDER:
        info = dd.WORKFLOW_INFO[workflow]
        flow = '<span class="ar">→</span>'.join(
            f'<span class="st">{escape(step)}</span>' for step in info["steps"]
        )
        model = models.get(workflow, config.BASE_MODEL)
        profile = profiles.get(workflow)
        if profile:
            measured = [m for m in profile["models"] if m != model]
            model_note = (
                f" · measured run used <code>{escape(', '.join(measured))}</code>" if measured else ""
            )
            good = (
                f"{profile['good']}/{profile['attempts']}" if profile["good_scored"] else "unscored"
            )
            stats = (
                '<div class="stats">'
                + _stat(charts.fmt_value(profile["llm_calls"], "number"), "LLM calls")
                + _stat(charts.fmt_value(profile["tokens"], "number"), "tokens")
                + _stat(
                    "–" if profile["wall_seconds"] is None
                    else f"{charts.fmt_value(profile['wall_seconds'], 'number')}s",
                    "wall time",
                )
                + _stat(good, "good resumes")
                + "</div>"
                + f'<div class="model">Averages over {profile["outputs"]} saved outputs'
                + (f", {profile['generation_failures']} failed" if profile["generation_failures"] else "")
                + "</div>"
            )
        else:
            model_note = ""
            stats = '<div class="stats"><div><span>Not in this evaluation</span></div></div>'
        base = " base" if workflow == dd.BASELINE else ""
        tag = ' <span class="tag">baseline</span>' if workflow == dd.BASELINE else ""
        cards.append(
            f'<div class="card{base}" style="--card-colour:{charts.colour_var(workflow)}">'
            f'<h3><span class="code" style="color:{charts.label_ink(workflow)}">{info["code"]}</span>'
            f'{escape(info["title"])}{tag}</h3>'
            f'<div class="model">{escape(info["role"])} · model <code>{escape(model)}</code>'
            f"{model_note}</div>"
            f"<p>{escape(info['summary'])}</p>"
            f'<div class="flow">{flow}</div>{stats}</div>'
        )
    intro = (
        '<p class="note" style="margin:0 0 .7rem">Seven ways to tailor the same resume to the same '
        "job, all with a small local model. W2 is the shared baseline every other workflow is "
        "compared with; ‖ marks steps that run in parallel and ↺ a bounded retry loop.</p>"
    )
    return charts.wrap(intro + f'<div class="cards">{"".join(cards)}</div>')


# --- 2. leaderboard and trade-offs -------------------------------------------------------------


def _cell_html(spec, cell, best, workflow=""):
    if cell["status"] == "missing":
        return '<td><span class="na">–</span></td>'
    if cell["value"] is None:
        label = "unscored" if cell["status"] == "unscored" else "n/a"
        return (
            f'<td><span class="na">{label}</span>'
            f'<span class="n">0/{cell["eligible"]} scored</span></td>'
        )
    low, high = cell["ci"]
    ci = (
        f"95% CI {charts.fmt_value(low, spec.kind)} – {charts.fmt_value(high, spec.kind)}"
        if low is not None and high is not None else ""
    )
    cls = ' class="best"' if best else ""
    if spec.kind == "rate" and cell["numerator"] is not None:
        count = f"{cell['numerator']}/{cell['scored']}"
    else:
        count = f"n={cell['scored']}"
    if cell["scored"] < cell["eligible"]:
        count += f" of {cell['eligible']}"
    text = f"{charts.fmt_value(cell['value'], spec.kind)} · {spec.label} | {dd.display_name(workflow)}"
    if cell.get("verdicts"):
        text += f" | {cell['numerator']} good of {cell['scored']} attempts"
        text += f" | {_verdict_text(cell['verdicts'])}"
    else:
        text += f" | {cell['scored']} scored of {cell['eligible']} eligible"
    if ci:
        text += f" | {ci}"
    return (
        f'<td{charts.tip(text)}><span{cls}>{charts.fmt_value(cell["value"], spec.kind)}</span>'
        f'<span class="n">{count}</span></td>'
    )


def _kpi_tiles(board) -> str:
    tiles = []
    for finding in dd.headline_findings(board):
        spec = dd.METRIC_SPECS[finding["metric"]]
        if finding["workflow"] is None:
            tiles.append(
                f'<div class="kpi"><div class="k-label">{escape(finding["title"])}</div>'
                '<div class="k-value" style="color:var(--viz-muted)">–</div>'
                '<div class="k-delta">not scored</div></div>'
            )
            continue
        value = charts.fmt_value(finding["value"], spec.kind)
        if finding["workflow"] == dd.BASELINE:
            delta = "the baseline itself"
        elif finding["baseline"] is None:
            delta = "W2 not scored on this metric"
        else:
            diff = finding["value"] - finding["baseline"]
            if spec.kind == "rate":
                delta = f"{diff * 100:+.0f} pts vs W2 ({charts.fmt_value(finding['baseline'], 'rate')})"
            elif spec.kind == "number":
                delta = f"{diff / finding['baseline'] * 100:+.0f}% vs W2" if finding["baseline"] else ""
            else:
                delta = f"{diff:+.2f} vs W2 ({charts.fmt_value(finding['baseline'], spec.kind)})"
        if finding["ties"]:
            delta = "tied with " + ", ".join(dd.short_name(w) for w in finding["ties"])
        sw = "sw base" if finding["workflow"] == dd.BASELINE else "sw"
        tiles.append(
            f'<div class="kpi"><div class="k-label">{escape(finding["title"])}</div>'
            f'<div class="k-value">{value}</div>'
            f'<div class="k-who"><i class="{sw}" style="background:'
            f'{charts.colour_var(finding["workflow"])}"></i>'
            f'{escape(dd.display_name(finding["workflow"]))}</div>'
            f'<div class="k-delta">{escape(delta)} · n={finding["scored"]}</div></div>'
        )
    return f'<div class="grid4">{"".join(tiles)}</div>'


RUBRIC_DIMENSION_KEYS = {f"rubric_{name}" for name in dd.RUBRIC_DIMENSION_LABELS}


def _visible_metrics(board):
    """Every leaderboard metric; the four rubric dimensions only when some output was judged."""
    return [
        spec for spec in dd.LEADERBOARD_METRICS
        if spec.key not in RUBRIC_DIMENSION_KEYS or dd.metric_is_present(board, spec.key)
    ]


def _best(board, spec):
    values = [row["cells"][spec.key]["value"] for row in board]
    values = [value for value in values if value is not None]
    if len(values) < 2 or spec.tier == "cost" or len(set(values)) == 1:
        return None
    return max(values) if spec.better == "higher" else min(values)


def _header_note(spec, gates) -> str:
    better = "↑ better" if spec.better == "higher" else "↓ better"
    if spec.key == "nli_pass_rate":
        return f"≥ {gates['nli']:.2f} · {better}"
    if spec.key == "rubric_pass_rate":
        return f"≥ {gates['rubric']:g} · {better}"
    if spec.kind == "scale":
        return f"1–5 · {better}"
    return better


def _bars(board, metric):
    spec = dd.METRIC_SPECS[metric]
    items = []
    for row in board:
        cell = row["cells"][metric]
        if cell["value"] is None:
            label = f"unscored (0/{cell['eligible']})"
        elif spec.kind == "rate" and cell["numerator"] is not None:
            label = f"{charts.fmt_value(cell['value'], 'rate')} · {cell['numerator']}/{cell['scored']}"
        else:
            label = f"{charts.fmt_value(cell['value'], spec.kind)} · n={cell['scored']}"
        item = {"workflow": row["workflow"], "value": cell["value"], "label": label}
        if cell.get("verdicts") and cell["value"] is not None:
            item["tip"] = (
                f"{label} | {dd.display_name(row['workflow'])} | {_verdict_text(cell['verdicts'])}"
            )
        elif cell["value"] is not None and cell["scored"] < cell["eligible"]:
            item["tip"] = (
                f"{label} | {dd.display_name(row['workflow'])} | {cell['scored']} scored of "
                f"{cell['eligible']} eligible"
            )
        items.append(item)
    return charts.bar_chart(items, kind=spec.kind)


def render_leaderboard(path, category=ALL, fit_level=ALL) -> str:
    bundle = _bundle(path)
    if bundle is None:
        return charts.empty(NO_RUN)
    board, source = dd.leaderboard(bundle, category or None, fit_level or None)
    if not board:
        return charts.empty("No scored outputs match this filter.")
    gates = dd.final_gates(bundle)
    specs = _visible_metrics(board)
    headline = (
        f'<p class="note" style="margin:0 0 .7rem">{_criteria(gates)}</p>'
        + _kpi_tiles(board)
        + '<div class="gap"></div><div class="grid3">'
        + charts.panel(
            "Good resumes", _bars(board, "good_rate"),
            "Pass both gates, out of every attempt. Hover a bar for why the rest are not good.",
        )
        + charts.panel(
            "Truth gate", _bars(board, "nli_pass_rate"),
            f"Claim faithfulness ≥ {gates['nli']:.2f}: share of the output's claims the NLI "
            "referee finds supported by the original resume. Over resumes NLI scored.",
        )
        + charts.panel(
            "Writing gate", _bars(board, "rubric_pass_rate"),
            f"Rubric v3 overall ≥ {gates['rubric']:g}: mean of quantified evidence, clarity, "
            "concision and logical order (1–5). Over resumes the judge scored.",
        )
        + "</div>"
    )
    groups: list[list] = []
    for spec in specs:
        if groups and groups[-1][0] == spec.tier:
            groups[-1][1] += 1
        else:
            groups.append([spec.tier, 1])
    group_row = '<tr><th rowspan="2">Workflow</th>' + "".join(
        f'<th colspan="{span}" style="text-align:center;color:var(--viz-ink-2);font-weight:600">'
        f"{escape(dd.METRIC_TIERS.get(tier, tier))}</th>"
        for tier, span in groups
    ) + "</tr>"
    header = "".join(
        f"<th>{escape(spec.label)}<br><span class=\"n\">{escape(_header_note(spec, gates))}"
        "</span></th>"
        for spec in specs
    )
    best = {spec.key: _best(board, spec) for spec in specs}
    body = []
    for row in board:
        cells = "".join(
            _cell_html(
                spec, row["cells"][spec.key],
                best[spec.key] is not None and row["cells"][spec.key]["value"] == best[spec.key],
                row["workflow"],
            )
            for spec in specs
        )
        cls = ' class="base"' if row["workflow"] == dd.BASELINE else ""
        body.append(f"<tr{cls}><td>{charts._wf_label(row['workflow'])}</td>{cells}</tr>")
    filters = [value for value in (category, fit_level) if value]
    scope = " · ".join(value.replace("_", " ") for value in filters) or "all pairs"
    note = (
        f"Scope: {escape(scope)}. Good resumes count every attempt, so a failed generation lowers "
        "the rate. Every other cell shows the mean (or rate) over the outputs the metric could "
        "score and how many that was; outputs a metric could not score are counted as unscored, "
        "never as 0. Bold = best among scored workflows. Hover a value for its 95% CI. Source: "
        f"<code>{escape(source)}</code>."
    )
    table = charts.panel(
        "Leaderboard", f'<div class="tblwrap"><table class="tbl"><thead>{group_row}<tr>'
        f"{header}</tr></thead><tbody>{''.join(body)}</tbody></table></div>",
        note=note,
    )
    return charts.wrap(headline + '<div class="gap"></div>' + table)


def quality_metric(board, requested=DEFAULT_QUALITY) -> str:
    """The requested quality axis, or claim faithfulness when nothing is scored on it."""
    requested = requested if requested in dd.METRIC_SPECS else DEFAULT_QUALITY
    return requested if dd.metric_is_present(board, requested) else "claim_faithfulness"


def _clamp(spec):
    return (1, 5) if spec.kind == "scale" else (0, 1)


def render_tradeoffs(path, category=ALL, fit_level=ALL, cost="total_tokens",
                     quality=DEFAULT_QUALITY) -> tuple[str, str]:
    bundle = _bundle(path)
    if bundle is None:
        return charts.empty(NO_RUN), ""
    board, _ = dd.leaderboard(bundle, category or None, fit_level or None)
    workflows = [row["workflow"] for row in board]
    cost_spec = dd.METRIC_SPECS[cost]
    quality_spec = dd.METRIC_SPECS[quality_metric(board, quality)]
    cost_points = dd.tradeoff_points(board, cost_spec.key, quality_spec.key)
    missing = [dd.short_name(w) for w in workflows if w not in {p["workflow"] for p in cost_points}]
    first = charts.panel(
        f"Quality vs cost: {quality_spec.label.lower()} vs {cost_spec.label.lower()}",
        charts.scatter_chart(
            cost_points, x_label=cost_spec.label, y_label=quality_spec.label,
            x_kind="number", y_kind=quality_spec.kind, x_better="lower", x_clamp=(0, None),
            y_clamp=_clamp(quality_spec),
        ),
        f"Top-left is best: a higher {quality_spec.label.lower()} for less compute. Dashed lines "
        "mark W2.",
        note=("Not scored on both: " + ", ".join(missing)) if missing else "",
    )
    truth, writing = dd.METRIC_SPECS["claim_faithfulness"], dd.METRIC_SPECS["rubric_overall"]
    gate_points = dd.tradeoff_points(board, truth.key, writing.key)
    missing = [
        dd.short_name(w) for w in workflows if w not in {p["workflow"] for p in gate_points}
    ]
    second = charts.panel(
        "Truth vs writing: claim faithfulness and rubric v3",
        charts.scatter_chart(
            gate_points, x_label=truth.label, y_label=writing.label, x_kind=truth.kind,
            y_kind=writing.kind, x_better="higher", x_clamp=_clamp(truth),
            y_clamp=_clamp(writing),
        ),
        "Top-right is best: faithful to the original resume and well written. Means over each "
        "workflow's resumes; a good resume must pass both gates on its own.",
        note=("Not scored on both: " + ", ".join(missing)) if missing else "",
    )
    return charts.wrap(charts.legend(workflows) + first), charts.wrap(
        charts.legend(workflows) + second
    )


# --- 3. top resumes ---------------------------------------------------------------------------


def render_top(path):
    """(intro html, {workflow: (visible, header html, resume md, source md)}) for the top resumes."""
    bundle = _bundle(path)
    slots = {workflow: (False, "", "", "") for workflow in SLOT_WORKFLOWS}
    if bundle is None:
        return charts.empty(NO_RUN), slots
    top = dd.top_resumes(bundle)
    if not top["has_rubric"]:
        return charts.empty("Top resumes need rubric v3 scores; no rubric v3 run was found."), slots
    for item in top["items"]:
        workflow = item["workflow"]
        if workflow not in slots:
            continue
        info = dd.WORKFLOW_INFO.get(workflow, {"code": workflow, "title": ""})
        if item["best"] is None:
            header = charts.wrap(
                f'<div class="out-head" style="--card-colour:{charts.colour_var(workflow)}">'
                f'<h4><span style="color:{charts.colour_var(workflow)}">{escape(info["code"])}</span> '
                f"{escape(info['title'])}</h4>"
                f'<div class="model">{escape(item["reason"] or "no good resume")} · '
                f"{item['good']} of {item['total']} good</div></div>"
            )
            slots[workflow] = (True, header, "", "")
            continue
        best = item["best"]
        facts = [best["job_title"], best["category"].replace("_", " "),
                 f"{best['fit_level']} fit" if best["fit_level"] else "", best["pair_id"]]
        tie = (f" · {item['ties']} other good resume{'s' if item['ties'] != 1 else ''} "
               "scored the same" if item["ties"] else "")
        header = output_header(workflow, best["output"], top["gates"], True).replace(
            '<div class="chips">',
            f'<div class="model">Top of {item["good"]} good resumes out of {item["total"]}{tie}'
            f'<br>{escape(" · ".join(fact for fact in facts if fact))}</div><div class="chips">', 1)
        source = ("#### Job description\n\n" + (best["job_description"] or "_Not available._")
                  + "\n\n---\n\n#### Original resume\n\n"
                  + (best["original_resume"] or "_Not available._"))
        slots[workflow] = (True, header, best["output"]["resume_md"] or "_Empty output._", source)
    intro = charts.wrap(
        '<div class="status-line">Each workflow\'s best resume: among its good resumes '
        "(both gates passed), the one with the highest rubric v3 overall score; ties go to higher "
        "claim faithfulness, then fewer unsupported claims."
        f"<br>{_criteria(top['gates'])}</div>"
    )
    return intro, slots


# --- 4. output viewer --------------------------------------------------------------------------

# Verdict chip per resume: text (with a mark, so state never rests on colour) and chip state.
VERDICT_CHIPS = {
    "good": ("✓ good resume", "good"),
    "nli_fail": ("✗ failed truth", "bad"),
    "rubric_fail": ("✗ failed writing", "bad"),
    "unscored": ("unscored", ""),
    "rejected": ("✗ no resume", "bad"),
}


def _gate_chip(name, value, passed, gate, digits):
    """``"NLI 0.93 ✓ ≥ 0.80"``-style text and state; ``None`` value means unscored."""
    if value is None:
        return f"{name} unscored", ""
    if passed:
        return f"{name} {value:.{digits}f} ✓ ≥ {gate:.{digits}f}", "good"
    return f"{name} {value:.{digits}f} ✗ < {gate:.{digits}f}", "bad"


def _score_chips(output, gates=None, has_rubric=True):
    gates = gates or dc.DEFAULT_GATES
    chips = []

    def chip(text, state=""):
        chips.append(f'<span class="chip {state}">{escape(text)}</span>')

    text, state = VERDICT_CHIPS.get(output.get("verdict"), ("unscored", ""))
    chip(text, state)
    if output["error"]:
        chip(f"generation failed: {str(output['error'])[:80]}", "bad")
        return f'<div class="chips">{"".join(chips)}</div>'
    chip(*_gate_chip("NLI", output["claim_faithfulness"], output["nli_pass"], gates["nli"], 2))
    if has_rubric:
        chip(*_gate_chip("rubric v3", output["rubric_overall"], output["rubric_pass"],
                         gates["rubric"], 2))
    else:
        chip("rubric v3 not run")
    violations = output["protected_field_violation_count"]
    if violations is not None:
        chip(f"{violations} protected-field violation{'s' if violations != 1 else ''}",
             "bad" if violations else "")
    bait = output["fabrication_bait_match_count"]
    if bait:
        chip(f"{bait} bait term{'s' if bait != 1 else ''}", "bad")
    html = f'<div class="chips">{"".join(chips)}</div>'
    scores = output.get("rubric_scores") or {}
    if any(value is not None for value in scores.values()):
        dims = " · ".join(
            f"{label} {'–' if scores.get(name) is None else format(scores[name], 'g')}"
            for name, label in dd.RUBRIC_DIMENSION_LABELS.items()
        )
        html += f'<div class="model">Rubric v3 (1–5): {escape(dims)}</div>'
    return html


def output_header(workflow, output, gates=None, has_rubric=True) -> str:
    info = dd.WORKFLOW_INFO.get(workflow, {"code": workflow, "title": ""})
    base = " base" if workflow == dd.BASELINE else ""
    tag = ' <span class="tag">baseline</span>' if workflow == dd.BASELINE else ""
    meta = []
    if output.get("total_tokens") is not None:
        meta.append(f"{output['total_tokens']:,} tokens")
    if output.get("wall_seconds") is not None:
        meta.append(f"{output['wall_seconds']:.0f}s")
    return charts.wrap(
        f'<div class="out-head{base}" style="--card-colour:{charts.colour_var(workflow)}">'
        f'<h4><span style="color:{charts.colour_var(workflow)}">{escape(info["code"])}</span> '
        f"{escape(info['title'])}{tag}</h4>"
        f'<div class="model">{" · ".join(meta)}</div>'
        f"{_score_chips(output, gates, has_rubric)}</div>"
    )


def render_pair(path, pair_id, selected=None):
    """(pair info html, resume md, jd md, {workflow: (visible, header html, body md)})."""
    bundle = _bundle(path)
    view = dd.pair_view(bundle, pair_id) if bundle and pair_id else None
    slots = {workflow: (False, "", "") for workflow in SLOT_WORKFLOWS}
    if view is None:
        return charts.empty("Choose a pair to compare outputs."), "", "", slots
    selected = set(selected) if selected is not None else set(view["outputs"])
    for workflow, output in view["outputs"].items():
        if workflow not in slots:
            continue
        body = output["resume_md"] or (
            "_No output: generation failed._" if output["error"] else "_Empty output._"
        )
        header = output_header(workflow, output, view["gates"], view["has_rubric"])
        slots[workflow] = (workflow in selected, header, body)
    facts = [
        ("Pair", view["pair_id"]), ("Category", view["category"].replace("_", " ")),
        ("Fit", view["fit_level"]), ("Role", view["role_family"].replace("_", " ")),
        ("Seniority", view["seniority"]),
    ]
    info = " · ".join(f"{label}: <b>{escape(value)}</b>" for label, value in facts if value)
    available = ", ".join(dd.short_name(name) for name in view["outputs"])
    missing = [dd.short_name(name) for name in SLOT_WORKFLOWS if name not in view["outputs"]]
    info += f"<br>Outputs: {escape(available)}"
    if missing:
        info += f" · not in this evaluation: {escape(', '.join(missing))}"
    good = [dd.short_name(name) for name, output in view["outputs"].items()
            if output["verdict"] == "good"]
    if view["has_rubric"]:
        info += f" · good resumes: <b>{escape(', '.join(good) or 'none')}</b>"
    else:
        info += " · good resumes: unscored (no rubric v3 run found)"
    info += f"<br>{_criteria(view['gates'])}"
    return (
        charts.wrap(f'<div class="status-line">{info}</div>'),
        view["original_resume"] or "_Original resume not available._",
        view["job_description"] or "_Job description not available._",
        slots,
    )


# --- Gradio wiring -----------------------------------------------------------------------------


def _root_choices(evaluations_dir):
    return [(root.label, str(root.path)) for root in dd.discover_roots(evaluations_dir)]


def build_results_tab(blocks: gr.Blocks, evaluations_dir=None, selected_view=None):
    """Add the Results tab inside the current ``gr.Tabs`` of ``blocks``.

    ``selected_view`` opens a sub-tab first: "workflows", "leaderboard", "top" or "viewer".
    """
    models = _models()

    with gr.Tab("Results", id="results"):
        gr.HTML(charts.wrap(
            '<div class="status-line" style="font-size:.95rem">Saved evaluation results for the '
            "seven workflows. This tab only reads files under <code>results/evaluations/</code>; "
            "it never runs a model.</div>"
        ))
        with gr.Row(equal_height=True):
            root_dd = gr.Dropdown(
                choices=[], label="Evaluation run (newest first)", scale=8,
                info="results/evaluations/<dataset>/<generation scope>/<evaluator>",
            )
            refresh = gr.Button("Refresh list", scale=1, size="sm")
        status = gr.HTML(apply_default_css=False)

        with gr.Tabs(selected=selected_view):
            with gr.Tab("1 · Workflows", id="workflows"):
                explainer = gr.HTML(apply_default_css=False)
            with gr.Tab("2 · Leaderboard and trade-offs", id="leaderboard"):
                with gr.Row():
                    category_dd = gr.Dropdown(choices=[("All categories", ALL)], value=ALL,
                                              label="Category")
                    fit_dd = gr.Dropdown(choices=[("All fit levels", ALL)], value=ALL,
                                         label="Fit level")
                board = gr.HTML(**CHART_HTML)
                with gr.Row():
                    cost_radio = gr.Radio(COST_CHOICES, value="total_tokens", label="Cost axis")
                    quality_radio = gr.Radio(
                        QUALITY_CHOICES, value=DEFAULT_QUALITY, label="Quality axis",
                        info="Good-resume rate and rubric v3 need a rubric v3 run; without one "
                        "the plot shows claim faithfulness.",
                    )
                with gr.Row():
                    cost_plot = gr.HTML(**CHART_HTML)
                    gates_plot = gr.HTML(**CHART_HTML)
            with gr.Tab("3 · Top resumes", id="top"):
                top_intro = gr.HTML(apply_default_css=False)
                top_cards, top_headers, top_bodies, top_sources = [], [], [], []
                for workflow in SLOT_WORKFLOWS:
                    with gr.Group(visible=False) as card:
                        top_headers.append(gr.HTML(apply_default_css=False))
                        with gr.Accordion("Tailored resume", open=False):
                            top_bodies.append(gr.Markdown(elem_classes=["out-body"]))
                        with gr.Accordion("Job description and original resume", open=False):
                            top_sources.append(gr.Markdown(elem_classes=["out-body"]))
                    top_cards.append(card)
            with gr.Tab("4 · Output viewer", id="viewer"):
                with gr.Row():
                    pair_dd = gr.Dropdown(choices=[], label="Resume and job pair", scale=3)
                    workflow_pick = gr.CheckboxGroup(
                        choices=[(dd.display_name(name), name) for name in SLOT_WORKFLOWS],
                        value=list(SLOT_WORKFLOWS), label="Show workflows", scale=4,
                    )
                pair_info = gr.HTML(apply_default_css=False)
                with gr.Row():
                    with gr.Accordion("Original resume", open=False):
                        original_md = gr.Markdown()
                    with gr.Accordion("Job description", open=False):
                        jd_md = gr.Markdown()
                slot_columns, slot_headers, slot_bodies = [], [], []
                with gr.Row(elem_classes=["arb-results"]):
                    for workflow in SLOT_WORKFLOWS:
                        with gr.Column(visible=False, min_width=340) as column:
                            slot_headers.append(gr.HTML(apply_default_css=False))
                            slot_bodies.append(gr.Markdown(elem_classes=["out-body"]))
                        slot_columns.append(column)

    top_outputs = [top_intro]
    for card, header, body, source in zip(top_cards, top_headers, top_bodies, top_sources):
        top_outputs += [card, header, body, source]

    def top(path):
        intro, slots = render_top(path)
        values = [intro]
        for workflow in SLOT_WORKFLOWS:
            visible, header, body, source = slots[workflow]
            values += [gr.update(visible=visible), header, body, source]
        return values

    viewer_outputs = [pair_info, original_md, jd_md]
    for column, header, body in zip(slot_columns, slot_headers, slot_bodies):
        viewer_outputs += [column, header, body]

    def viewer(path, pair_id, selected):
        info, resume, jd, slots = render_pair(path, pair_id, selected)
        values = [info, resume, jd]
        for workflow in SLOT_WORKFLOWS:
            visible, header, body = slots[workflow]
            values += [gr.update(visible=visible), header, body]
        return values

    def on_root(path):
        bundle = _bundle(path)
        categories, fit_levels = dd.filter_options(bundle) if bundle else ([], [])
        pairs = dd.pair_choices(bundle) if bundle else []
        pair_id = pairs[0][1] if pairs else None
        quality = (
            quality_metric(dd.leaderboard(bundle)[0]) if bundle else DEFAULT_QUALITY
        )
        cost_html, gates_html = render_tradeoffs(path, ALL, ALL, "total_tokens", quality)
        return [
            render_status(path),
            render_explainer(path, models),
            gr.update(choices=[("All categories", ALL)]
                      + [(c.replace("_", " "), c) for c in categories], value=ALL),
            gr.update(choices=[("All fit levels", ALL)] + [(f, f) for f in fit_levels], value=ALL),
            render_leaderboard(path),
            gr.update(value="total_tokens"),
            gr.update(value=quality),
            cost_html,
            gates_html,
            *top(path),
            gr.update(choices=pairs, value=pair_id),
            gr.update(value=list(SLOT_WORKFLOWS)),
            *viewer(path, pair_id, list(SLOT_WORKFLOWS)),
        ]

    root_outputs = [
        status, explainer, category_dd, fit_dd, board, cost_radio, quality_radio, cost_plot,
        gates_plot, *top_outputs, pair_dd, workflow_pick,
        *viewer_outputs,
    ]

    def initial():
        choices = _root_choices(evaluations_dir)
        value = choices[0][1] if choices else None
        return [gr.update(choices=choices, value=value), *on_root(value)]

    def refresh_list(current):
        choices = _root_choices(evaluations_dir)
        values = [value for _, value in choices]
        value = current if current in values else (values[0] if values else None)
        return [gr.update(choices=choices, value=value), *on_root(value)]

    # Every handler only reads saved files (milliseconds), so no progress overlay is shown.
    quiet = {"show_progress": "hidden"}
    blocks.load(initial, None, [root_dd, *root_outputs], **quiet)
    refresh.click(refresh_list, root_dd, [root_dd, *root_outputs], **quiet)
    root_dd.input(on_root, root_dd, root_outputs, **quiet)

    def on_filter(path, category, fit_level, cost, quality):
        cost_html, gates_html = render_tradeoffs(path, category, fit_level, cost, quality)
        return render_leaderboard(path, category, fit_level), cost_html, gates_html

    filter_inputs = [root_dd, category_dd, fit_dd, cost_radio, quality_radio]
    gr.on([category_dd.input, fit_dd.input], on_filter, filter_inputs,
          [board, cost_plot, gates_plot], **quiet)
    gr.on([cost_radio.input, quality_radio.input], render_tradeoffs, filter_inputs,
          [cost_plot, gates_plot], **quiet)
    gr.on([pair_dd.input, workflow_pick.input], viewer, [root_dd, pair_dd, workflow_pick],
          viewer_outputs, **quiet)

    return {
        "root": root_dd, "refresh": refresh, "status": status, "explainer": explainer,
        "board": board, "top": top,
        "pair": pair_dd, "initial": initial, "on_root": on_root, "viewer": viewer,
    }
