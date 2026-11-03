"""The read-only "Results" tab: explains W1-W7 and shows saved evaluation results.

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
from ui import charts

ALL = ""
COST_CHOICES = [("Tokens per output", "total_tokens"), ("Wall time per output", "historical_run_wall_seconds")]
QUALITY_CHOICES = [("Claim faithfulness", "claim_faithfulness"), ("Rubric judge", "judge_overall_mean")]
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
    return charts.wrap(
        '<div class="status-line">'
        f"Evaluator <code>{escape(root.name)}</code> · generation scope "
        f"<code>{escape(root.parent.name)}</code> · dataset <code>{escape(dataset[:12])}</code>"
        f"<br><b>{len(bundle.rows)}</b> scored outputs · workflows: <b>{escape(workflows)}</b>"
        f" · {mark('score')} · {mark('pairwise')} · {mark('robustness')} · {summaries}"
        "</div>"
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
            task = (
                f"{profile['task_success']}/{profile['task_scored']}"
                if profile["task_scored"] else "unscored"
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
                + _stat(task, "task success")
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


def _visible_metrics(board):
    return [
        spec for spec in dd.LEADERBOARD_METRICS
        if spec.key != "judge_overall_mean" or dd.metric_is_present(board, spec.key)
    ]


def _best(board, spec):
    values = [row["cells"][spec.key]["value"] for row in board]
    values = [value for value in values if value is not None]
    if len(values) < 2 or spec.tier == "cost" or len(set(values)) == 1:
        return None
    return max(values) if spec.better == "higher" else min(values)


def render_leaderboard(path, category=ALL, fit_level=ALL) -> str:
    bundle = _bundle(path)
    if bundle is None:
        return charts.empty(NO_RUN)
    board, source = dd.leaderboard(bundle, category or None, fit_level or None)
    if not board:
        return charts.empty("No scored outputs match this filter.")
    specs = _visible_metrics(board)

    def bars(metric):
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
            items.append({"workflow": row["workflow"], "value": cell["value"], "label": label})
        return charts.bar_chart(items, kind=spec.kind, width=480)

    headline = (
        _kpi_tiles(board)
        + '<div class="gap"></div><div class="grid2">'
        + charts.panel(
            "Task success", bars("task_success"),
            "Finished, all four sections, 200-650 words, claim faithfulness ≥ "
            f"{config.TASK_FAITHFULNESS_THRESHOLD:.0%}, no protected-field or bait violation.",
        )
        + charts.panel(
            "Claim faithfulness", bars("claim_faithfulness"),
            "Share of output claims the NLI referee finds supported by the original resume.",
        )
        + "</div>"
    )
    header = "".join(
        f"<th>{escape(spec.label)}<br><span class=\"n\">"
        f"{'↑ better' if spec.better == 'higher' else '↓ better'}</span></th>"
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
        f"Scope: {escape(scope)}. Each cell shows the mean (or rate) and how many outputs were "
        "scored; outputs a metric could not score are counted as unscored, never as 0. Bold = "
        "best among scored workflows. Hover a value for its 95% CI. Source: "
        f"<code>{escape(source)}</code>."
    )
    table = charts.panel(
        "Leaderboard", f'<div class="tblwrap"><table class="tbl"><thead><tr><th>Workflow</th>'
        f"{header}</tr></thead><tbody>{''.join(body)}</tbody></table></div>",
        note=note,
    )
    return charts.wrap(headline + '<div class="gap"></div>' + table)


def render_tradeoffs(path, category=ALL, fit_level=ALL, cost="total_tokens",
                     quality="claim_faithfulness") -> tuple[str, str]:
    bundle = _bundle(path)
    if bundle is None:
        return charts.empty(NO_RUN), ""
    board, _ = dd.leaderboard(bundle, category or None, fit_level or None)
    workflows = [row["workflow"] for row in board]
    cost_spec, quality_spec = dd.METRIC_SPECS[cost], dd.METRIC_SPECS[quality]
    if not dd.metric_is_present(board, quality):
        quality_spec = dd.METRIC_SPECS["claim_faithfulness"]
    cost_points = dd.tradeoff_points(board, cost_spec.key, quality_spec.key)
    missing = [dd.short_name(w) for w in workflows if w not in {p["workflow"] for p in cost_points}]
    first = charts.panel(
        f"Quality vs cost: {quality_spec.label.lower()} vs {cost_spec.label.lower()}",
        charts.scatter_chart(
            cost_points, x_label=cost_spec.label, y_label=quality_spec.label,
            x_kind="number", y_kind=quality_spec.kind, x_better="lower", x_clamp=(0, None),
            y_clamp=(0, 1) if quality_spec.kind != "scale" else (1, 5),
        ),
        "Top-left is best: more faithful output for less compute. Dashed lines mark W2.",
        note=("Not scored on both: " + ", ".join(missing)) if missing else "",
    )
    coverage_points = dd.tradeoff_points(board, "requirement_coverage", "claim_faithfulness")
    missing = [
        dd.short_name(w) for w in workflows if w not in {p["workflow"] for p in coverage_points}
    ]
    second = charts.panel(
        "Faithfulness vs requirement coverage",
        charts.scatter_chart(
            coverage_points, x_label="Requirement coverage", y_label="Claim faithfulness",
            x_kind="score", y_kind="score", x_better="higher", x_clamp=(0, 1), y_clamp=(0, 1),
        ),
        "Top-right is best: covers the JD's supportable requirements without unsupported claims.",
        note=("Not scored on both: " + ", ".join(missing)) if missing else "",
    )
    return charts.wrap(charts.legend(workflows) + first), charts.wrap(
        charts.legend(workflows) + second
    )


# --- 3. pairwise and robustness ----------------------------------------------------------------


def _rate(value):
    return "–" if value is None else f"{float(value) * 100:.0f}%"


def render_pairwise(path) -> str:
    bundle = _bundle(path)
    if bundle is None:
        return charts.empty(NO_RUN)
    rows = dd.pairwise_table(bundle)
    if not rows:
        return charts.empty("Pairwise comparison not run for this evaluation.")
    chart = charts.pairwise_legend() + charts.pairwise_chart(rows)
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{charts._wf_label(row.get('challenger_workflow', ''))}</td>"
            f"<td>{row.get('challenger_wins', 0)}</td><td>{row.get('ties', 0)}</td>"
            f"<td>{row.get('inconsistent', 0)}</td><td>{row.get('baseline_wins', 0)}</td>"
            f"<td>{_rate(row.get('strict_win_rate'))}</td>"
            f"<td>{_rate(row.get('position_swap_consistency'))}"
            f"<span class=\"n\">{row.get('consistent', 0)}/{row.get('scored', 0)}</span></td>"
            f"<td>{row.get('scored', 0)}<span class=\"n\">of {row.get('eligible', 0)} eligible"
            f"</span></td><td>{row.get('unscored', 0)}"
            f"<span class=\"n\">{row.get('failed', 0)} failed · {row.get('invalid', 0)} invalid"
            f"</span></td><td>{row.get('generation_unavailable', 0)}</td>"
            "</tr>"
        )
    table = (
        '<div class="tblwrap"><table class="tbl"><thead><tr><th>Challenger</th><th>Wins</th>'
        "<th>Ties</th><th>Order-dependent</th><th>W2 wins</th><th>Strict win rate</th>"
        "<th>Swap consistency</th><th>Scored</th><th>Unscored</th><th>Unavailable</th>"
        f"</tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>"
    )
    note = (
        "Each pair is judged twice with the two resumes in swapped positions. A win or loss "
        "counts only when both orders agree; otherwise it is order-dependent. Rates use scored "
        "pairs as the denominator; unavailable = a missing or failed generation on either side."
    )
    return charts.wrap(
        charts.panel("Head-to-head against the W2 baseline", chart,
                     "How often a judge prefers each challenger's resume over W2's for the same pair.")
        + '<div class="gap"></div>'
        + charts.panel("Pairwise counts", table, note=note)
    )


def render_robustness(path) -> str:
    bundle = _bundle(path)
    if bundle is None:
        return charts.empty(NO_RUN)
    if not bundle.has_robustness:
        return charts.empty("Robustness not run for this evaluation.")
    panels = dd.robustness_family_panels(bundle)
    if not any(panel["items"] for panel in panels):
        return charts.empty(
            "Robustness rows exist but no v2 family tables could be built (older robustness run)."
        )
    blocks = []
    workflows = set()
    for panel in panels:
        items = []
        for item in panel["items"]:
            workflows.add(item["workflow"])
            if item["value"] is None:
                label = f"unscored (0/{item['denominator']} of {item['runs']} runs)"
            else:
                label = f"{_rate(item['value'])} · {item['numerator']}/{item['denominator']}"
            text = f"{label} | {dd.display_name(item['workflow'])} | variants: {item['variants']}"
            text += f" | {item['runs']} runs"
            if item["note"]:
                text += f" | {item['note']}"
            items.append({"workflow": item["workflow"], "value": item["value"], "label": label,
                          "tip": text})
        subtitle = (
            "Stable runs: no pass/fail check flipped to worse and no 0-1 metric dropped by more "
            f"than {config.ROBUSTNESS_STABILITY_TOLERANCE:.2f}."
            if panel["measure"] == "stability"
            else "Expected behaviour: refused or errored cleanly, or completed without "
            "fabrication or protected-field changes."
        )
        body = charts.bar_chart(items) if items else '<div class="empty">Not run</div>'
        blocks.append(charts.panel(panel["title"], body, subtitle))
    oos = dd.out_of_scope_variant_rows(bundle)
    oos_table = ""
    if oos:
        rows = "".join(
            "<tr>"
            f"<td>{charts._wf_label(row.get('workflow', ''))}</td>"
            f"<td style=\"text-align:left\">{escape(str(row.get('variant', '')))}</td>"
            f"<td>{_rate(row.get('expected_behaviour_pass_rate'))}<span class=\"n\">"
            f"{row.get('expected_behaviour_passes', 0)}/{row.get('scored', 0)} scored"
            f"</span></td>"
            f"<td>{row.get('completed', 0)}</td><td>{row.get('refused', 0)}</td>"
            f"<td>{row.get('handled_errors', 0)}</td><td>{row.get('crashed', 0)}</td>"
            f"<td>{row.get('fabrication_failures', 0)}<span class=\"n\">of "
            f"{row.get('fabrication_checked', 0)} checked</span></td>"
            f"<td>{row.get('protected_field_failures', 0)}<span class=\"n\">of "
            f"{row.get('protected_fields_checked', 0)} checked</span></td>"
            "</tr>"
            for row in oos
        )
        oos_table = charts.panel(
            "Out-of-scope inputs by variant",
            '<div class="tblwrap"><table class="tbl"><thead><tr><th>Workflow</th>'
            '<th style="text-align:left">Variant</th><th>Pass rate</th><th>Completed</th>'
            "<th>Refused</th><th>Handled error</th><th>Crashed</th><th>Fabrication</th>"
            f"<th>Protected fields</th></tr></thead><tbody>{rows}</tbody></table></div>",
            "mismatched_jd: a JD from another field · degenerate_jd: title only, one line, or "
            "non-job text.",
        )
    return charts.wrap(
        charts.legend(dd.order_workflows(workflows))
        + f'<div class="grid3">{"".join(blocks)}</div>'
        + ('<div class="gap"></div>' + oos_table if oos_table else "")
    )


def render_deltas(path, metric="faithfulness") -> str:
    bundle = _bundle(path)
    if bundle is None or not bundle.has_robustness:
        return ""
    matrix = dd.robustness_delta_matrix(bundle, metric or "faithfulness")
    if not matrix["variants"]:
        return charts.empty("No in-scope robustness deltas for this metric.")
    direction = matrix["direction"]
    header = "".join(
        f"<th>{escape(variant.replace('_', ' '))}<br><span class=\"n\">{escape(family)}</span></th>"
        for family, variant in matrix["variants"]
    )
    body = []
    for workflow in matrix["workflows"]:
        cells = []
        for _, variant in matrix["variants"]:
            row = matrix["cells"].get((workflow, variant))
            if not row:
                cells.append('<td><span class="na">–</span></td>')
                continue
            delta = row.get("mean_delta")
            if delta is None:
                cells.append(
                    f'<td><span class="na">unscored</span><span class="n">0/'
                    f'{row.get("eligible", 0)}</span></td>'
                )
                continue
            text = (
                f"Δ {float(delta):+.3f} · {variant.replace('_', ' ')} | {dd.display_name(workflow)}"
                f" | 95% CI {row.get('ci_95_low')} to {row.get('ci_95_high')}"
                f" | worst degradation {row.get('worst_degradation')}"
                f" | {row.get('scored', 0)} paired of {row.get('eligible', 0)} eligible,"
                f" {row.get('failed', 0)} variant generations failed"
            )
            cells.append(
                f'<td{charts.tip(text)} style="{charts.delta_style(float(delta), direction)}">'
                f"{float(delta):+.3f}<span class=\"n\">n={row.get('scored', 0)}"
                f"{' of ' + str(row.get('eligible')) if row.get('scored') != row.get('eligible') else ''}"
                "</span></td>"
            )
        cls = ' class="base"' if workflow == dd.BASELINE else ""
        body.append(f"<tr{cls}><td>{charts._wf_label(workflow)}</td>{''.join(cells)}</tr>")
    better = "higher" if direction == "higher_is_better" else "lower"
    return charts.wrap(charts.panel(
        f"Per-variant change: {dd.DELTA_METRIC_LABELS.get(metric, metric)}",
        f'<div class="tblwrap"><table class="tbl heat"><thead><tr><th>Workflow</th>{header}'
        f"</tr></thead><tbody>{''.join(body)}</tbody></table></div>",
        f"Mean of (variant − original) on the same pair; {better} is better. Blue = improved, "
        "red = degraded, grey ≈ no change. n = pairs with both runs scored.",
    ))


# --- 4. output viewer --------------------------------------------------------------------------


def _score_chips(output):
    chips = []
    if output["error"]:
        return f'<div class="chips"><span class="chip bad">generation failed: {escape(str(output["error"])[:80])}</span></div>'

    def chip(text, state=""):
        chips.append(f'<span class="chip {state}">{escape(text)}</span>')

    faith = output["claim_faithfulness"]
    chip(f"faithfulness {faith:.2f}" if faith is not None else "faithfulness unscored")
    coverage = output["requirement_coverage"]
    if coverage is not None:
        chip(f"coverage {coverage:.2f}")
    elif output["requirement_coverage_status"] == "no_supportable_requirements":
        chip("coverage n/a")
    else:
        chip("coverage unscored")
    violations = output["protected_field_violation_count"]
    if violations is not None:
        chip(f"{violations} protected-field violation{'s' if violations != 1 else ''}",
             "bad" if violations else "")
    bait = output["fabrication_bait_match_count"]
    if bait:
        chip(f"{bait} bait term{'s' if bait != 1 else ''}", "bad")
    task = output["task_success"]
    chip("task ✓" if task else ("task ✗" if task is False else "task unscored"),
         "good" if task else ("bad" if task is False else ""))
    pairwise = output["pairwise"]
    if pairwise and pairwise != "baseline":
        chip(f"vs W2: {pairwise}",
             "good" if pairwise == "win" else ("bad" if pairwise == "loss" else ""))
    return f'<div class="chips">{"".join(chips)}</div>'


def output_header(workflow, output) -> str:
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
        f'<div class="model">{" · ".join(meta)}</div>{_score_chips(output)}</div>'
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
        slots[workflow] = (workflow in selected, output_header(workflow, output), body)
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

    ``selected_view`` opens a sub-tab first: "workflows", "leaderboard", "pairwise" or "viewer".
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
                    quality_radio = gr.Radio(QUALITY_CHOICES, value="claim_faithfulness",
                                             label="Quality axis",
                                             info="Rubric judge appears only when it was run.")
                with gr.Row():
                    cost_plot = gr.HTML(**CHART_HTML)
                    coverage_plot = gr.HTML(**CHART_HTML)
            with gr.Tab("3 · Pairwise and robustness", id="pairwise"):
                pairwise = gr.HTML(**CHART_HTML)
                robustness = gr.HTML(**CHART_HTML)
                delta_dd = gr.Dropdown(
                    choices=[(label, key) for key, label in dd.DELTA_METRIC_LABELS.items()],
                    value="faithfulness", label="Per-variant change in",
                )
                deltas = gr.HTML(**CHART_HTML)
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
        delta_metrics = dd.robustness_delta_metrics(bundle) if bundle else []
        delta_metric = "faithfulness" if "faithfulness" in delta_metrics else (
            delta_metrics[0] if delta_metrics else "faithfulness"
        )
        quality = "claim_faithfulness"
        cost_html, coverage_html = render_tradeoffs(path, ALL, ALL, "total_tokens", quality)
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
            coverage_html,
            render_pairwise(path),
            render_robustness(path),
            gr.update(
                choices=[(dd.DELTA_METRIC_LABELS[m], m) for m in delta_metrics]
                or [(label, key) for key, label in dd.DELTA_METRIC_LABELS.items()],
                value=delta_metric,
            ),
            render_deltas(path, delta_metric),
            gr.update(choices=pairs, value=pair_id),
            gr.update(value=list(SLOT_WORKFLOWS)),
            *viewer(path, pair_id, list(SLOT_WORKFLOWS)),
        ]

    root_outputs = [
        status, explainer, category_dd, fit_dd, board, cost_radio, quality_radio, cost_plot,
        coverage_plot, pairwise, robustness, delta_dd, deltas, pair_dd, workflow_pick,
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
        cost_html, coverage_html = render_tradeoffs(path, category, fit_level, cost, quality)
        return render_leaderboard(path, category, fit_level), cost_html, coverage_html

    filter_inputs = [root_dd, category_dd, fit_dd, cost_radio, quality_radio]
    gr.on([category_dd.input, fit_dd.input], on_filter, filter_inputs,
          [board, cost_plot, coverage_plot], **quiet)
    gr.on([cost_radio.input, quality_radio.input], render_tradeoffs, filter_inputs,
          [cost_plot, coverage_plot], **quiet)
    delta_dd.input(render_deltas, [root_dd, delta_dd], deltas, **quiet)
    gr.on([pair_dd.input, workflow_pick.input], viewer, [root_dd, pair_dd, workflow_pick],
          viewer_outputs, **quiet)

    return {
        "root": root_dd, "refresh": refresh, "status": status, "explainer": explainer,
        "board": board, "pairwise": pairwise, "robustness": robustness, "deltas": deltas,
        "pair": pair_dd, "initial": initial, "on_root": on_root, "viewer": viewer,
    }
