"""Small inline-SVG/HTML charts for the Results tab.

Pure functions that return HTML strings, so they need no plotting dependency and can be unit
tested. Colours are CSS custom properties defined in ``RESULTS_CSS`` (light and ``.dark``), one
fixed hue per workflow; the W2 baseline is a neutral ink with a dashed outline so it reads as the
reference everywhere. Hues W1, W3-W7 are slots 1-6 of the validated categorical palette; every
mark is direct-labelled (W1...W7) and every chart has a table next to it, so identity never rests
on colour alone.
"""
from __future__ import annotations

import math
from html import escape

from eval.dashboard import BASELINE, WORKFLOW_INFO, display_name, short_name

WORKFLOW_COLOURS = {
    # name: (light, dark)
    "w1_single_pass": ("#2a78d6", "#3987e5"),
    "w2_baseline": ("#52514e", "#a8a79f"),
    "w3_critic": ("#eb6834", "#d95926"),
    "w4_evidence": ("#1baf7a", "#199e70"),
    "w5_fine_tuned": ("#eda100", "#c98500"),
    "w6_best_of_n": ("#e87ba4", "#d55181"),
    "w7_plan_execute": ("#008300", "#008300"),
}


def colour_var(workflow: str) -> str:
    code = short_name(workflow).lower()
    return f"var(--wf-{code}, var(--viz-muted))" if workflow in WORKFLOW_COLOURS else "var(--viz-muted)"


def label_ink(workflow: str) -> str:
    """Text colour on a workflow fill: dark on yellow, the surface on the neutral baseline."""
    if workflow == BASELINE:
        return "var(--viz-surface)"
    return "#0b0b0b" if workflow == "w5_fine_tuned" else "#ffffff"


def _colour_rules(index: int) -> str:
    return "\n".join(
        f"  --wf-{short_name(name).lower()}: {colours[index]};"
        for name, colours in WORKFLOW_COLOURS.items()
    )


RESULTS_CSS = f"""
/* Results tab: shadcn/ui-style tokens (zinc neutrals, hairline borders, 12px radius). The chart
   surfaces are the ones the categorical palette was validated against. */
.arb-results {{
  --viz-surface: #fcfcfb;
  --viz-card: #ffffff;
  --viz-ink: #09090b;
  --viz-ink-2: #52525b;
  --viz-muted: #71717a;
  --viz-grid: #e4e4e7;
  --viz-axis: #d4d4d8;
  --viz-border: #e4e4e7;
  --viz-neutral: #f4f4f5;
  --viz-better: #2a78d6;
  --viz-worse: #e34948;
  --viz-shadow: 0 1px 2px rgba(0, 0, 0, .05);
{_colour_rules(0)}
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--viz-ink);
}}
.dark .arb-results {{
  --viz-surface: #18181b;
  --viz-card: #18181b;
  --viz-ink: #fafafa;
  --viz-ink-2: #a1a1aa;
  --viz-muted: #8b8b94;
  --viz-grid: #27272a;
  --viz-axis: #3f3f46;
  --viz-border: #27272a;
  --viz-neutral: #27272a;
  --viz-better: #3987e5;
  --viz-worse: #e66767;
  --viz-shadow: none;
{_colour_rules(1)}
}}
.arb-results code {{ font-size: .78em; padding: .05rem .3rem; border-radius: 4px;
  background: var(--viz-neutral); color: var(--viz-ink); }}
.arb-results .viz {{
  padding: 1.1rem 1.25rem 1rem;
  border: 1px solid var(--viz-border);
  border-radius: 12px;
  background: var(--viz-surface);
  box-shadow: var(--viz-shadow);
}}
.arb-results .viz h3 {{ margin: 0 0 .2rem; font-size: .98rem; font-weight: 600;
  letter-spacing: -.01em; color: var(--viz-ink); }}
.arb-results .viz .sub {{ margin: 0 0 .75rem; font-size: .8rem; line-height: 1.45;
  color: var(--viz-muted); }}
.arb-results .viz svg {{ width: 100%; height: auto; display: block; overflow: visible; }}
.arb-results .viz svg text {{ font-family: inherit; }}
.arb-results .mark {{ cursor: default; outline: none; }}
.arb-results .mark:hover, .arb-results .mark:focus-visible {{ filter: brightness(1.12) saturate(1.05); }}
.arb-results .note {{ font-size: .78rem; line-height: 1.5; color: var(--viz-muted); margin: .6rem 0 0; }}
.arb-results .empty {{
  padding: 1.6rem; border: 1px dashed var(--viz-axis); border-radius: 12px;
  color: var(--viz-muted); background: var(--viz-surface); text-align: center; font-size: .9rem;
}}
.arb-results .gap {{ height: 1rem; }}
.arb-results .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
.arb-results .grid3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }}
.arb-results .grid4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }}
@media (max-width: 860px) {{
  .arb-results .grid2, .arb-results .grid3 {{ grid-template-columns: 1fr; }}
  .arb-results .grid4 {{ grid-template-columns: 1fr 1fr; }}
}}
.arb-results .legend {{ display: flex; flex-wrap: wrap; gap: .35rem 1.1rem; font-size: .78rem;
  color: var(--viz-ink-2); margin: .1rem 0 .7rem; }}
.arb-results .legend span {{ display: inline-flex; align-items: center; gap: .4rem; }}
.arb-results .sw {{ width: 10px; height: 10px; border-radius: 3px; display: inline-block; flex: none; }}
.arb-results .sw.base {{ outline: 1.5px dashed var(--viz-ink-2); outline-offset: 1px; }}
.arb-results .kpi {{ padding: .95rem 1.1rem; border: 1px solid var(--viz-border); border-radius: 12px;
  background: var(--viz-surface); box-shadow: var(--viz-shadow); }}
.arb-results .kpi .k-label {{ font-size: .76rem; font-weight: 500; color: var(--viz-muted); }}
.arb-results .kpi .k-value {{ font-size: 1.65rem; font-weight: 650; letter-spacing: -.02em;
  line-height: 1.2; margin: .2rem 0 .1rem; color: var(--viz-ink); }}
.arb-results .kpi .k-who {{ display: flex; align-items: center; gap: .4rem; font-size: .82rem;
  font-weight: 600; color: var(--viz-ink); }}
.arb-results .kpi .k-delta {{ font-size: .74rem; color: var(--viz-muted); margin-top: .2rem; }}
.arb-results table.tbl {{ width: 100%; border-collapse: collapse; font-size: .83rem;
  font-variant-numeric: tabular-nums; border: none; margin: 0; }}
.arb-results table.tbl th {{ text-align: right; font-weight: 500; font-size: .76rem;
  color: var(--viz-muted); padding: .5rem .6rem; border: none;
  border-bottom: 1px solid var(--viz-border); vertical-align: bottom; background: none; }}
.arb-results table.tbl th:first-child, .arb-results table.tbl td:first-child {{ text-align: left; }}
.arb-results table.tbl td {{ text-align: right; padding: .5rem .6rem; border: none;
  border-bottom: 1px solid var(--viz-border); color: var(--viz-ink); background: none; }}
.arb-results table.tbl tbody tr:hover td {{ background: var(--viz-neutral); }}
.arb-results table.tbl tbody tr:last-child td {{ border-bottom: none; }}
.arb-results table.tbl tr.base td {{ background: color-mix(in srgb, var(--viz-neutral) 65%, transparent); }}
.arb-results table.tbl td[data-tip] {{ cursor: default; outline: none; }}
.arb-results table.tbl td[data-tip]:focus-visible {{ box-shadow: inset 0 0 0 2px var(--viz-axis); }}
.arb-results table.tbl .n {{ display: block; font-size: .7rem; color: var(--viz-muted); }}
.arb-results table.tbl .na {{ color: var(--viz-muted); font-style: italic; }}
.arb-results table.tbl .best {{ font-weight: 700; }}
.arb-results .tblwrap {{ overflow-x: auto; }}
.arb-results .wf {{ display: inline-flex; align-items: center; gap: .45rem; white-space: nowrap;
  font-weight: 500; }}
.arb-results .tag {{ font-size: .66rem; font-weight: 600; padding: .08rem .45rem; border-radius: 6px;
  border: 1px solid var(--viz-border); background: var(--viz-neutral); color: var(--viz-ink-2); }}
.arb-results .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1rem; }}
.arb-results .card {{ padding: 1.05rem 1.15rem; border: 1px solid var(--viz-border);
  border-radius: 12px; background: var(--viz-surface); box-shadow: var(--viz-shadow);
  color: var(--viz-ink); display: flex; flex-direction: column; gap: .6rem;
  position: relative; overflow: hidden; }}
.arb-results .card::before {{ content: ""; position: absolute; inset: 0 0 auto 0; height: 3px;
  background: var(--card-colour); }}
.arb-results .card.base::before {{ background: repeating-linear-gradient(90deg,
  var(--card-colour) 0 8px, transparent 8px 12px); }}
.arb-results .card h3 {{ margin: 0; font-size: 1rem; font-weight: 600; letter-spacing: -.01em;
  display: flex; align-items: center; gap: .45rem; }}
.arb-results .card .code {{ display: inline-grid; place-items: center; min-width: 1.9rem;
  height: 1.35rem; padding: 0 .3rem; border-radius: 6px; font-size: .74rem; font-weight: 700;
  color: #fff; background: var(--card-colour); }}
.arb-results .card p {{ margin: 0; font-size: .84rem; line-height: 1.55; color: var(--viz-ink-2); }}
.arb-results .flow {{ display: flex; flex-wrap: wrap; align-items: center; gap: .3rem; }}
.arb-results .flow .st {{ font-size: .72rem; font-weight: 500; padding: .2rem .5rem;
  border-radius: 6px; border: 1px solid var(--viz-border); background: var(--viz-neutral);
  color: var(--viz-ink); }}
.arb-results .flow .ar {{ color: var(--viz-muted); font-size: .78rem; }}
.arb-results .model {{ font-size: .75rem; color: var(--viz-muted); }}
.arb-results .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: .3rem;
  margin-top: auto; padding-top: .65rem; border-top: 1px solid var(--viz-border); }}
.arb-results .stats b {{ display: block; font-size: 1rem; font-weight: 650; color: var(--viz-ink); }}
.arb-results .stats span {{ font-size: .68rem; color: var(--viz-muted); }}
.arb-results .heat td {{ min-width: 4.6rem; }}
.arb-results .status-line {{ font-size: .82rem; line-height: 1.6; color: var(--viz-muted);
  padding: .15rem .1rem; }}
.arb-results .status-line b {{ color: var(--viz-ink); font-weight: 600; }}
.arb-results .chips {{ display: flex; flex-wrap: wrap; gap: .3rem; margin: .35rem 0 .2rem; }}
.arb-results .chip {{ font-size: .7rem; font-weight: 600; padding: .12rem .5rem; border-radius: 6px;
  border: 1px solid var(--viz-border); background: var(--viz-neutral); color: var(--viz-ink); }}
.arb-results .chip.bad {{ border-color: color-mix(in srgb, var(--viz-worse) 55%, transparent);
  background: color-mix(in srgb, var(--viz-worse) 12%, transparent); }}
.arb-results .chip.good {{ border-color: color-mix(in srgb, var(--viz-better) 55%, transparent);
  background: color-mix(in srgb, var(--viz-better) 12%, transparent); }}
.arb-results .out-head {{ padding: .7rem .85rem; border: 1px solid var(--viz-border);
  border-radius: 12px; background: var(--viz-surface); position: relative; overflow: hidden; }}
.arb-results .out-head::before {{ content: ""; position: absolute; inset: 0 0 auto 0; height: 3px;
  background: var(--card-colour); }}
.arb-results .out-head.base::before {{ background: repeating-linear-gradient(90deg,
  var(--card-colour) 0 8px, transparent 8px 12px); }}
.arb-results .out-head h4 {{ margin: 0; font-size: .92rem; font-weight: 600; color: var(--viz-ink); }}
.arb-results .out-body {{ max-height: 640px !important; overflow-y: auto !important;
  font-size: .86rem; padding: 0 .3rem; }}
.arb-results .out-body h1 {{ font-size: 1.2rem !important; margin: .4rem 0 !important; }}
.arb-results .out-body h2 {{ font-size: 1rem !important; margin: .7rem 0 .3rem !important; }}
.arb-results .out-body h3 {{ font-size: .92rem !important; }}
.arb-tip {{ position: fixed; z-index: 1000; pointer-events: none; opacity: 0;
  transition: opacity .08s ease; max-width: 320px; padding: .45rem .6rem;
  border: 1px solid var(--viz-border); border-radius: 8px; background: var(--viz-card);
  box-shadow: 0 4px 12px rgba(0, 0, 0, .12); font-size: .75rem; line-height: 1.45;
  color: var(--viz-ink-2); }}
.arb-tip strong {{ display: block; font-size: .85rem; font-weight: 650; color: var(--viz-ink); }}
"""


# --- hover tooltips ----------------------------------------------------------------------------

# Attached with ``gr.HTML(js_on_load=TOOLTIP_JS)``: one tooltip per component, driven by
# ``data-tip`` attributes ("value | line | line"); text is inserted with textContent only.
TOOLTIP_JS = """
const tip = document.createElement('div');
tip.className = 'arb-results arb-tip';
tip.setAttribute('role', 'tooltip');
element.appendChild(tip);
const hide = () => { tip.style.opacity = '0'; };
const show = (target, x, y) => {
  if (!tip.isConnected) element.appendChild(tip);
  const [value, ...rest] = (target.getAttribute('data-tip') || '').split(' | ');
  tip.replaceChildren();
  const strong = document.createElement('strong');
  strong.textContent = value;
  tip.appendChild(strong);
  rest.forEach((line) => {
    const row = document.createElement('div');
    row.textContent = line;
    tip.appendChild(row);
  });
  tip.style.opacity = '1';
  const box = tip.getBoundingClientRect();
  let left = x + 14, top = y + 14;
  if (left + box.width > window.innerWidth - 8) left = x - box.width - 14;
  if (top + box.height > window.innerHeight - 8) top = y - box.height - 14;
  tip.style.left = left + 'px';
  tip.style.top = top + 'px';
};
element.addEventListener('pointermove', (event) => {
  const target = event.target.closest ? event.target.closest('[data-tip]') : null;
  if (!target || !element.contains(target)) { hide(); return; }
  show(target, event.clientX, event.clientY);
});
element.addEventListener('pointerleave', hide);
element.addEventListener('focusin', (event) => {
  const target = event.target.closest ? event.target.closest('[data-tip]') : null;
  if (!target) return;
  const box = target.getBoundingClientRect();
  show(target, box.left + box.width / 2, box.top + box.height / 2);
});
element.addEventListener('focusout', hide);
"""


def tip(text: str) -> str:
    """Attributes for a hover/focus tooltip; the first ``|`` segment is shown as the value."""
    value = escape(text, quote=True)
    label = escape(text.replace(" | ", ", "), quote=True)
    return f' data-tip="{value}" tabindex="0" aria-label="{label}"'


# --- formatting --------------------------------------------------------------------------------


def fmt_value(value, kind: str) -> str:
    if value is None:
        return "–"
    if kind == "rate":
        return f"{value * 100:.0f}%"
    if kind == "score":
        return f"{value:.2f}"
    if kind == "scale":
        return f"{value:.2f}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}"


def _wf_label(workflow: str) -> str:
    info = WORKFLOW_INFO.get(workflow)
    swatch = (
        f'<span class="sw{" base" if workflow == BASELINE else ""}" '
        f'style="background:{colour_var(workflow)}"></span>'
    )
    name = f"{info['code']} {info['title']}" if info else workflow
    tag = ' <span class="tag">baseline</span>' if workflow == BASELINE else ""
    return f'<span class="wf">{swatch}{escape(name)}{tag}</span>'


def legend(workflows) -> str:
    items = []
    for workflow in workflows:
        info = WORKFLOW_INFO.get(workflow)
        name = f"{info['code']} {info['title']}" if info else workflow
        cls = "sw base" if workflow == BASELINE else "sw"
        items.append(
            f'<span><i class="{cls}" style="background:{colour_var(workflow)}"></i>'
            f"{escape(name)}</span>"
        )
    return f'<div class="legend">{"".join(items)}</div>'


def empty(message: str) -> str:
    return f'<div class="arb-results"><div class="empty">{escape(message)}</div></div>'


def panel(title: str, body: str, subtitle: str = "", note: str = "") -> str:
    sub = f'<p class="sub">{subtitle}</p>' if subtitle else ""
    foot = f'<p class="note">{note}</p>' if note else ""
    return f'<div class="viz"><h3>{escape(title)}</h3>{sub}{body}{foot}</div>'


def wrap(html: str) -> str:
    return f'<div class="arb-results">{html}</div>'


# --- horizontal bars ---------------------------------------------------------------------------


def bar_chart(items, *, kind: str = "rate", domain=(0.0, 1.0), width: int = 360) -> str:
    """One bar per workflow. ``items``: workflow, value (None = unscored), label (right text)."""
    if not items:
        return '<div class="empty">No rows</div>'
    row_h, gap, left, right = 22, 8, 34, 92
    plot_w = width - left - right
    height = len(items) * (row_h + gap) + 6
    low, high = domain
    span = (high - low) or 1.0

    def x(value):
        return left + max(0.0, min(1.0, (value - low) / span)) * plot_w

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="bar chart" '
        f'style="max-width:{int(width * 1.4)}px">',
        f'<line x1="{left}" y1="0" x2="{left}" y2="{height - 4}" '
        'style="stroke:var(--viz-axis);stroke-width:1"/>',
    ]
    for tick in (0.5, 1.0):
        tx = left + tick * plot_w
        parts.append(
            f'<line x1="{tx:.1f}" y1="0" x2="{tx:.1f}" y2="{height - 4}" '
            'style="stroke:var(--viz-grid);stroke-width:1"/>'
        )
    for index, item in enumerate(items):
        workflow, value = item["workflow"], item.get("value")
        y = index * (row_h + gap) + 3
        mid = y + row_h / 2
        parts.append(
            f'<text x="{left - 6}" y="{mid + 4:.1f}" text-anchor="end" '
            f'style="fill:var(--viz-ink);font-size:12px;font-weight:600">'
            f"{escape(short_name(workflow))}</text>"
        )
        text = item.get("tip") or f"{item.get('label', '')} | {display_name(workflow)}"
        if value is None:
            parts.append(
                f'<text x="{left + 6}" y="{mid + 4:.1f}"{tip(text)} '
                f'style="fill:var(--viz-muted);font-size:11px;font-style:italic">'
                f'{escape(item.get("label") or "unscored")}</text>'
            )
            continue
        end = x(value)
        bar_w = max(end - left, 2.0)
        baseline = workflow == BASELINE
        stroke = (
            ';stroke:var(--viz-ink-2);stroke-width:1.2;stroke-dasharray:3 2' if baseline else ""
        )
        # 4px rounded data end, square at the baseline: a rounded rect plus a square cap.
        parts.append(
            f'<g class="mark"{tip(text)}>'
            f'<rect x="0" y="{y - gap / 2}" width="{width}" height="{row_h + gap}" '
            'style="fill:transparent"/>'
            f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="{row_h}" rx="4" '
            f'style="fill:{colour_var(workflow)}{stroke}"/>'
            f'<rect x="{left}" y="{y}" width="{min(bar_w, 4):.1f}" height="{row_h}" '
            f'style="fill:{colour_var(workflow)}"/></g>'
        )
        parts.append(
            f'<text x="{end + 6:.1f}" y="{mid + 4:.1f}" '
            f'style="fill:var(--viz-ink);font-size:11.5px">{escape(item.get("label", ""))}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


# --- scatter -----------------------------------------------------------------------------------


def _nice_step(span, count=4):
    raw = span / count
    magnitude = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 2.5, 5, 10):
        if raw <= factor * magnitude:
            return factor * magnitude
    return 10 * magnitude


def _nice_domain(values, pad=0.08, clamp=None):
    """Padded domain snapped to round tick steps, kept inside ``clamp`` (None = open end)."""
    low, high = min(values), max(values)
    if high - low < 1e-9:
        spread = abs(low) * 0.1 or 0.5
        low, high = low - spread, high + spread
    margin = (high - low) * pad
    low, high = low - margin, high + margin
    if clamp:
        if clamp[0] is not None:
            low = max(clamp[0], low)
        if clamp[1] is not None:
            high = min(clamp[1], high)
    step = _nice_step(high - low)
    low, high = math.floor(low / step) * step, math.ceil(high / step) * step
    if clamp:
        if clamp[0] is not None:
            low = max(clamp[0], low)
        if clamp[1] is not None:
            high = min(clamp[1], high)
    return low, high


def _ticks(low, high):
    step = _nice_step(high - low)
    first = math.ceil(low / step - 1e-9) * step
    ticks, value = [], first
    while value <= high + 1e-9:
        ticks.append(round(value, 10))
        value += step
    return ticks


def scatter_chart(
    points, *, x_label: str, y_label: str, x_kind: str, y_kind: str,
    x_better: str = "lower", y_better: str = "higher", width: int = 460, height: int = 320,
    x_clamp=None, y_clamp=None,
) -> str:
    """Workflows as labelled dots; the baseline is a dashed reference cross-hair."""
    if not points:
        return '<div class="empty">No workflow is scored on both measures yet.</div>'
    left, right, top, bottom = 52, 18, 14, 44
    plot_w, plot_h = width - left - right, height - top - bottom
    x_low, x_high = _nice_domain([p["x"] for p in points], clamp=x_clamp)
    y_low, y_high = _nice_domain([p["y"] for p in points], clamp=y_clamp)

    def sx(value):
        return left + (value - x_low) / ((x_high - x_low) or 1) * plot_w

    def sy(value):
        return top + plot_h - (value - y_low) / ((y_high - y_low) or 1) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="scatter plot" '
        f'style="max-width:{int(width * 1.3)}px;margin:0 auto">'
    ]
    for tick in _ticks(y_low, y_high):
        ty = sy(tick)
        parts.append(
            f'<line x1="{left}" y1="{ty:.1f}" x2="{left + plot_w}" y2="{ty:.1f}" '
            'style="stroke:var(--viz-grid);stroke-width:1"/>'
            f'<text x="{left - 6}" y="{ty + 4:.1f}" text-anchor="end" '
            f'style="fill:var(--viz-muted);font-size:10.5px">{fmt_value(tick, y_kind)}</text>'
        )
    for tick in _ticks(x_low, x_high):
        tx = sx(tick)
        parts.append(
            f'<text x="{tx:.1f}" y="{top + plot_h + 16}" text-anchor="middle" '
            f'style="fill:var(--viz-muted);font-size:10.5px">{fmt_value(tick, x_kind)}</text>'
        )
    parts.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" '
        'style="stroke:var(--viz-axis);stroke-width:1"/>'
    )
    x_hint = "← better" if x_better == "lower" else "better →"
    y_hint = "better ↑" if y_better == "higher" else "better ↓"
    parts.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 6}" text-anchor="middle" '
        f'style="fill:var(--viz-ink-2);font-size:11.5px">{escape(x_label)} '
        f'<tspan style="fill:var(--viz-muted)">({x_hint})</tspan></text>'
        f'<text transform="translate(13 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" '
        f'style="fill:var(--viz-ink-2);font-size:11.5px">{escape(y_label)} '
        f'<tspan style="fill:var(--viz-muted)">({y_hint})</tspan></text>'
    )
    baseline = next((p for p in points if p["workflow"] == BASELINE), None)
    if baseline:
        bx, by = sx(baseline["x"]), sy(baseline["y"])
        parts.append(
            f'<line x1="{bx:.1f}" y1="{top}" x2="{bx:.1f}" y2="{top + plot_h}" '
            'style="stroke:var(--viz-ink-2);stroke-width:1;stroke-dasharray:4 3;opacity:.55"/>'
            f'<line x1="{left}" y1="{by:.1f}" x2="{left + plot_w}" y2="{by:.1f}" '
            'style="stroke:var(--viz-ink-2);stroke-width:1;stroke-dasharray:4 3;opacity:.55"/>'
        )
    placed = []
    for point in sorted(points, key=lambda p: (p["workflow"] == BASELINE, p["x"])):
        px, py = sx(point["x"]), sy(point["y"])
        workflow = point["workflow"]
        text = (
            f"{display_name(workflow)} | {y_label}: {fmt_value(point['y'], y_kind)} "
            f"(n={point['y_n']}) | {x_label}: {fmt_value(point['x'], x_kind)} (n={point['x_n']})"
        )
        if workflow == BASELINE:
            mark = (
                f'<rect x="{px - 6:.1f}" y="{py - 6:.1f}" width="12" height="12" rx="2" '
                f'transform="rotate(45 {px:.1f} {py:.1f})" '
                f'style="fill:{colour_var(workflow)};stroke:var(--viz-surface);stroke-width:2"/>'
            )
        else:
            mark = (
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6.5" '
                f'style="fill:{colour_var(workflow)};stroke:var(--viz-surface);stroke-width:2"/>'
            )
        # Direct label; flip to the left or nudge vertically when it would collide.
        lx, ly, anchor = px + 10, py + 4, "start"
        if lx + 24 > left + plot_w:
            lx, anchor = px - 10, "end"
        for other_x, other_y in placed:
            if abs(other_x - lx) < 26 and abs(other_y - ly) < 12:
                ly += 13
        placed.append((lx, ly))
        parts.append(
            f'<g class="mark"{tip(text)}>'
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="14" style="fill:transparent"/>{mark}'
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" '
            f'style="fill:var(--viz-ink);font-size:11.5px;font-weight:600">'
            f"{escape(short_name(workflow))}</text></g>"
        )
    parts.append("</svg>")
    return "".join(parts)


