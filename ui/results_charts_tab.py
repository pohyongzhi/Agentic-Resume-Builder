"""The read-only "Results (charts)" tab: the d3 results page for one saved evaluation.

The page (``ui/static/results_charts.html``) is self-contained: its data comes from
``eval.dashboard_charts`` and d3 v7.8.5 is inlined from ``ui/static/``, so nothing loads from the
network. ``gr.HTML`` does not run ``<script>`` tags, so the page is shown in an
``<iframe srcdoc=...>``; a small script inside the page grows the frame to its content height.
The evaluation-run list reuses ``eval.dashboard.discover_roots`` (and ``ARB_EVALUATIONS_DIR``).
"""
from __future__ import annotations

import json
from functools import lru_cache
from html import escape
from pathlib import Path

import gradio as gr

from eval import dashboard as dd
from eval import dashboard_charts as dc
from ui import charts

STATIC = Path(__file__).resolve().parent / "static"
TEMPLATE = STATIC / "results_charts.html"
D3 = STATIC / "d3.v7.8.5.min.js"
FRAME_HEIGHT = 2400  # starting height; the page resizes the frame once it has drawn

NO_RUN = "No saved evaluation found. Run evaluate.py, then press Refresh."
NO_OUTPUTS = "This evaluation has no saved outputs to chart yet."


@lru_cache(maxsize=1)
def _assets() -> tuple[str, str]:
    return TEMPLATE.read_text(encoding="utf-8"), D3.read_text(encoding="utf-8")


def _script_safe_json(payload) -> str:
    """JSON that cannot close or comment out its ``<script>`` element."""
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def render_page(payload: dict) -> str:
    """The complete, offline chart page for one payload from ``build_payload``."""
    template, d3 = _assets()
    thresholds = payload["thresholds"]
    page = (
        template.replace("__FAITH_MIN__", f"{thresholds['faith']:g}")
        .replace("__MIN_WORDS__", str(thresholds["min_words"]))
        .replace("__MAX_WORDS__", str(thresholds["max_words"]))
        .replace("__DATA__", _script_safe_json(payload))
    )
    # d3 last, so no placeholder text inside the library can be replaced by mistake.
    head, tail = page.split("__D3__", 1)
    return head + d3.replace("</script", "<\\/script") + tail


def render_frame(path) -> str:
    """The tab's HTML: the chart page in a srcdoc iframe, or a friendly message."""
    if not path or not Path(path).is_dir():
        return charts.empty(NO_RUN)
    payload = dc.build_payload(path)
    if payload is None:
        return charts.empty(NO_OUTPUTS)
    return (
        f'<iframe class="arb-charts-frame" title="Results charts" srcdoc="{escape(render_page(payload), quote=True)}" '
        f'style="width:100%;height:{FRAME_HEIGHT}px;border:0;display:block;border-radius:8px" '
        'scrolling="auto"></iframe>'
    )


def _root_choices(evaluations_dir):
    return [(root.label, str(root.path)) for root in dd.discover_roots(evaluations_dir)]


def build_results_charts_tab(blocks: gr.Blocks, evaluations_dir=None):
    """Add the "Results (charts)" tab inside the current ``gr.Tabs`` of ``blocks``."""
    with gr.Tab("Results (charts)", id="results_charts"):
        gr.HTML(charts.wrap(
            '<div class="status-line" style="font-size:.95rem">The same saved results as the '
            "Results tab, drawn as one chart page. Reads files under "
            "<code>results/evaluations/</code> only; it never runs a model.</div>"
        ))
        with gr.Row(equal_height=True):
            root_dd = gr.Dropdown(
                choices=[], label="Evaluation run (newest first)", scale=8,
                info="results/evaluations/<dataset>/<generation scope>/<evaluator>",
            )
            refresh = gr.Button("Refresh list", scale=1, size="sm")
        page = gr.HTML(apply_default_css=False, padding=False)

    def initial():
        choices = _root_choices(evaluations_dir)
        value = choices[0][1] if choices else None
        return gr.update(choices=choices, value=value), render_frame(value)

    def refresh_list(current):
        choices = _root_choices(evaluations_dir)
        values = [value for _, value in choices]
        value = current if current in values else (values[0] if values else None)
        return gr.update(choices=choices, value=value), render_frame(value)

    quiet = {"show_progress": "hidden"}
    blocks.load(initial, None, [root_dd, page], **quiet)
    refresh.click(refresh_list, root_dd, [root_dd, page], **quiet)
    root_dd.input(render_frame, root_dd, page, **quiet)
    return {"root": root_dd, "refresh": refresh, "page": page, "initial": initial,
            "refresh_list": refresh_list, "render": render_frame}

