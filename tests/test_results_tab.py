"""Results tab: render functions, charts and a Blocks smoke test (nothing is launched)."""
import socket

import gradio as gr
import pytest

from eval import dashboard as dd
from tests.dashboard_fixture import PAIRS, WORKFLOWS, build_fixture_root
from ui import charts
from ui import results_tab as rt


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted in a results-tab test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(dd.config, "REPORT_BOOTSTRAP_SAMPLES", 50)
    dd.clear_cache()


@pytest.fixture(scope="module")
def evaluations(tmp_path_factory):
    base = tmp_path_factory.mktemp("evaluations")
    build_fixture_root(base, bootstrap_samples=50)
    return base


@pytest.fixture
def root(evaluations):
    return str(dd.discover_roots(evaluations)[0].path)


def test_every_workflow_has_a_distinct_colour_and_description():
    assert set(charts.WORKFLOW_COLOURS) == set(dd.WORKFLOW_ORDER) == set(dd.WORKFLOW_INFO)
    light = [colours[0] for colours in charts.WORKFLOW_COLOURS.values()]
    assert len(set(light)) == len(light)
    for name in dd.WORKFLOW_ORDER:
        assert f"--wf-{dd.short_name(name).lower()}" in charts.RESULTS_CSS


def test_explainer_shows_models_and_measured_averages(root):
    html = rt.render_explainer(root, models={name: "base-model" for name in WORKFLOWS})
    for name in WORKFLOWS:
        assert dd.WORKFLOW_INFO[name]["title"] in html
    assert "LLM calls" in html and "task success" in html
    assert "2/3" in html  # W3: one generation failure leaves 3 task-success scores
    assert "agentic-resume-v4-1" in html  # the measured W5 model differs from the passed one


def test_explainer_without_results_still_explains(tmp_path):
    html = rt.render_explainer(str(tmp_path / "missing"))
    assert html.count('<h3><span class="code"') == 7 and "Not in this evaluation" in html


def test_leaderboard_shows_denominators_and_unscored(root):
    html = rt.render_leaderboard(root)
    assert "Task success" in html and "Requirement coverage" in html
    assert "n=3 of 4" in html  # W6 coverage: one evaluator error
    assert "Rubric judge" not in html  # judge skipped in the fixture, column hidden
    assert "summaries/overall.json" in html
    filtered = rt.render_leaderboard(root, "software_engineering", "strong")
    assert "software engineering · strong" in filtered and "evaluations.jsonl" in filtered


def test_tradeoffs_render_both_scatter_plots(root):
    cost, coverage = rt.render_tradeoffs(root, cost="historical_run_wall_seconds")
    assert "Wall time" in cost and cost.count("<circle") >= 12
    assert "Requirement coverage" in coverage and "<svg" in coverage
    # Asking for the absent judge falls back to faithfulness rather than an empty chart.
    judge_cost, _ = rt.render_tradeoffs(root, quality="judge_overall_mean")
    assert "Claim faithfulness" in judge_cost


def test_pairwise_and_robustness_views(root):
    pairwise = rt.render_pairwise(root)
    assert "Head-to-head against the W2 baseline" in pairwise
    assert "3 scored of 3 · 1 unavailable" in pairwise
    robustness = rt.render_robustness(root)
    assert "Noisy input" in robustness and "Out of scope" in robustness
    assert "mismatched_jd" in robustness
    deltas = rt.render_deltas(root, "unsupported_claim_rate")
    assert "Unsupported claims" in deltas and "lower is better" in deltas


def test_missing_stages_render_messages(tmp_path):
    root = build_fixture_root(tmp_path, pairwise=False, robustness=False, bootstrap_samples=50)
    assert "Pairwise comparison not run" in rt.render_pairwise(str(root))
    assert "Robustness not run for this evaluation" in rt.render_robustness(str(root))
    assert rt.render_deltas(str(root)) == ""
    assert "pairwise — not run" in rt.render_status(str(root))


def test_no_results_at_all(tmp_path):
    for render in (rt.render_status, rt.render_leaderboard, rt.render_pairwise,
                   rt.render_robustness):
        assert "No saved evaluation found" in render(None)
    assert rt.render_pair(None, None)[3]["w1_single_pass"][0] is False


def test_output_viewer(root):
    info, resume, jd, slots = rt.render_pair(root, PAIRS[3][0], ["w2_baseline", "w3_critic"])
    assert "moderate" in info and resume.startswith("# Ravi") and "Junior Analyst" in jd
    assert slots["w2_baseline"][0] and slots["w3_critic"][0] and not slots["w1_single_pass"][0]
    assert "generation failed" in slots["w3_critic"][1]
    assert "No output" in slots["w3_critic"][2]
    assert "vs W2: unscored" in slots["w7_plan_execute"][1]


def test_leaderboard_headline_tiles(root):
    html = rt.render_leaderboard(root)
    assert html.count('class="kpi"') == 4
    assert "Most faithful" in html and "vs W2" in html and "Cheapest per output" in html


def test_tooltips_are_escaped_data_attributes():
    attrs = charts.tip('<b>0.9</b> | W1 "Single" pass')
    assert 'data-tip="&lt;b&gt;0.9&lt;/b&gt; | W1 &quot;Single&quot; pass"' in attrs
    assert 'tabindex="0"' in attrs and "textContent" in charts.TOOLTIP_JS
    assert "innerHTML" not in charts.TOOLTIP_JS
    html = charts.bar_chart([{"workflow": "w4_evidence", "value": 0.9, "label": "0.90"}])
    assert "data-tip=" in html and "<title>" not in html


def test_bar_chart_never_draws_unscored_as_zero():
    html = charts.bar_chart([
        {"workflow": "w1_single_pass", "value": None, "label": "unscored (0/4)"},
        {"workflow": "w2_baseline", "value": 0.5, "label": "50%"},
    ])
    assert "unscored (0/4)" in html
    assert "var(--wf-w1" not in html  # no bar for the unscored workflow
    assert "stroke-dasharray" in html  # the baseline bar is outlined


def test_scatter_marks_baseline_and_labels_points():
    points = [
        {"workflow": "w2_baseline", "x": 9000, "y": 0.8, "x_n": 4, "y_n": 4},
        {"workflow": "w4_evidence", "x": 21000, "y": 0.95, "x_n": 4, "y_n": 4},
    ]
    html = charts.scatter_chart(points, x_label="Tokens", y_label="Faithfulness",
                                x_kind="number", y_kind="score", x_clamp=(0, None))
    assert "rotate(45" in html and ">W2<" in html and ">W4<" in html
    assert "-" not in html.split("Tokens")[0].split(">0<")[0][-3:]  # no negative token ticks


def test_blocks_build_and_handlers_run(evaluations):
    with gr.Blocks() as demo:
        with gr.Tabs():
            handles = rt.build_results_tab(demo, evaluations_dir=evaluations)
    assert isinstance(demo, gr.Blocks)
    first = handles["initial"]()
    assert first[0]["value"] == str(dd.discover_roots(evaluations)[0].path)
    assert "28</b> scored outputs" in first[1]
    viewer = handles["viewer"](first[0]["value"], PAIRS[0][0], list(WORKFLOWS))
    assert len(viewer) == 3 + 3 * len(WORKFLOWS)


def test_blocks_build_with_no_results(tmp_path):
    with gr.Blocks() as demo:
        with gr.Tabs():
            handles = rt.build_results_tab(demo, evaluations_dir=tmp_path)
    first = handles["initial"]()
    assert first[0]["value"] is None and "No saved evaluation found" in first[1]


def test_app_mounts_results_tab():
    import app

    labels = [getattr(block, "label", None) for block in app.demo.blocks.values()]
    assert "Tailor a resume" in labels and "Results" in labels
