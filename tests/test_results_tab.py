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
    build_fixture_root(base, rubric=True, bootstrap_samples=50)
    return base


@pytest.fixture(scope="module")
def plain_root(tmp_path_factory):
    """Scored outputs without a rubric v3 run."""
    base = tmp_path_factory.mktemp("plain")
    return str(build_fixture_root(base, pairwise=False, robustness=False, bootstrap_samples=50))


SUPERSEDED_LABELS = ("Task success", "task success", "Requirement coverage", "Keyword coverage",
                     "JD similarity", "Rubric judge", "Pairwise", "vs W2:", "coverage ")


@pytest.fixture
def root(evaluations):
    return str(dd.discover_roots(evaluations)[0].path)


def test_every_workflow_has_a_distinct_colour_and_description():
    assert set(charts.WORKFLOW_COLOURS) == set(dd.WORKFLOW_ORDER) == set(dd.WORKFLOW_INFO)
    light = [colours[0] for colours in charts.WORKFLOW_COLOURS.values()]
    assert len(set(light)) == len(light)
    for name in dd.WORKFLOW_ORDER:
        assert f"--wf-{dd.short_name(name).lower()}" in charts.RESULTS_CSS


def test_explainer_shows_models_and_measured_averages(root, plain_root):
    html = rt.render_explainer(root, models={name: "base-model" for name in WORKFLOWS})
    for name in WORKFLOWS:
        assert dd.WORKFLOW_INFO[name]["title"] in html
    assert "LLM calls" in html and "good resumes" in html and "task success" not in html
    assert "<b>3/4</b><span>good resumes" in html  # W4: one resume fails the writing gate
    assert "<b>1/4</b><span>good resumes" in html  # W3: the failed generation is not good
    assert "agentic-resume-v4-1" in html  # the measured W5 model differs from the passed one
    plain = rt.render_explainer(plain_root, models={name: "base-model" for name in WORKFLOWS})
    assert "<b>unscored</b><span>good resumes" in plain


def test_explainer_without_results_still_explains(tmp_path):
    html = rt.render_explainer(str(tmp_path / "missing"))
    assert html.count('<h3><span class="code"') == 7 and "Not in this evaluation" in html


def test_leaderboard_shows_final_criteria_only(root):
    html = rt.render_leaderboard(root)
    for label in ("Good resumes", "NLI pass rate", "Claim faithfulness", "Unsupported claims",
                  "Fabrication bait", "Protected-field violations", "Rubric pass rate",
                  "Rubric v3 overall", "Quantified evidence", "Clarity", "Concision",
                  "Logical order", "Tokens / output", "Wall time (s)"):
        assert label in html, label
    for label in SUPERSEDED_LABELS:
        assert label not in html, label
    for tier in dd.METRIC_TIERS.values():
        assert tier in html
    assert "≥ 0.80 · ↑ better" in html and "≥ 4 · ↑ better" in html
    assert "at least 0.80 (truth gate)" in html and "at least 4 out of 5" in html
    assert "summaries/overall.json + rubric v3 run rubric00000000v3" in html


def test_leaderboard_shows_denominators_and_unscored(root, plain_root):
    html = rt.render_leaderboard(root)
    assert "<span>75%</span><span class=\"n\">3/4</span>" in html  # W4 good resumes
    assert "1 good · 1 failed truth · 1 unscored · 1 no resume" in html  # W3 breakdown
    assert "n=3 of 4" in html  # W6 rubric: the judge could not score one resume
    assert "2/3" in html  # W3 NLI pass rate: the failed generation is not eligible
    filtered = rt.render_leaderboard(root, "software_engineering", "strong")
    assert "software engineering · strong" in filtered and "evaluations.jsonl" in filtered
    plain = rt.render_leaderboard(plain_root)
    assert "Good resumes" in plain and "unscored (0/4)" in plain and "0/4 scored" in plain
    assert "Quantified evidence" not in plain  # no rubric v3 run: dimension columns hidden
    assert "rubric v3 run" not in plain


def test_tradeoffs_render_both_scatter_plots(root, plain_root):
    cost, gates = rt.render_tradeoffs(root, cost="historical_run_wall_seconds")
    assert "Wall time" in cost and cost.count("<circle") >= 12
    assert "Quality vs cost: good resumes vs wall time" in cost  # good-resume rate by default
    assert "Truth vs writing" in gates and "Rubric v3 overall" in gates and "<svg" in gates
    assert "Requirement coverage" not in gates
    rubric_cost, _ = rt.render_tradeoffs(root, quality="rubric_overall")
    assert "Rubric v3 overall" in rubric_cost
    # Without a rubric v3 run the plot falls back to faithfulness rather than an empty chart.
    plain_cost, plain_gates = rt.render_tradeoffs(plain_root)
    assert "Claim faithfulness" in plain_cost and "No workflow is scored on both" in plain_gates
    assert [value for _, value in rt.QUALITY_CHOICES] == [
        "good_rate", "claim_faithfulness", "rubric_overall"]


def test_robustness_view_without_pairwise(root):
    assert not hasattr(rt, "render_pairwise")
    robustness = rt.render_robustness(root)
    assert "Noisy input" in robustness and "Out of scope" in robustness
    assert "mismatched_jd" in robustness
    deltas = rt.render_deltas(root, "unsupported_claim_rate")
    assert "Unsupported claims" in deltas and "lower is better" in deltas


def test_missing_stages_render_messages(plain_root, root):
    assert "Robustness not run for this evaluation" in rt.render_robustness(plain_root)
    assert rt.render_deltas(plain_root) == ""
    status = rt.render_status(plain_root)
    assert "rubric v3 — not found" in status and "pairwise" not in status
    assert "<b>rubric v3</b> ✓ <code>rubric00000000v3</code>" in rt.render_status(root)


def test_no_results_at_all(tmp_path):
    for render in (rt.render_status, rt.render_leaderboard, rt.render_robustness):
        assert "No saved evaluation found" in render(None)
    assert rt.render_pair(None, None)[3]["w1_single_pass"][0] is False


def test_output_viewer(root, plain_root):
    info, resume, jd, slots = rt.render_pair(root, PAIRS[3][0], ["w2_baseline", "w3_critic"])
    assert "moderate" in info and resume.startswith("# Ravi") and "Junior Analyst" in jd
    assert "good resumes: <b>W4</b>" in info and "at least 0.80 (truth gate)" in info
    assert slots["w2_baseline"][0] and slots["w3_critic"][0] and not slots["w1_single_pass"][0]
    assert "✗ no resume" in slots["w3_critic"][1] and "generation failed" in slots["w3_critic"][1]
    assert "No output" in slots["w3_critic"][2]
    w4 = slots["w4_evidence"][1]  # rubric exactly at the 4.0 writing gate
    assert '<span class="chip good">✓ good resume</span>' in w4
    assert "NLI 0.9" in w4 and "✓ ≥ 0.80" in w4 and "rubric v3 4.00 ✓ ≥ 4.00" in w4
    assert ("Rubric v3 (1–5): Quantified evidence 4 · Clarity 4 · Concision 4 · Logical order 4"
            in w4)
    w1 = slots["w1_single_pass"][1]
    assert "✗ failed truth" in w1 and "✗ &lt; 0.80" in w1
    _, _, _, other = rt.render_pair(root, PAIRS[1][0])
    assert "✗ failed writing" in other["w4_evidence"][1]
    assert "rubric v3 3.75 ✗ &lt; 4.00" in other["w4_evidence"][1]
    _, _, _, unscored = rt.render_pair(root, PAIRS[2][0])
    assert '<span class="chip ">unscored</span>' in unscored["w3_critic"][1]
    assert "rubric v3 unscored" in unscored["w3_critic"][1]
    for _, header, _ in slots.values():
        for label in SUPERSEDED_LABELS:
            assert label not in header, label
    info, _, _, plain = rt.render_pair(plain_root, PAIRS[0][0])
    assert "rubric v3 not run" in plain["w4_evidence"][1] and "no rubric v3 run found" in info


def test_leaderboard_headline_tiles(root):
    html = rt.render_leaderboard(root)
    assert html.count('class="kpi"') == 4
    assert "Most good resumes" in html and "+50 pts vs W2 (25%)" in html
    assert "Most faithful" in html and "Best written (rubric v3)" in html
    assert "Cheapest per output" in html


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
    assert "pairwise" not in handles
    labels = [getattr(block, "label", None) for block in demo.blocks.values()]
    assert "3 · Robustness" in labels and "3 · Pairwise and robustness" not in labels
    quality = next(value for value in first if isinstance(value, dict)
                   and value.get("value") in dict(rt.QUALITY_CHOICES).values())
    assert quality["value"] == "good_rate"
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
