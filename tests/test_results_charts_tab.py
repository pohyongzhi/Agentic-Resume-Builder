"""Results (charts) tab: payload builder, offline page and a Blocks smoke test (nothing launched)."""
import html
import json
import re
import socket

import gradio as gr
import pytest

from eval import dashboard as dd
from eval import dashboard_charts as dc
from tests.dashboard_fixture import (
    DATASET_HASH,
    EVALUATOR_HASH,
    PAIRS,
    WORKFLOWS,
    build_fixture_root,
    build_score_rows,
)
from ui import results_charts_tab as rct


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted in a results-charts test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(dd.config, "REPORT_BOOTSTRAP_SAMPLES", 50)
    dd.clear_cache()
    dc.clear_cache()


@pytest.fixture(scope="module")
def evaluations(tmp_path_factory):
    base = tmp_path_factory.mktemp("evaluations")
    build_fixture_root(base, pairwise=False, robustness=False, bootstrap_samples=50)
    return base


@pytest.fixture
def root(evaluations):
    return dd.discover_roots(evaluations)[0].path


@pytest.fixture
def payload(root):
    return dc.build_payload(root)


def _records(payload):
    return [dict(zip(payload["cols"], row)) for row in payload["rows"]]


def _write_root(base, rows):
    root = base / DATASET_HASH / "scope" / EVALUATOR_HASH
    root.mkdir(parents=True)
    (root / "evaluations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return root


def test_payload_follows_registry_order_and_counts_every_output(payload):
    from workflows import WORKFLOWS as REGISTRY

    assert [w["key"] for w in payload["workflows"]] == [w for w in REGISTRY if w in WORKFLOWS]
    assert [w["short"] for w in payload["workflows"]] == [f"W{i}" for i in range(1, 8)]
    assert payload["baseline"] == "W2"
    assert len(payload["rows"]) == len(WORKFLOWS) * len(PAIRS)
    assert payload["run"]["outputs"] == 28 and payload["run"]["pairs"] == len(PAIRS)
    assert all(len(row) == len(dc.COLUMNS) for row in payload["rows"])


def test_outcomes_follow_the_severity_order():
    assert dc.outcome({"error": "X", "task_success": True}) == "rejected"
    assert dc.outcome({"task_success": True}) == "pass"
    assert dc.outcome({"task_success": False, "task_failure_reasons": [
        "word_count_out_of_range", "missing_required_sections", "fabrication_bait_violation",
    ]}) == "unfaithful"
    assert dc.outcome({"task_success": False, "task_failure_reasons": [
        "word_count_out_of_range", "missing_required_sections"]}) == "sections"
    assert dc.outcome({"task_success": False,
                       "task_failure_reasons": ["word_count_out_of_range"]}) == "length"
    assert dc.outcome({"task_success": None}) == "other"


def test_payload_outcomes_match_the_fixture(payload):
    records = _records(payload)
    w3 = [r for r in records if r["wf"] == "W3"]
    # The fixture's W3 fails generation on the fourth pair: counted once, as rejected, no metrics.
    rejected = [r for r in w3 if r["outcome"] == "rejected"]
    assert len(rejected) == 1
    assert all(rejected[0][key] is None for key in ("faith", "words", "wall", "bait", "edu"))
    summary = {s["short"]: s for s in payload["summary"]}
    assert summary["W3"]["total"] == 4 and summary["W3"]["done"] == 3
    assert summary["W3"]["rejected"] == 1
    assert {r["outcome"] for r in records} <= set(dc.OUTCOMES)
    # Pairs 0 with quality < 0.6 carry a protected-field violation; W7 pair 1 carries bait.
    w1_first = next(r for r in records if r["wf"] == "W1")
    assert w1_first["outcome"] == "unfaithful"
    assert sum(r["bait"] == 1 for r in records if r["wf"] == "W7") == 1


def test_summary_has_bootstrap_intervals(payload):
    for item in payload["summary"]:
        for key in ("faith", "bait", "edu", "length_ok", "task"):
            mean, low, high = item[key]
            assert mean is not None and low <= mean <= high, (item["short"], key)
        assert item["wall_median"] is not None and item["calls_median"] is not None
    # The interval is reproducible: fixed seed, 1,000 resamples.
    values = [0.5, 0.7, 0.9, 1.0]
    assert dc.boot_ci(values) == dc.boot_ci(values)
    assert dc.boot_ci([]) == [None, None, None]


def test_unscored_values_stay_null(tmp_path):
    rows = build_score_rows()
    target = next(r for r in rows if r["workflow"] == "w5_fine_tuned" and not r.get("error"))
    target.update(claim_faithfulness=None, fabrication_bait_violation=None, task_success=None,
                  n_claims=None, supported_claims=None)
    for row in rows:
        if row["workflow"] == "w6_best_of_n":
            row["claim_faithfulness"] = None
    payload = dc.build_payload(_write_root(tmp_path, rows))
    record = next(r for r in _records(payload)
                  if r["wf"] == "W5" and r["faith"] is None and r["outcome"] != "rejected")
    assert record["bait"] is None and record["unsupported"] is None
    summary = {s["short"]: s for s in payload["summary"]}
    assert summary["W5"]["faith_n"] == 3 and summary["W5"]["bait_n"] == 3
    assert summary["W6"]["faith"] == [None, None, None] and summary["W6"]["faith_n"] == 0
    assert summary["W6"]["done"] == 4  # outputs exist; only the score is missing


def test_absent_and_zero_output_workflows(tmp_path):
    rows = [r for r in build_score_rows() if r["workflow"] not in {"w4_evidence", "w6_best_of_n"}]
    for row in rows:
        if row["workflow"] == "w7_plan_execute":
            row.update(error="W7_FAILED: x", resume_md="", claim_faithfulness=None, words=0)
    payload = dc.build_payload(_write_root(tmp_path, rows))
    assert [w["short"] for w in payload["workflows"]] == ["W1", "W2", "W3", "W5", "W7"]
    w7 = next(s for s in payload["summary"] if s["short"] == "W7")
    assert w7["done"] == 0 and w7["rejected"] == 4
    assert w7["faith"] == [None, None, None] and w7["wall_median"] is None
    assert all(isinstance(text, str) and text for text in payload["headlines"].values())
    page = rct.render_page(payload)
    assert '"W7"' in page and "W4" not in json.dumps(payload["workflows"])


def test_records_fallback_when_no_index(tmp_path):
    root = tmp_path / DATASET_HASH / "scope" / EVALUATOR_HASH
    for row in build_score_rows(workflows=("w1_single_pass", "w2_baseline")):
        path = root / "records" / row["workflow"] / row["pair_id"] / "repeat_0.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row), encoding="utf-8")
    payload = dc.build_payload(root)
    assert [w["short"] for w in payload["workflows"]] == ["W1", "W2"]
    assert len(payload["rows"]) == 2 * len(PAIRS)


def test_headlines_are_computed_from_the_data(payload):
    headlines = payload["headlines"]
    assert set(headlines) == {"good", "quadrant", "faith", "bait", "rubric", "dims", "heat", "fit",
                              "time", "calls", "tokens", "value"}
    # The fixture has no rubric v3 run, so the good-resume headline says so instead of ranking.
    assert payload["has_rubric"] is False and "no rubric v3 scores" in headlines["good"]
    # W1 has the lowest fixture quality (0.55) and W3 the highest cost.
    assert headlines["faith"].startswith("W1 is the least faithful")
    assert headlines["calls"].startswith("W3 makes the most model calls")
    assert headlines["bait"].startswith("W7 invents a qualification")
    caveat = payload["text"]["caveat"]
    assert DATASET_HASH[:12] in caveat and EVALUATOR_HASH in caveat and "28 outputs" in caveat
    assert "PR" not in caveat


def test_empty_or_missing_root(tmp_path):
    assert dc.build_payload(tmp_path / "missing") is None
    empty = tmp_path / "d" / "g" / "e"
    empty.mkdir(parents=True)
    (empty / "evaluations.jsonl").write_text("", encoding="utf-8")
    assert dc.build_payload(empty) is None
    assert "No saved evaluation found" in rct.render_frame(None)
    assert "no saved outputs" in rct.render_frame(str(empty))


NETWORK_REF = re.compile(
    r"<(?:script|link|img|iframe)\b[^>]*\b(?:src|href)\s*=\s*[\"']?\s*(?:https?:)?//"
    r"|@import\s+(?:url\()?[\"']?\s*(?:https?:)?//"
    r"|url\(\s*[\"']?\s*(?:https?:)?//",
    re.I,
)


def test_page_is_offline_and_inlines_d3(payload):
    page = rct.render_page(payload)
    assert "https://d3js.org v7.8.5" in page  # d3's licence header, kept
    assert not NETWORK_REF.search(page)
    assert "fonts.googleapis" not in page and "cdnjs" not in page
    assert not re.search(r"__[A-Z0-9_]+__", page)  # every placeholder filled
    data = re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.S)
    assert json.loads(data.group(1))["run"]["outputs"] == 28


def test_blocks_build_and_srcdoc_is_offline(evaluations, tmp_path):
    with gr.Blocks() as demo:
        with gr.Tabs():
            handles = rct.build_results_charts_tab(demo, evaluations_dir=evaluations)
    dropdown, frame = handles["initial"]()
    assert dropdown["value"] == str(dd.discover_roots(evaluations)[0].path)
    assert frame.startswith('<iframe class="arb-charts-frame"') and "<script" not in frame
    srcdoc = html.unescape(re.search(r'srcdoc="([^"]*)"', frame).group(1))
    assert "<script>" in srcdoc and "d3js.org v7.8.5" in srcdoc
    assert not NETWORK_REF.search(srcdoc)
    with gr.Blocks() as empty_demo:
        with gr.Tabs():
            empty = rct.build_results_charts_tab(empty_demo, evaluations_dir=tmp_path)
    dropdown, frame = empty["initial"]()
    assert dropdown["value"] is None and "No saved evaluation found" in frame


def test_app_mounts_both_results_tabs():
    import app

    labels = [getattr(block, "label", None) for block in app.demo.blocks.values()]
    assert {"Tailor a resume", "Results", "Results (charts)"} <= set(labels)
    assert labels.index("Results") < labels.index("Results (charts)")


def _write_rubric(root, scores, gates=(0.8, 4.0)):
    """A rubric v3 run next to ``root``: {(workflow, pair_id): overall or None}."""
    run = root.parent.parent / "quality-rubric" / "rubricid"
    (run / "records").mkdir(parents=True)
    (run / "identity.json").write_text(json.dumps({"version": dc.RUBRIC_VERSION, "provider": "deepinfra",
                                                   "model": "openai/gpt-oss-120b", "reasoning_effort": "medium"}))
    (run / "summary.json").write_text(json.dumps({"nli_threshold": gates[0], "rubric_threshold": gates[1]}))
    for (workflow, pair_id), overall in scores.items():
        path = run / "records" / workflow / f"{pair_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        dims = None if overall is None else dict.fromkeys(dc.RUBRIC_DIMENSIONS, overall)
        path.write_text(json.dumps({"workflow": workflow, "pair_id": pair_id, "overall": overall,
                                    "status": "scored" if overall is not None else "unscored",
                                    "scores": dims}))


def test_verdict_checks_truth_before_writing():
    gates = {"nli": 0.8, "rubric": 4.0}
    ok = {"task_success": True, "claim_faithfulness": 0.9}
    assert dc.verdict({"error": "X"}, {"overall": 5}, gates) == "rejected"
    assert dc.verdict({**ok, "claim_faithfulness": 0.7}, {"overall": 5}, gates) == "nli_fail"
    assert dc.verdict({**ok, "claim_faithfulness": 0.7}, {"overall": 2}, gates) == "nli_fail"
    assert dc.verdict(ok, {"overall": 3.75}, gates) == "rubric_fail"
    assert dc.verdict(ok, {"overall": 4.0}, gates) == "good"
    assert dc.verdict(ok, {"overall": None}, gates) == "unscored"
    assert dc.verdict(ok, None, gates) == "unscored"
    assert dc.verdict({**ok, "claim_faithfulness": None}, {"overall": 5}, gates) == "unscored"


def test_payload_joins_rubric_scores_and_counts_good_resumes(tmp_path):
    rows = build_score_rows()
    root = _write_root(tmp_path, rows)
    scores = {(r["workflow"], r["pair_id"]): 4.5 for r in rows}
    # The fixture's W4 resumes all pass the truth gate, so the writing gate decides.
    w4 = [r["pair_id"] for r in rows if r["workflow"] == "w4_evidence"]
    scores[("w4_evidence", w4[0])] = 3.0     # fails the writing gate
    scores[("w4_evidence", w4[1])] = None    # judge left it unscored
    _write_rubric(root, scores)
    payload = dc.build_payload(root)
    assert payload["has_rubric"] is True
    assert payload["thresholds"]["faith"] == 0.8 and payload["thresholds"]["rubric"] == 4.0
    summary = {s["short"]: s for s in payload["summary"]}
    w4 = summary["W4"]["verdicts"]
    assert w4 == {"good": 2, "rubric_fail": 1, "unscored": 1, "nli_fail": 0, "rejected": 0}
    assert summary["W4"]["rubric_scored"] == 3 and summary["W4"]["rubric_pass"] == 2
    assert summary["W1"]["verdicts"]["nli_fail"] == 4   # every fixture W1 resume is below 0.8
    assert summary["W3"]["verdicts"]["rejected"] == 1   # the fixture's failed generation
    records = _records(payload)
    assert all(r["verdict"] in dc.VERDICTS for r in records)
    assert all(r["verdict"] != "good" or (r["faith"] >= 0.8 and r["rubric"] >= 4.0) for r in records)
    good = {code: s["verdicts"]["good"] for code, s in summary.items()}
    top = [code for code, n in good.items() if n == max(good.values())]
    assert payload["headlines"]["good"].startswith(dc.join_codes(top))
    assert "openai/gpt-oss-120b via deepinfra" in payload["text"]["footer"]
    page = rct.render_page(payload)
    assert not re.search(r"__[A-Z0-9_]+__", page)
