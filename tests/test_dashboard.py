"""Results-dashboard loader and aggregations, on synthetic roots written by the real report code."""
import json
import os
import shutil
import socket

import pytest

from eval import dashboard as dd
from tests.dashboard_fixture import PAIRS, WORKFLOWS
from tests.dashboard_fixture import build_fixture_root as _build


def build_fixture_root(base, **kwargs):
    return _build(base, bootstrap_samples=50, **kwargs)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted in a dashboard test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(dd.config, "REPORT_BOOTSTRAP_SAMPLES", 50)
    dd.clear_cache()


@pytest.fixture(scope="module")
def full_root(tmp_path_factory):
    return build_fixture_root(tmp_path_factory.mktemp("evaluations"))


@pytest.fixture
def bundle(full_root):
    return dd.load_bundle(full_root)


def test_discover_roots_newest_first_with_stages(tmp_path):
    older = build_fixture_root(tmp_path, pairwise=False, robustness=False, evaluator_hash="a" * 16)
    newer = build_fixture_root(tmp_path, evaluator_hash="b" * 16)
    for path in [older / "evaluations.jsonl", *(older / "summaries").glob("*.json")]:
        os.utime(path, (1_000_000, 1_000_000))
    (tmp_path / "x" / "y" / "not-a-root").mkdir(parents=True)

    roots = dd.discover_roots(tmp_path)

    assert [root.path for root in roots] == [newer, older]
    assert roots[0].stages == ("score", "pairwise", "robustness")
    assert roots[1].stages == ("score",)
    assert roots[0].outputs == len(WORKFLOWS) * len(PAIRS)
    assert roots[0].workflows == WORKFLOWS
    assert "W1, W2, W3" in roots[0].label and "bbbbbbbb" in roots[0].label


def test_discover_roots_missing_directory(tmp_path):
    assert dd.discover_roots(tmp_path / "absent") == []


def test_default_dir_can_be_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("ARB_EVALUATIONS_DIR", str(tmp_path))
    assert dd.default_evaluations_dir() == tmp_path


def test_read_jsonl_skips_blank_and_truncated_lines(tmp_path):
    path = tmp_path / "evaluations.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n{"a": 3', encoding="utf-8")
    assert dd.read_jsonl(path) == [{"a": 1}, {"a": 2}]
    assert dd.read_jsonl(tmp_path / "missing.jsonl") == []


def test_load_bundle_reads_every_stage(bundle):
    assert len(bundle.rows) == len(WORKFLOWS) * len(PAIRS)
    assert bundle.workflows == list(WORKFLOWS)
    assert {"overall", "by_category", "by_fit_level", "pairwise", "robustness_families",
            "robustness_deltas", "robustness_out_of_scope"} <= set(bundle.summaries)
    assert len(bundle.pairwise_rows) == (len(WORKFLOWS) - 1) * len(PAIRS)
    assert bundle.has_pairwise and bundle.has_robustness


def test_load_bundle_cache_refreshes_when_files_change(tmp_path):
    root = build_fixture_root(tmp_path, pairwise=False, robustness=False)
    first = dd.load_bundle(root)
    assert dd.load_bundle(root) is first
    index = root / "evaluations.jsonl"
    rows = dd.read_jsonl(index)[:3]
    index.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    os.utime(index, (first_time := index.stat().st_mtime + 5, first_time))
    assert len(dd.load_bundle(root).rows) == 3


def test_leaderboard_matches_overall_summary(bundle):
    board, source = dd.leaderboard(bundle)
    assert source == "summaries/overall.json"
    assert [row["workflow"] for row in board] == list(WORKFLOWS)
    summary = {
        (row["workflow"], row["metric"]): row for row in bundle.summaries["overall"]
    }
    for row in board:
        for spec in dd.LEADERBOARD_METRICS:
            cell = row["cells"][spec.key]
            expected = summary[(row["workflow"], spec.key)]
            if expected["scored"]:
                assert cell["value"] == pytest.approx(expected["mean"])
            assert (cell["eligible"], cell["scored"]) == (expected["eligible"], expected["scored"])


def test_unscored_is_never_zero(bundle):
    board, _ = dd.leaderboard(bundle)
    cells = {row["workflow"]: row["cells"] for row in board}
    # The fixture skips the rubric judge: eligible outputs exist but none is scored.
    judge = cells["w2_baseline"]["judge_overall_mean"]
    assert judge["value"] is None and judge["status"] == "unscored" and judge["eligible"] == 4
    assert not dd.metric_is_present(board, "judge_overall_mean")
    # W6 has one coverage evaluator error: 3 scored of 4 eligible, mean over the 3 only.
    coverage = cells["w6_best_of_n"]["requirement_coverage"]
    assert (coverage["scored"], coverage["eligible"], coverage["unscored"]) == (3, 4, 1)
    assert coverage["value"] == pytest.approx(1.0)
    # W1's pair without supportable requirements is excluded from eligibility.
    assert cells["w1_single_pass"]["requirement_coverage"]["eligible"] == 3
    # W3's generation failure leaves its quality metrics at 3 eligible.
    assert cells["w3_critic"]["claim_faithfulness"]["eligible"] == 3
    assert cells["w3_critic"]["task_success"]["eligible"] == 4


def test_filters_use_stratified_summaries_or_rebuild(bundle):
    _, source = dd.leaderboard(bundle, category="software_engineering")
    assert source == "summaries/by_category.json"
    _, source = dd.leaderboard(bundle, fit_level="strong")
    assert source == "summaries/by_fit_level.json"
    board, source = dd.leaderboard(bundle, "business_technology", "strong")
    assert source == "evaluations.jsonl"
    assert all(row["cells"]["task_success"]["eligible"] == 1 for row in board)
    assert dd.filter_options(bundle) == (
        ["business_technology", "software_engineering"], ["strong", "moderate", "weak"],
    )


def test_rebuilt_summaries_match_written_ones(tmp_path, bundle):
    root = build_fixture_root(tmp_path, pairwise=False, robustness=False, summaries=False)
    rebuilt, source = dd.leaderboard(dd.load_bundle(root))
    written, _ = dd.leaderboard(bundle)
    assert source == "evaluations.jsonl"
    for left, right in zip(rebuilt, written):
        for metric in ("task_success", "claim_faithfulness", "requirement_coverage"):
            assert left["cells"][metric]["value"] == pytest.approx(right["cells"][metric]["value"])
            assert left["cells"][metric]["scored"] == right["cells"][metric]["scored"]


def test_tradeoff_points_skip_unscored(bundle):
    board, _ = dd.leaderboard(bundle)
    points = dd.tradeoff_points(board, "total_tokens", "claim_faithfulness")
    assert {point["workflow"] for point in points} == set(WORKFLOWS)
    assert dd.tradeoff_points(board, "total_tokens", "judge_overall_mean") == []


def test_headline_findings(bundle):
    board, _ = dd.leaderboard(bundle)
    findings = {item["metric"]: item for item in dd.headline_findings(board)}
    assert findings["claim_faithfulness"]["workflow"] == "w4_evidence"
    assert findings["claim_faithfulness"]["baseline"] == pytest.approx(
        next(r for r in board if r["workflow"] == "w2_baseline")["cells"]["claim_faithfulness"]["value"]
    )
    assert findings["total_tokens"]["workflow"] in {"w1_single_pass", "w5_fine_tuned"}
    coverage = findings["requirement_coverage"]  # four workflows cover every requirement
    assert coverage["workflow"] == "w2_baseline"
    assert coverage["ties"] == ["w3_critic", "w4_evidence", "w6_best_of_n"]
    empty = [{"workflow": "w1_single_pass",
              "cells": {m: {"value": None} for m in dd.METRIC_SPECS}}]
    assert all(item["workflow"] is None for item in dd.headline_findings(empty))


def test_workflow_profiles(bundle):
    profiles = dd.workflow_profiles(bundle)
    w3 = profiles["w3_critic"]
    assert w3["outputs"] == 4 and w3["generation_failures"] == 1
    assert w3["llm_calls"] == pytest.approx(9.0)
    assert (w3["task_success"], w3["task_scored"]) == (2, 3)
    assert profiles["w5_fine_tuned"]["models"] == ["agentic-resume-v4-1:q4_K_M"]


def test_pairwise_table_denominators(bundle):
    rows = {row["challenger_workflow"]: row for row in dd.pairwise_table(bundle)}
    assert list(rows) == [w for w in WORKFLOWS if w != "w2_baseline"]
    w3 = rows["w3_critic"]  # one order-dependent pair, one missing generation
    assert (w3["scored"], w3["inconsistent"], w3["challenger_wins"]) == (3, 1, 2)
    assert w3["generation_unavailable"] == 1
    w7 = rows["w7_plan_execute"]  # one judge failure stays unscored
    assert (w7["eligible"], w7["scored"], w7["unscored"], w7["failed"]) == (4, 3, 1, 1)
    assert rows["w6_best_of_n"]["ties"] == 1
    assert isinstance(w3["dimension_position_swap_consistency"], dict)


def test_pairwise_table_rebuilt_from_index(tmp_path, bundle):
    root = build_fixture_root(tmp_path, robustness=False)
    shutil.rmtree(root / "summaries")
    rebuilt = dd.pairwise_table(dd.load_bundle(root))
    written = dd.pairwise_table(bundle)
    assert [(r["challenger_workflow"], r["scored"], r["challenger_wins"]) for r in rebuilt] == [
        (r["challenger_workflow"], r["scored"], r["challenger_wins"]) for r in written
    ]


def test_pairwise_outcome_labels():
    assert dd.pairwise_outcome(None) == "not judged"
    assert dd.pairwise_outcome({"status": "generation_unavailable"}) == "unavailable"
    assert dd.pairwise_outcome({"status": "unscored"}) == "unscored"
    assert dd.pairwise_outcome({"status": "scored", "position_swap_consistent": False}) == (
        "inconsistent"
    )
    for winner, label in (("challenger", "win"), ("baseline", "loss"), ("tie", "tie")):
        assert dd.pairwise_outcome({
            "status": "scored", "position_swap_consistent": True, "consensus_winner": winner,
        }) == label


def test_robustness_family_panels(bundle):
    panels = {panel["family"]: panel for panel in dd.robustness_family_panels(bundle)}
    assert list(panels) == ["noisy", "format", "out_of_scope"]
    noisy = {item["workflow"]: item for item in panels["noisy"]["items"]}
    # W2's failed pdf_extraction run is excluded from stability scoring, not counted as unstable.
    assert noisy["w2_baseline"]["runs"] == 6 and noisy["w2_baseline"]["denominator"] == 5
    oos = {item["workflow"]: item for item in panels["out_of_scope"]["items"]}
    assert "1 crashed" in oos["w1_single_pass"]["note"]
    assert oos["w4_evidence"]["value"] == 1.0


def test_robustness_tables_rebuilt_without_summaries(tmp_path, bundle):
    root = build_fixture_root(tmp_path, pairwise=False)
    shutil.rmtree(root / "summaries")
    rebuilt = dd.robustness_family_panels(dd.load_bundle(root))
    written = dd.robustness_family_panels(bundle)
    assert [[(i["workflow"], i["value"], i["denominator"]) for i in p["items"]] for p in rebuilt] == [
        [(i["workflow"], i["value"], i["denominator"]) for i in p["items"]] for p in written
    ]


def test_robustness_delta_matrix(bundle):
    matrix = dd.robustness_delta_matrix(bundle, "faithfulness")
    families = [family for family, _ in matrix["variants"]]
    assert families == sorted(families, key=["noisy", "format"].index)
    assert len(matrix["variants"]) == 7
    assert matrix["direction"] == "higher_is_better"
    cell = matrix["cells"][("w2_baseline", "pdf_extraction")]
    assert (cell["eligible"], cell["scored"], cell["failed"]) == (2, 1, 1)
    assert "faithfulness" in dd.robustness_delta_metrics(bundle)
    assert dd.robustness_delta_matrix(bundle, "unsupported_claim_rate")["direction"] == (
        "lower_is_better"
    )
    assert {row["variant"] for row in dd.out_of_scope_variant_rows(bundle)} == {
        "mismatched_jd", "degenerate_jd",
    }


def test_missing_stages_are_reported_not_invented(tmp_path):
    root = build_fixture_root(tmp_path, pairwise=False, robustness=False)
    bundle = dd.load_bundle(root)
    assert not bundle.has_pairwise and not bundle.has_robustness
    assert dd.pairwise_table(bundle) == []
    assert all(not panel["items"] for panel in dd.robustness_family_panels(bundle))
    assert dd.robustness_delta_matrix(bundle)["variants"] == []
    view = dd.pair_view(bundle, PAIRS[0][0])
    assert all(output["pairwise"] in (None, "baseline") for output in view["outputs"].values())


def test_pair_choices_and_view(bundle):
    choices = dd.pair_choices(bundle)
    assert [value for _, value in choices] == sorted(pair[0] for pair in PAIRS)
    assert "business technology · strong fit · 7 outputs" in choices[0][0]

    view = dd.pair_view(bundle, PAIRS[3][0])
    assert view["fit_level"] == "moderate" and view["original_resume"].startswith("# Ravi")
    assert list(view["outputs"]) == list(WORKFLOWS)
    w3 = view["outputs"]["w3_critic"]
    assert w3["error"] and w3["resume_md"] == "" and w3["pairwise"] == "unavailable"
    assert view["outputs"]["w7_plan_execute"]["pairwise"] == "unscored"
    assert view["outputs"]["w2_baseline"]["pairwise"] == "baseline"
    assert dd.pair_view(bundle, PAIRS[1][0])["outputs"]["w3_critic"]["pairwise"] == "inconsistent"
    assert dd.pair_view(bundle, "unknown") is None
