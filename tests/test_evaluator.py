import json

import evaluate as offline_evaluation
import evaluate_results
from eval.judge import JudgeConfig
from eval.operational import evaluate_operational
from eval.robustness import perturb_pair, select_stratified_pairs, summarise_robustness


def _resume():
    body = " ".join(["supported evidence"] * 30)
    return (
        f"# Alex\n\n## Summary\n\n{body}\n\n"
        f"## Experience\n\n- {body}\n\n"
        f"## Education\n\n{body}\n\n"
        f"## Skills\n\n{body}\n"
    )


def _totals(llm=1, tools=0):
    return {
        "llm_calls": llm,
        "tool_calls": tools,
        "input_tokens": 100,
        "output_tokens": 50,
        "seconds": 2.5,
    }


def test_task_success_and_cost_metrics_for_toolless_workflow():
    output = {"resume_md": _resume(), "finished": True, "hit_budget": False}
    result = evaluate_operational(
        "w1_single_pass",
        {"split": "test"},
        output,
        _totals(),
        [],
        {
            "faithfulness": 0.9,
            "sim_after": 0.7,
            "kw_after": 0.8,
            "protected_field_violation": False,
            "fabrication_bait_violation": False,
        },
    )
    assert result["task_success"] is True
    assert result["tool_applicable"] is False
    assert result["tool_call_accuracy"] is None
    assert result["step_efficiency"] == 1.0
    assert result["total_tokens"] == 150
    assert result["tokens_per_successful_task"] == 150


def test_task_success_unscored_without_faithfulness():
    result = evaluate_operational(
        "w1_single_pass",
        {"split": "test"},
        {"resume_md": _resume(), "finished": True},
        _totals(),
        [],
        {"sim_after": 0.7, "kw_after": 0.8},
    )
    assert result["task_success"] is None
    assert "faithfulness_not_scored" in result["task_failure_reasons"]
    assert "protected_fields_not_scored" in result["task_failure_reasons"]
    assert "fabrication_bait_not_scored" in result["task_failure_reasons"]


def test_task_success_fails_each_deterministic_faithfulness_gate():
    output = {"resume_md": _resume(), "finished": True, "hit_budget": False}
    for field, reason in (
        ("protected_field_violation", "protected_field_violation"),
        ("fabrication_bait_violation", "fabrication_bait_violation"),
    ):
        scores = {
            "faithfulness": 1.0,
            "protected_field_violation": False,
            "fabrication_bait_violation": False,
        }
        scores[field] = True
        result = evaluate_operational(
            "w1_single_pass", {"split": "test"}, output, _totals(), [], scores,
        )
        assert result["task_success"] is False
        assert reason in result["task_failure_reasons"]


def test_offline_record_includes_deterministic_faithfulness_components(
    monkeypatch, tmp_path,
):
    source = _resume()
    tailored = source + "\nPython\n"
    pair = {
        "split": "test",
        "resume": source,
        "jd": "Need Python",
        "keywords": [{"keyword": "Python", "synonyms": []}],
    }
    monkeypatch.setattr(offline_evaluation, "_quality", lambda *_: {})
    record = offline_evaluation._evaluate_output(
        {"workflow": "w1_single_pass", "error": None},
        pair,
        tailored,
        tmp_path,
        JudgeConfig(),
        skip_nli=True,
        skip_judge=True,
        trace=[],
        totals=_totals(),
    )
    assert record["protected_field_violation"] is False
    assert record["protected_field_evaluator_version"] == "protected-fields-v1"
    assert record["fabrication_bait_violation"] is True
    assert record["fabrication_bait_evaluator_version"] == "keyword-bait-v1"
    assert record["fabrication_bait_matches"][0]["keyword"] == "Python"


def test_offline_record_reuses_complete_saved_quality_metrics(monkeypatch, tmp_path):
    saved_quality = {
        "words": 250,
        "sim_before": 0.4,
        "sim_after": 0.5,
        "sim_gain": 0.1,
        "kw_before": 0.25,
        "kw_after": 0.5,
        "kw_missing": ["AWS"],
    }
    monkeypatch.setattr(
        offline_evaluation,
        "_quality",
        lambda *_: (_ for _ in ()).throw(AssertionError("must reuse saved metrics")),
    )
    record = offline_evaluation._evaluate_output(
        {
            "workflow": "w1_single_pass",
            "error": None,
            **saved_quality,
        },
        {"split": "test", "resume": _resume(), "jd": "", "keywords": []},
        _resume(),
        tmp_path,
        JudgeConfig(),
        skip_nli=True,
        skip_judge=True,
        trace=[],
        totals=_totals(),
    )
    assert all(record[field] == value for field, value in saved_quality.items())
    assert record["evaluation_failures"] == {}


def test_tool_sequence_accuracy_and_recovery():
    trace = [
        {"type": "tool", "name": "validate_resume", "ok": True},
        {"type": "tool", "name": "render_resume", "ok": False},
        {"type": "tool", "name": "render_resume", "ok": True},
    ]
    result = evaluate_operational(
        "w2_baseline",
        {"split": "test"},
        {"resume_md": _resume(), "finished": True},
        _totals(llm=3, tools=3),
        trace,
        {"faithfulness": 0.9},
    )
    assert result["tool_sequence_valid"] is True
    assert result["tool_call_accuracy"] == 0.6667
    assert result["tool_recovery_rate"] == 1.0


def test_robustness_variants_and_summary():
    pair = {
        "resume": "# A\n\n## Summary\n\nText\n\n## Skills\n\nPython\n",
        "jd": "# Role\n\n## Responsibilities\n\n- Build\n\n## Requirements\n\n- Python\n",
    }
    formatted_resume, _ = perturb_pair(pair, "formatting")
    reordered_resume, _ = perturb_pair(pair, "reordered_sections")
    assert "* Build" not in formatted_resume  # only the JD contains this bullet
    assert "## SUMMARY" in formatted_resume
    assert reordered_resume.index("## Skills") < reordered_resume.index("## Summary")

    rows = [
        {"workflow": "w1", "base_pair_id": "p1", "variant": "original",
         "sim_after": 0.8, "kw_after": 0.8, "faithfulness": 0.9, "error": None},
        {"workflow": "w1", "base_pair_id": "p1", "variant": "formatting",
         "sim_after": 0.7, "kw_after": 0.8, "faithfulness": 0.8, "error": None},
        {"workflow": "w1", "base_pair_id": "p1", "variant": "reordered_sections",
         "sim_after": 0.6, "kw_after": 0.7, "faithfulness": 0.7, "error": "failed"},
    ]
    summary = summarise_robustness(rows)[0]
    assert summary["robust_success_rate"] == 0.6667
    assert summary["faithfulness_worst_degradation"] == -0.2
    assert summary["sim_after_mean_delta"] == -0.15


def test_stratified_robustness_selection_is_balanced_and_deterministic():
    pairs = []
    for category in "abcde":
        for fit, count in (("strong", 8), ("moderate", 8), ("weak", 4)):
            for index in range(count):
                pairs.append({
                    "pair_id": f"{category}_{fit}_{index}",
                    "category": category,
                    "fit_level": fit,
                    "role_family": f"role_{index % 4}",
                })
    selected = select_stratified_pairs(pairs, seed=42)
    assert selected == select_stratified_pairs(list(reversed(pairs)), seed=42)
    assert len(selected) == 20
    assert {category: sum(p["category"] == category for p in selected) for category in "abcde"} == {
        category: 4 for category in "abcde"
    }
    assert {fit: sum(p["fit_level"] == fit for p in selected) for fit in ("strong", "moderate", "weak")} == {
        "strong": 8, "moderate": 8, "weak": 4,
    }


def test_offline_evaluator_reads_saved_resume_and_trace(monkeypatch, tmp_path):
    resume_path = tmp_path / "resume.md"
    trace_path = tmp_path / "trace.json"
    metrics_path = tmp_path / "metrics.json"
    results_path = tmp_path / "results.jsonl"
    output_path = tmp_path / "evaluated.jsonl"
    resume_path.write_text(_resume())
    trace_path.write_text("[]")
    row = {
        **_totals(),
        "workflow": "w1_single_pass",
        "pair_id": "p1",
        "resume_path": str(resume_path),
        "trace_path": str(trace_path),
        "metrics_path": str(metrics_path),
        "faithfulness": 0.9,
        "protected_field_violation": False,
        "fabrication_bait_violation": False,
        "sim_after": 0.7,
        "kw_after": 0.8,
        "error": None,
    }
    results_path.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(
        evaluate_results,
        "load_pairs",
        lambda: [{"pair_id": "p1", "split": "test", "resume": "", "jd": "", "keywords": []}],
    )

    evaluated = evaluate_results.evaluate(results_path, output_path)

    assert evaluated[0]["task_success"] is True
    assert output_path.exists()
    assert not metrics_path.exists()  # immutable generation metrics are not overwritten
