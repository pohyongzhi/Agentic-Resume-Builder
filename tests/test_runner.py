import json
import sys

import config
import run_all
import summarise
from common.llm import Tracker


def test_config_hash_changes_with_shared_prompt(monkeypatch):
    first = run_all.config_hash()
    monkeypatch.setattr(run_all.prompts, "CREATOR_SYSTEM", run_all.prompts.CREATOR_SYSTEM + "\nchanged")
    assert run_all.config_hash() != first


def test_config_hash_changes_with_w6_candidate_count(monkeypatch):
    first = run_all.config_hash()
    monkeypatch.setattr(config, "W6_NUM_CANDIDATES", config.W6_NUM_CANDIDATES + 1)
    assert run_all.config_hash() != first


def test_config_hash_changes_with_fine_tuned_model(monkeypatch):
    first = run_all.config_hash()
    monkeypatch.setattr(config, "FINE_TUNED_MODEL", config.FINE_TUNED_MODEL + "-changed")
    assert run_all.config_hash() != first


def test_config_hash_changes_with_generation_source_but_not_evaluator_settings(
    monkeypatch, tmp_path,
):
    source = tmp_path / "writer.py"
    source.write_text("PROMPT = 'first'\n")
    monkeypatch.setattr(run_all, "_generation_source_paths", lambda workflow=None: [source])
    first = run_all.config_hash("w1_single_pass")
    source.write_text("PROMPT = 'second'\n")
    assert run_all.config_hash("w1_single_pass") != first

    second = run_all.config_hash("w1_single_pass")
    monkeypatch.setattr(config, "NLI_EVALUATOR_VERSION", "evaluator-only-change")
    assert run_all.config_hash("w1_single_pass") == second


def test_done_keys_includes_model_config_and_dataset_hash(monkeypatch, tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps({
        "workflow": "w2_baseline", "pair_id": "p1", "repeat": 0,
        "model": "model-a", "config_hash": "abc", "dataset_hash": "dataset-1",
    }) + "\n")
    monkeypatch.setattr(config, "RESULTS_FILE", str(path))
    assert run_all.done_keys() == {
        ("w2_baseline", "p1", 0, "model-a", "abc", "dataset-1")
    }


def test_run_directory_is_deterministic(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESULTS_FILE", str(tmp_path / "results.jsonl"))
    manifest = {"version": "1.0.0"}
    expected = (tmp_path / "runs" / "1.0.0" / "cfg" / "w1_single_pass"
                / "resume_1__jd_1" / "repeat_0")
    assert run_all._run_dir(manifest, "cfg", "w1_single_pass", "resume_1__jd_1", 0) == expected


def test_online_scoring_failure_preserves_generated_resume(monkeypatch, tmp_path):
    results_path = tmp_path / "results.jsonl"
    generated = "# Alex\n\n## Summary\n\nSupported profile.\n"
    pair = {
        "pair_id": "resume_1__jd_1",
        "resume_id": "resume_1",
        "jd_id": "jd_1",
        "category": "general",
        "role_family": "analyst",
        "seniority": "junior",
        "fit_level": "strong",
        "split": "test",
        "resume": "# Alex\n",
        "jd": "Analyst",
        "keywords": [],
    }
    monkeypatch.setattr(config, "RESULTS_FILE", str(results_path))
    monkeypatch.setattr(run_all, "load_pairs", lambda **_: [pair])
    monkeypatch.setattr(run_all, "load_manifest", lambda: {"version": "test-v1"})
    monkeypatch.setattr(run_all, "dataset_hash", lambda: "dataset-hash")
    monkeypatch.setattr(run_all, "done_keys", lambda: set())
    monkeypatch.setattr(
        run_all,
        "WORKFLOWS",
        {
            "w1_single_pass": lambda *_: {
                "resume_md": generated,
                "finished": True,
                "hit_budget": False,
            }
        },
    )
    monkeypatch.setattr(
        run_all, "score",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("scorer unavailable")),
    )
    monkeypatch.setattr(
        sys, "argv", ["run_all.py", "--workflows", "w1_single_pass", "--limit", "1"],
    )

    run_all.main()

    row = json.loads(results_path.read_text().strip())
    assert row["error"] is None
    assert row["evaluation_failures"] == {
        "online_score": "RuntimeError: scorer unavailable"
    }
    assert row["task_success"] is None
    assert (tmp_path / row["resume_path"].removeprefix(str(tmp_path) + "/")).read_text() == generated


def test_operational_scoring_failure_also_preserves_generated_resume(
    monkeypatch, tmp_path,
):
    results_path = tmp_path / "results.jsonl"
    generated = "# Alex\n\n## Summary\n\nSupported profile.\n"
    pair = {
        "pair_id": "resume_1__jd_1", "resume_id": "resume_1", "jd_id": "jd_1",
        "category": "general", "role_family": "analyst", "seniority": "junior",
        "fit_level": "strong", "split": "test", "resume": "# Alex\n",
        "jd": "Analyst", "keywords": [],
    }
    monkeypatch.setattr(config, "RESULTS_FILE", str(results_path))
    monkeypatch.setattr(run_all, "load_pairs", lambda **_: [pair])
    monkeypatch.setattr(run_all, "load_manifest", lambda: {"version": "test-v1"})
    monkeypatch.setattr(run_all, "dataset_hash", lambda: "dataset-hash")
    monkeypatch.setattr(run_all, "done_keys", lambda: set())
    monkeypatch.setattr(
        run_all, "WORKFLOWS",
        {"w1_single_pass": lambda *_: {
            "resume_md": generated, "finished": True, "hit_budget": False,
        }},
    )
    monkeypatch.setattr(
        run_all, "score", lambda *_args, **_kwargs: {"evaluation_failures": {}}
    )
    monkeypatch.setattr(
        run_all, "evaluate_operational",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("ops unavailable")),
    )
    monkeypatch.setattr(
        sys, "argv", ["run_all.py", "--workflows", "w1_single_pass", "--limit", "1"],
    )

    run_all.main()

    row = json.loads(results_path.read_text().strip())
    assert row["error"] is None
    assert row["evaluation_failures"] == {
        "operational": "RuntimeError: ops unavailable"
    }
    assert (tmp_path / row["resume_path"].removeprefix(str(tmp_path) + "/")).read_text() == generated


def test_score_keeps_successful_components_when_one_evaluator_fails(monkeypatch):
    pair = {
        "resume": "# Alex\n\n## Skills\nSQL\n",
        "jd": "Need SQL",
        "keywords": [{"keyword": "SQL", "synonyms": []}],
    }
    output = {
        "resume_md": "# Alex\n\n## Skills\nSQL\n",
        "finished": True,
        "hit_budget": False,
    }
    monkeypatch.setattr(
        run_all.metrics,
        "jd_similarity",
        lambda *_: (_ for _ in ()).throw(RuntimeError("embedding unavailable")),
    )
    row = run_all.score(pair, output, Tracker(), skip_faithfulness=True)
    assert row["evaluation_failures"] == {
        "relevance": "RuntimeError: embedding unavailable"
    }
    assert row["kw_after"] == 1.0
    assert row["protected_field_violation"] is False
    assert row["fabrication_bait_violation"] is False


def test_summary_counts_failed_runs(monkeypatch, tmp_path):
    results = tmp_path / "results.jsonl"
    rows = [
        {"workflow": "w2_baseline", "error": None, "sim_after": 0.5, "finished": True},
        {"workflow": "w2_baseline", "error": "TOOL_BUDGET_EXCEEDED: x", "finished": False},
    ]
    results.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(config, "RESULTS_FILE", str(results))

    summarise.main()

    summary = (tmp_path / "summary.csv").read_text()
    assert "success_%" in summary
    assert "50" in summary
    assert "TOOL_BUDGET_EXCEEDED x1" in summary
    assert (tmp_path / "summaries" / "overall.csv").exists()
    assert (tmp_path / "summaries" / "by_category.csv").exists()
    assert (tmp_path / "summaries" / "by_fit_level.csv").exists()
