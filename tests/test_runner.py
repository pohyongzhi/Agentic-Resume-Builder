import json

import config
import run_all
import summarise


def test_config_hash_changes_with_shared_prompt(monkeypatch):
    first = run_all.config_hash()
    monkeypatch.setattr(run_all.prompts, "CREATOR_SYSTEM", run_all.prompts.CREATOR_SYSTEM + "\nchanged")
    assert run_all.config_hash() != first


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
