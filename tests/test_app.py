from pathlib import Path

import app
from common.llm import Tracker


def test_comparison_workflow_is_saved_for_download(monkeypatch, tmp_path):
    resume = tmp_path / "resume.md"
    resume.write_text("# Alex Tan\nalex@example.com\n\n## Skills\nPython")
    monkeypatch.setattr(app.config, "OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setitem(
        app.WORKFLOWS,
        "fake_workflow",
        lambda pair, tracker: {
            "resume_md": f"# Tailored\n\n{pair['jd']}",
            "finished": True,
            "hit_budget": False,
        },
    )

    state = app._run_selected_workflow(
        "fake_workflow", str(resume), "", "A reproducible saved JD", Tracker()
    )

    assert state["errors"] == []
    assert state["resume_md"] == "# Tailored\n\nA reproducible saved JD"
    assert (tmp_path / "outputs").is_dir()
    assert Path(state["output_path"]).read_text(encoding="utf-8") == state["resume_md"]


def test_comparison_workflow_empty_output_becomes_error(monkeypatch, tmp_path):
    resume = tmp_path / "resume.md"
    resume.write_text("# Alex Tan\nalex@example.com")
    monkeypatch.setitem(
        app.WORKFLOWS,
        "empty_workflow",
        lambda pair, tracker: {"resume_md": "", "finished": False, "hit_budget": True},
    )

    state = app._run_selected_workflow(
        "empty_workflow", str(resume), "", "A saved JD", Tracker()
    )

    assert state["hit_budget"] is True
    assert state["errors"][0].startswith("EMPTY_OUTPUT")
