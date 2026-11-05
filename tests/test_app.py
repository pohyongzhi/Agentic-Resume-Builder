from pathlib import Path

import pytest

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


@pytest.mark.parametrize("workflow", [app.W6_NAME, app.W7_NAME])
def test_harness_workflow_download_uses_harness_saved_file(monkeypatch, tmp_path, workflow):
    resume = tmp_path / "original.md"
    resume.write_text("# Alex\nalex@example.com")
    saved = tmp_path / "validated.md"
    saved.write_text("# Validated output")
    monkeypatch.setitem(app.WORKFLOWS, workflow, lambda pair, tracker: {
        "resume_md": saved.read_text(), "output_path": str(saved),
        "finished": True, "errors": [],
    })
    monkeypatch.setattr(app, "_save_markdown", lambda *_: pytest.fail("must use harness file"))
    state = app._run_selected_workflow(workflow, str(resume), "", "Saved JD", Tracker())
    assert state["output_path"] == str(saved)
    assert not state["errors"]
