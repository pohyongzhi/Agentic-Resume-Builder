"""Real harness tests at both public workflow entry points, without Ollama."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import config
from common import finalisation
from common.harness import to_markdown
from common.llm import Tracker
from eval.operational import evaluate_operational
from workflows import w6_best_of_n, w7_plan_execute


@pytest.fixture(params=[w6_best_of_n, w7_plan_execute], ids=["w6", "w7"])
def workflow(request):
    return request.param


def setup_run(monkeypatch, tmp_path, workflow, candidate, good_resume, replies=None):
    original = tmp_path / "original.md"
    original.write_text(candidate["raw_text"])
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "outputs"))
    draft = {"resume_md": to_markdown(good_resume), "finished": True,
             "hit_budget": False, "details": {"selected_strategy": "skills_first"}}
    monkeypatch.setattr(workflow, "generate_draft", lambda pair, tracker: draft)
    responses = list(replies if replies is not None else
                     [json.dumps(candidate), json.dumps(good_resume)])
    calls = []

    def chat(messages, tracker, **kwargs):
        calls.append({"messages": messages, **kwargs})
        tracker.log(type="llm", input_tokens=10, output_tokens=5, thinking_chars=0, seconds=0)
        return SimpleNamespace(content=responses.pop(0))

    monkeypatch.setattr(finalisation, "chat", chat)
    pair = {"resume": candidate["raw_text"], "resume_path": str(original),
            "pair_id": "source__job", "jd": "Kubernetes preferred", "split": "test"}
    return pair, draft, calls, original


def test_public_workflows_validate_then_render_and_log_real_tools(
    monkeypatch, tmp_path, workflow, candidate, good_resume,
):
    pair, draft, calls, original = setup_run(
        monkeypatch, tmp_path, workflow, candidate, good_resume,
    )
    tracker = Tracker()
    out = workflow.run(pair, tracker)
    assert out["finished"] and not out["errors"]
    assert Path(out["output_path"]).is_relative_to(tmp_path / "outputs")
    assert Path(out["output_path"]).read_text() == out["resume_md"]
    assert original.read_text() == candidate["raw_text"]
    assert out["generated_resume"]["name"] == candidate["name"]
    assert out["details"]["pre_harness_resume_md"] == draft["resume_md"]
    assert out["details"]["selected_strategy"] == "skills_first"
    assert len(calls) == 2  # source parser, then schema transcription
    assert calls[0]["messages"][1]["content"] == pair["resume"]
    assert calls[1]["messages"][1]["content"] == draft["resume_md"]
    assert "raw_text" not in calls[0]["format"]["properties"]
    tools = [event for event in tracker.events if event["type"] == "tool"]
    assert [event["name"] for event in tools] == ["validate_resume", "render_resume"]
    assert all(event["ok"] for event in tools)
    assert tracker.totals()["tool_calls"] == 2
    operational = evaluate_operational(
        workflow.NAME, pair, out, tracker.totals(), tracker.events,
    )
    assert operational["tool_applicable"] is True
    assert operational["tool_sequence_valid"] is True
    assert operational["tool_call_accuracy"] == 1.0


@pytest.mark.parametrize("change", ["name", "employer", "dates", "skill", "contact"])
def test_harness_rejection_never_renders_or_returns_a_download(
    monkeypatch, tmp_path, workflow, candidate, good_resume, change,
):
    if change == "name":
        good_resume["name"] = "Another Person"
    elif change == "employer":
        good_resume["experience"][0]["employer"] = "Invented Corporation"
    elif change == "dates":
        good_resume["experience"][0]["start"] = "Jan 2001"
    elif change == "skill":
        good_resume["skills"].append("Kubernetes")
    else:
        good_resume["contact"]["email"] = "invented@example.test"
    pair, draft, _, original = setup_run(
        monkeypatch, tmp_path, workflow, candidate, good_resume,
    )
    tracker = Tracker()
    out = workflow.run(pair, tracker)
    assert not out["finished"] and not out["resume_md"] and not out["output_path"]
    assert "FINAL_VALIDATION_FAILED" in out["errors"][0]
    assert out["details"]["pre_harness_resume_md"] == draft["resume_md"]
    assert original.read_text() == candidate["raw_text"]
    assert not (tmp_path / "outputs").exists()
    tools = [event for event in tracker.events if event["type"] == "tool"]
    assert [event["name"] for event in tools] == ["validate_resume"]
    assert json.loads(tools[0]["result"])["valid"] is False


def test_section_first_draft_uses_source_name_without_guessing_a_job_title(
    monkeypatch, tmp_path, workflow, candidate, good_resume,
):
    good_resume["name"] = ""
    good_resume["contact"] = {}
    pair, draft, calls, _ = setup_run(monkeypatch, tmp_path, workflow, candidate, good_resume)
    draft["resume_md"] = "## Summary\nFrontend Engineer with experience in SQL.\n\n## Skills\nSQL"
    tracker = Tracker()
    out = workflow.run(pair, tracker)
    assert calls[1]["format"]["properties"]["name"]["enum"] == [""]
    assert out["finished"], out["errors"]
    assert out["generated_resume"]["name"] == candidate["name"]
    assert out["generated_resume"]["contact"]["email"] == candidate["contact"]["email"]
    assert tracker.totals()["tool_calls"] == 2


@pytest.mark.parametrize("heading", ["## Summary", "**EXPERIENCE**", "Skills:"])
def test_section_heading_cannot_become_a_candidate_name(heading):
    schema = finalisation._transcription_schema(heading + "\nFrontend Engineer")
    assert schema["properties"]["name"]["enum"] == [""]


def test_explicit_candidate_header_still_goes_through_name_validation():
    schema = finalisation._transcription_schema("# Another Person\n## Summary\nFrontend Engineer")
    assert "enum" not in schema["properties"]["name"]


@pytest.mark.parametrize("raw", ['{"broken":', '{"experience": [{"employer": "missing title"}]}'])
def test_bad_transcription_is_rejected_once_without_repair_loop(
    monkeypatch, tmp_path, workflow, candidate, good_resume, raw,
):
    pair, _, calls, _ = setup_run(
        monkeypatch, tmp_path, workflow, candidate, good_resume,
        replies=[json.dumps(candidate), raw],
    )
    tracker = Tracker()
    out = workflow.run(pair, tracker)
    assert not out["finished"]
    assert len(calls) == 2
    assert tracker.totals()["tool_calls"] == 1
    assert not (tmp_path / "outputs").exists()


def test_tool_budget_prevents_rendering_after_valid_draft(
    monkeypatch, tmp_path, workflow, candidate, good_resume,
):
    monkeypatch.setattr(config, "MAX_TOOL_CALLS", 1)
    pair, _, _, _ = setup_run(monkeypatch, tmp_path, workflow, candidate, good_resume)
    tracker = Tracker()
    out = workflow.run(pair, tracker)
    assert not out["finished"] and out["hit_budget"]
    assert "TOOL_BUDGET_EXCEEDED" in out["errors"][0]
    assert not (tmp_path / "outputs").exists()


def test_failed_generation_is_not_transcribed_or_saved(
    monkeypatch, tmp_path, workflow, candidate, good_resume,
):
    pair, draft, calls, _ = setup_run(monkeypatch, tmp_path, workflow, candidate, good_resume)
    draft["finished"] = False
    out = workflow.run(pair, Tracker())
    assert not out["finished"] and not out["resume_md"] and not out["output_path"]
    assert calls == []


def test_parser_failure_has_clear_error_and_no_tool_calls(
    monkeypatch, tmp_path, workflow, candidate, good_resume,
):
    pair, _, calls, _ = setup_run(
        monkeypatch, tmp_path, workflow, candidate, good_resume, replies=["not json"],
    )
    tracker = Tracker()
    out = workflow.run(pair, tracker)
    assert "HARNESS_PREPARATION_FAILED" in out["errors"][0]
    assert not out["finished"]
    assert len(calls) == 1 and tracker.totals()["tool_calls"] == 0


def test_historical_w6_trace_does_not_gain_harness_metrics():
    result = evaluate_operational(
        "w6_best_of_n", {}, {"resume_md": "old output", "finished": True},
        {"llm_calls": 5, "tool_calls": 0}, [],
    )
    assert result["tool_applicable"] is False
    assert result["tool_sequence_valid"] is None
