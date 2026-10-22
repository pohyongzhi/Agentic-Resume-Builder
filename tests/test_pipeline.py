"""Graph wiring and an end-to-end baseline run with a scripted fake model (no Ollama needed)."""
import json
from types import SimpleNamespace

from common import extraction
from common.llm import Tracker
from common.pipeline import build_pipeline
from workflows import w2_baseline


def _run(parser, jd, creator_calls):
    def creator(state):
        creator_calls.append(state)
        return {"resume_md": "# done"}
    return build_pipeline(creator, parser, jd).invoke({"errors": []})


def test_creator_runs_after_both_extractions():
    calls = []
    out = _run(lambda s: {"candidate_data": {"name": "A"}}, lambda s: {"job_data": {"title": "B"}}, calls)
    assert len(calls) == 1
    assert calls[0]["candidate_data"] == {"name": "A"} and calls[0]["job_data"] == {"title": "B"}
    assert out["resume_md"] == "# done"


def test_jd_failure_stops_creator():
    calls = []
    out = _run(lambda s: {"candidate_data": {"name": "A"}},
               lambda s: {"errors": ["JD_FETCH_FAILED: x. Please paste the job description instead."]}, calls)
    assert calls == []
    assert out["errors"][0].startswith("JD_FETCH_FAILED")
    assert "resume_md" not in out


def test_both_parallel_errors_kept():
    calls = []
    out = _run(lambda s: {"errors": ["SCANNED_PDF: x"]}, lambda s: {"errors": ["JD_FETCH_FAILED: y"]}, calls)
    assert calls == []
    assert sorted(e.split(":")[0] for e in out["errors"]) == ["JD_FETCH_FAILED", "SCANNED_PDF"]


def test_resume_parser_recovers_omitted_email(monkeypatch, tmp_path, candidate):
    resume_file = tmp_path / "resume.md"
    resume_file.write_text(candidate["raw_text"])
    parsed = {k: v for k, v in candidate.items() if k != "raw_text"}
    parsed["contact"] = {"email": "", "phone": "", "location": "", "links": []}
    monkeypatch.setattr(extraction, "_structured", lambda *args, **kwargs:
                        extraction.CandidateData.model_validate(parsed))

    state = extraction.make_resume_parser(Tracker())({"resume_path": str(resume_file)})

    assert state["candidate_data"]["contact"]["email"] == "alex.tan@example.com"


# ---------- end-to-end with a fake model ----------

def _msg(content="", tool_calls=None):
    calls = [SimpleNamespace(function=SimpleNamespace(name=n, arguments=a)) for n, a in (tool_calls or [])]
    return SimpleNamespace(content=content, tool_calls=calls or None)


def test_baseline_end_to_end(monkeypatch, tmp_path, candidate, good_resume):
    import config
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))

    resume_file = tmp_path / "resume.md"
    resume_file.write_text(candidate["raw_text"])
    job = {"title": "Junior Data Analyst", "company": "Example Retail", "responsibilities": ["Build dashboards"],
           "required_qualifications": ["SQL", "Excel"], "preferred_qualifications": ["Python"]}
    parsed = {k: v for k, v in candidate.items() if k != "raw_text"}

    def fake_extraction_chat(messages, tracker, format=None, **kw):
        props = format["properties"]
        return _msg(json.dumps(parsed if "experience" in props else job))

    monkeypatch.setattr(extraction, "chat", fake_extraction_chat)
    seen_formats = []

    def fake_creator_chat(messages, tracker, format=None, **kwargs):
        seen_formats.append(format)
        return _msg(json.dumps(good_resume))

    monkeypatch.setattr(w2_baseline, "chat", fake_creator_chat)

    tracker = Tracker()
    state = w2_baseline.generate(tracker, str(resume_file), pasted_jd="Junior Data Analyst ...", run_id="e2e")

    assert not state.get("errors")
    assert seen_formats[0]["properties"]["experience"]["type"] == "array"
    assert "### Business Analyst Intern, Example Logistics Pte Ltd (May 2025 - Aug 2025)" in state["resume_md"]
    assert state["output_path"].startswith(str((tmp_path / "out").resolve()))
    assert resume_file.read_text() == candidate["raw_text"]
    tool_events = [e for e in tracker.events if e["type"] == "tool"]
    assert [e["name"] for e in tool_events] == ["validate_resume", "render_resume"]
    assert all(e["ok"] for e in tool_events)


def test_baseline_budget_exceeded(monkeypatch, tmp_path, candidate, good_resume):
    import config
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(config, "MAX_TOOL_CALLS", 2)
    resume_file = tmp_path / "resume.md"
    resume_file.write_text(candidate["raw_text"])
    parsed = {k: v for k, v in candidate.items() if k != "raw_text"}
    job = {"title": "Analyst"}
    monkeypatch.setattr(extraction, "chat", lambda m, t, format=None, **k:
                        _msg(json.dumps(parsed if "experience" in format["properties"] else job)))
    bad = dict(good_resume, skills=["Airflow"])
    monkeypatch.setattr(w2_baseline, "chat", lambda *a, **k: _msg(json.dumps(bad)))

    state = w2_baseline.generate(Tracker(), str(resume_file), pasted_jd="Analyst ...", run_id="budget")
    assert state["errors"][0].startswith("TOOL_BUDGET_EXCEEDED")
    assert state["hit_budget"] and "resume_md" not in state


def test_baseline_retries_structured_validation_errors(monkeypatch, tmp_path, candidate, good_resume):
    import config
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    resume_file = tmp_path / "resume.md"
    resume_file.write_text(candidate["raw_text"])

    bad = dict(good_resume, skills=["Airflow"])
    replies = iter([_msg(json.dumps(bad)), _msg(json.dumps(good_resume))])
    monkeypatch.setattr(w2_baseline, "chat", lambda *a, **k: next(replies))

    creator = w2_baseline.make_creator(Tracker(), str(resume_file), "retry")
    state = creator({"candidate_data": candidate, "job_data": {"title": "Analyst"}})

    assert not state.get("errors")
    assert "Airflow" not in state["resume_md"]
