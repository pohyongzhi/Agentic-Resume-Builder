import json
from types import SimpleNamespace

import config
from common.llm import Tracker
from workflows import w3_critic


def _message(data):
    return SimpleNamespace(content=json.dumps(data))


def _feedback(critic, approved=True, severity=None):
    issues = []
    if severity:
        issues = [{
            "category": "relevance",
            "severity": severity,
            "evidence": "The strongest evidence is buried.",
            "recommendation": "Lead with the supported dashboard result.",
        }]
    return {
        "critic": critic,
        "approved": approved,
        "score": 5 if approved else 2,
        "summary": "Ready." if approved else "Needs revision.",
        "issues": issues,
    }


def _critic_name(system):
    if "fact-checker" in system:
        return "fact_checker"
    if "recruiter" in system:
        return "recruiter"
    return "formatting"


def test_three_critics_approve_in_parallel_without_revision(
    monkeypatch, tmp_path, candidate, good_resume
):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))

    def fake_chat(messages, tracker, format=None, **kwargs):
        system = messages[0]["content"]
        if system == w3_critic.STRUCTURED_CREATOR_SYSTEM:
            return _message(good_resume)
        return _message(_feedback(_critic_name(system)))

    monkeypatch.setattr(w3_critic, "chat", fake_chat)
    result = w3_critic.build_critic_graph(
        Tracker(), resume_path=str(tmp_path / "input.md"), run_id="approved"
    ).invoke({"candidate_data": candidate, "job_data": {"title": "Data Analyst"}})

    assert result["revisions"] == 0
    assert result["all_approved"] is True
    assert result["resume_md"].startswith("# Alex Tan")
    assert {r["critic"] for r in result["combined_feedback"]} == {
        "fact_checker", "recruiter", "formatting"
    }


def test_creator_revises_from_all_structured_feedback(
    monkeypatch, tmp_path, candidate, good_resume
):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    revision_prompts = []

    def fake_chat(messages, tracker, format=None, **kwargs):
        system, prompt = messages[0]["content"], messages[1]["content"]
        if system == w3_critic.STRUCTURED_CREATOR_SYSTEM:
            draft = dict(good_resume)
            if "CURRENT DRAFT" in prompt:
                revision_prompts.append(prompt)
                draft["summary"] = "Revised supported dashboard and SQL summary."
            else:
                draft["summary"] = "Initial summary."
            return _message(draft)
        critic = _critic_name(system)
        needs_revision = critic == "recruiter" and "Initial summary." in prompt
        return _message(_feedback(critic, not needs_revision, "major" if needs_revision else None))

    monkeypatch.setattr(w3_critic, "chat", fake_chat)
    result = w3_critic.build_critic_graph(Tracker(), run_id="revision").invoke(
        {"candidate_data": candidate, "job_data": {"title": "Data Analyst"}}
    )

    assert result["revisions"] == 1
    assert result["all_approved"] is True
    assert len(revision_prompts) == 1
    assert all(name in revision_prompts[0] for name in (
        "fact_checker", "recruiter", "formatting"
    ))


def test_revision_loop_stops_after_two_revisions(
    monkeypatch, tmp_path, candidate, good_resume
):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setattr(config, "MAX_CRITIC_ROUNDS", 2)
    creator_calls = 0

    def fake_chat(messages, tracker, format=None, **kwargs):
        nonlocal creator_calls
        system = messages[0]["content"]
        if system == w3_critic.STRUCTURED_CREATOR_SYSTEM:
            creator_calls += 1
            return _message(good_resume)
        critic = _critic_name(system)
        return _message(_feedback(critic, False, "major"))

    monkeypatch.setattr(w3_critic, "chat", fake_chat)
    result = w3_critic.build_critic_graph(Tracker(), run_id="budget").invoke(
        {"candidate_data": candidate, "job_data": {"title": "Data Analyst"}}
    )

    assert result["revisions"] == 2
    assert creator_calls == 3
    assert result["all_approved"] is False
    assert result["hit_budget"] is True
    assert result["resume_md"]


def test_final_harness_rejects_unsupported_fact(
    monkeypatch, tmp_path, candidate, good_resume
):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    fabricated = dict(good_resume, skills=["Airflow"])

    def fake_chat(messages, tracker, format=None, **kwargs):
        system = messages[0]["content"]
        if system == w3_critic.STRUCTURED_CREATOR_SYSTEM:
            return _message(fabricated)
        return _message(_feedback(_critic_name(system)))

    monkeypatch.setattr(w3_critic, "chat", fake_chat)
    result = w3_critic.build_critic_graph(Tracker(), run_id="rejected").invoke(
        {"candidate_data": candidate, "job_data": {"title": "Data Analyst"}}
    )

    assert result["errors"][0].startswith("W3_FINAL_VALIDATION_FAILED")
    assert "Airflow" in result["errors"][0]
    assert "resume_md" not in result
