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
    import copy
    fabricated = copy.deepcopy(good_resume)
    fabricated["experience"][0]["employer"] = "Invented Employer"

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
    assert "Invented Employer" in result["errors"][0]
    assert "resume_md" not in result


def test_malformed_response_gets_one_bounded_corrective_retry(monkeypatch, good_resume):
    calls = []
    replies = iter([SimpleNamespace(content='{"name":'), _message(good_resume)])

    def fake_chat(messages, tracker, **kwargs):
        calls.append((messages, kwargs))
        return next(replies)

    monkeypatch.setattr(w3_critic, "chat", fake_chat)
    tracker = Tracker()
    result = w3_critic._structured("creator", "source", w3_critic.GeneratedResume, tracker)
    assert result.name == good_resume["name"]
    assert len(calls) == 2
    assert calls[0][1]["num_predict"] == config.W3_DRAFT_MAX_TOKENS
    assert "previous response failed validation" in calls[1][0][1]["content"]
    assert tracker.events[0]["raw_response"] == '{"name":'


def test_invalid_critic_stops_after_retry_budget(monkeypatch):
    import pytest
    calls = []

    def fake_chat(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(content="{")

    monkeypatch.setattr(w3_critic, "chat", fake_chat)
    with pytest.raises(ValueError, match="CriticFeedback failed after 2 attempts"):
        w3_critic._structured("critic", "source", w3_critic.CriticFeedback, Tracker())
    assert len(calls) == 2
    assert all(c["num_predict"] == config.W3_CRITIC_MAX_TOKENS for c in calls)


def test_runaway_critic_issue_list_rejected():
    import pytest
    from pydantic import ValidationError
    data = _feedback("fact_checker", False, "major")
    data["issues"] *= 6
    with pytest.raises(ValidationError):
        w3_critic.CriticFeedback.model_validate(data)


def test_harness_errors_trigger_revision_even_when_critics_approve(
    monkeypatch, tmp_path, candidate, good_resume
):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    revision_prompts = []

    def fake_chat(messages, tracker, **kwargs):
        system, prompt = messages[0]["content"], messages[1]["content"]
        if system == w3_critic.STRUCTURED_CREATOR_SYSTEM:
            if "CURRENT DRAFT" in prompt:
                revision_prompts.append(prompt)
                return _message(good_resume)
            import copy
            draft = copy.deepcopy(good_resume)
            draft["experience"][0]["employer"] = "Invented Employer"
            return _message(draft)
        return _message(_feedback(_critic_name(system)))

    monkeypatch.setattr(w3_critic, "chat", fake_chat)
    result = w3_critic.build_critic_graph(Tracker(), run_id="fact-revision").invoke(
        {"candidate_data": candidate, "job_data": {"title": "Analyst"}}
    )
    assert result["revisions"] == 1
    assert "Invented Employer" in revision_prompts[0]
    assert "not in the original resume" in revision_prompts[0]
    assert result["resume_md"]


def test_absence_marker_and_omitted_education_are_normalised(candidate, good_resume):
    draft = dict(good_resume, education=[], certifications=["None listed in original resume data."])
    prepared = w3_critic._prepare_draft(draft, candidate)
    assert prepared["certifications"] == []
    assert prepared["education"] == candidate["education"]
    assert w3_critic.validate_resume(prepared, candidate) == []


def test_real_unknown_certification_is_still_rejected(candidate, good_resume):
    draft = dict(good_resume, certifications=["N/A", "AWS Certified Solutions Architect"])
    prepared = w3_critic._prepare_draft(draft, candidate)
    assert prepared["certifications"] == ["AWS Certified Solutions Architect"]
    assert any("certification" in error for error in w3_critic.validate_resume(prepared, candidate))


def test_supplied_education_is_not_silently_replaced(candidate, good_resume):
    import copy
    draft = copy.deepcopy(good_resume)
    draft["education"][0]["degree"] = "Invented PhD"
    prepared = w3_critic._prepare_draft(draft, candidate)
    assert prepared["education"][0]["degree"] == "Invented PhD"
    assert any("education" in error for error in w3_critic.validate_resume(prepared, candidate))


def test_placeholder_regression_runs_through_graph(monkeypatch, tmp_path, candidate, good_resume):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))

    def fake_chat(messages, tracker, **kwargs):
        system = messages[0]["content"]
        if system == w3_critic.STRUCTURED_CREATOR_SYSTEM:
            return _message(dict(good_resume, education=[],
                                 certifications=["None listed in original resume data."]))
        return _message(_feedback(_critic_name(system)))

    monkeypatch.setattr(w3_critic, "chat", fake_chat)
    result = w3_critic.build_critic_graph(Tracker(), run_id="absence-fix").invoke(
        {"candidate_data": candidate, "job_data": {"title": "Analyst"}}
    )
    assert not result.get("errors")
    assert "## Education" in result["resume_md"]
    assert "None listed" not in result["resume_md"]


def test_w3_leaves_skill_support_to_llm_critics(monkeypatch, tmp_path, candidate, good_resume):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    draft = dict(good_resume, skills=["Accessibility", "Design Systems"])

    def fake_chat(messages, tracker, **kwargs):
        system = messages[0]["content"]
        if system == w3_critic.STRUCTURED_CREATOR_SYSTEM:
            return _message(draft)
        return _message(_feedback(_critic_name(system)))

    monkeypatch.setattr(w3_critic, "chat", fake_chat)
    result = w3_critic.build_critic_graph(Tracker(), run_id="llm-skill-check").invoke(
        {"candidate_data": candidate, "job_data": {"title": "Analyst"}}
    )
    assert not result.get("errors")
    assert "Design Systems" in result["resume_md"]
    # W2/default callers still use strict skill checks.
    assert any("skill" in e for e in w3_critic.validate_resume(draft, candidate))
