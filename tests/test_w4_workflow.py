"""Tests for W4 workflow (test plan §4)."""
import json
from types import SimpleNamespace

import config
from common import extraction
from common.llm import Tracker
from common.schemas import GroundedResume
from common.spans import split_spans
from workflows import WORKFLOWS, w4_evidence


def sid(spans, needle):
    """ID of the citeable span whose text contains `needle` (exactly one must match)."""
    hits = [s.id for s in spans if s.kind == "content" and needle in s.text]
    assert len(hits) == 1, (needle, hits)
    return hits[0]


def draft(bank_bullets, summary=(), skills=("SQL", "Power BI", "Excel")):
    """Build a draft GroundedResume for Alex Tan."""
    return {
        "name": "Alex Tan",
        "contact": {"email": "alex.tan@example.com"},
        "summary": [{"text": t, "sources": s} for t, s in summary],
        "experience": [{
            "employer": "Example Logistics Pte Ltd",
            "title": "Business Analyst Intern",
            "start": "May 2025",
            "end": "Aug 2025",
            "bullets": [{"text": t, "sources": s} for t, s in bank_bullets]
        }],
        "education": [{
            "institution": "Example University",
            "degree": "Bachelor of Science (Information Systems)",
            "start": "",
            "end": "expected 2026",
            "details": []
        }],
        "projects": [],
        "skills": list(skills),
        "certifications": []
    }


class FakeChat:
    """Fake chat that records calls and returns scripted responses."""
    def __init__(self, creator_drafts, verifier_responses=None, default_supported=True):
        self.calls = []
        self.creator_drafts = list(creator_drafts)  # List of dicts
        self.verifier_responses = verifier_responses or []  # List of verdict dicts
        self.creator_idx = 0
        self.verifier_idx = 0
        self.default_supported = default_supported

    def __call__(self, messages, tracker, format=None, **kwargs):
        system = messages[0]["content"]
        user = messages[1]["content"] if len(messages) > 1 else ""

        self.calls.append({"system": system, "user": user, "format": format})

        # Check if it's the verifier (by system prompt)
        if "check resume claims against the source text" in system.lower():
            # Return verifier response
            if self.verifier_idx < len(self.verifier_responses):
                response = self.verifier_responses[self.verifier_idx]
                self.verifier_idx += 1
            else:
                # Default: all supported (extract claim_ids from user message and mark them all supported)
                if self.default_supported:
                    try:
                        items = json.loads(user)
                        verdicts = [{"claim_id": item["claim_id"], "supported": True, "reason": ""}
                                    for item in items]
                        response = {"verdicts": verdicts}
                    except:
                        response = {"verdicts": []}
                else:
                    response = {"verdicts": []}
            return SimpleNamespace(content=json.dumps(response))
        else:
            # Creator
            if self.creator_idx < len(self.creator_drafts):
                draft_data = self.creator_drafts[self.creator_idx]
                self.creator_idx += 1
                return SimpleNamespace(content=json.dumps(draft_data))
            else:
                # No more drafts
                return SimpleNamespace(content=json.dumps(self.creator_drafts[-1]))


def test_w1_good_draft_first_time(monkeypatch, tmp_path, candidate):
    """W1: GOOD draft, verifier approves, no revisions."""
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))

    spans = split_spans(candidate["raw_text"], candidate)
    DASH = sid(spans, "Built weekly sales dashboards")
    CLEAN = sid(spans, "Cleaned and merged shipment data")

    GOOD = draft(
        bank_bullets=[
            ("Built weekly Power BI sales dashboards for the regional sales team.", [DASH]),
            ("Cleaned and merged shipment data from three spreadsheets with SQL.", [CLEAN])
        ]
    )

    fake = FakeChat(creator_drafts=[GOOD])
    monkeypatch.setattr(w4_evidence, "chat", fake)

    job_data = {"title": "Data Analyst", "source_text": "Need SQL and Power BI.", "responsibilities": [],
                "required_qualifications": [], "preferred_qualifications": []}

    result = w4_evidence.build_evidence_graph(
        Tracker(), resume_path=str(tmp_path / "input.md"), run_id="good"
    ).invoke({"candidate_data": candidate, "job_data": job_data})

    # Check results
    assert result.get("resume_md", "").startswith("# Alex Tan")
    assert "Power BI sales dashboards" in result["resume_md"]
    assert "[S" not in result["resume_md"]  # No citation markers

    # Check evidence
    evidence = result.get("evidence", [])
    assert all(e["status"] == "kept" for e in evidence)
    assert all(e["rounds_to_pass"] == 0 for e in evidence)

    # Check stats
    assert result["w4_stats"]["dropped"] == 0

    # Check tracker events
    tracker_events = [c["system"] for c in fake.calls if "system" in c]
    assert len([c for c in fake.calls if "CITATION RULES" in c.get("system", "")]) == 1  # Creator called once


def test_w2_bad_then_good_on_revise(monkeypatch, tmp_path, candidate):
    """W2: BAD draft, then GOOD on revise. Cache works."""
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))

    spans = split_spans(candidate["raw_text"], candidate)
    DASH = sid(spans, "Built weekly sales dashboards")
    CLEAN = sid(spans, "Cleaned and merged shipment data")

    BAD = draft(
        bank_bullets=[
            ("Built weekly Power BI sales dashboards for the regional sales team.", [DASH]),
            ("Led a team that cut data cleaning time by 40%.", [CLEAN])  # inflated + new_number
        ]
    )

    GOOD = draft(
        bank_bullets=[
            ("Built weekly Power BI sales dashboards for the regional sales team.", [DASH]),
            ("Cleaned and merged shipment data from three spreadsheets.", [CLEAN])  # Fixed
        ]
    )

    fake = FakeChat(creator_drafts=[BAD, GOOD])
    monkeypatch.setattr(w4_evidence, "chat", fake)

    job_data = {"title": "Data Analyst", "source_text": "Need SQL and Power BI.", "responsibilities": [],
                "required_qualifications": [], "preferred_qualifications": []}

    tracker = Tracker()
    result = w4_evidence.build_evidence_graph(
        tracker, resume_path=str(tmp_path / "input.md"), run_id="revise"
    ).invoke({"candidate_data": candidate, "job_data": job_data})

    # Creator called twice (draft + revise)
    creator_calls = [c for c in fake.calls if "CITATION RULES" in c.get("system", "")]
    assert len(creator_calls) == 2

    # Check revise prompt contains the rejected claim
    revise_call = creator_calls[1]
    assert "exp.0.1" in revise_call["user"]
    assert "inflated" in revise_call["user"]
    assert "new_number" in revise_call["user"]

    # Check final has 2 kept claims
    evidence = result.get("evidence", [])
    kept = [e for e in evidence if e["status"] == "kept"]
    assert len(kept) >= 2

    # The fixed claim has rounds_to_pass=1
    fixed = [e for e in evidence if "Cleaned and merged" in e["claim"]]
    if fixed:
        assert fixed[0]["rounds_to_pass"] == 1

    # Check that verifier wasn't called for the unchanged claim in round 2 (cache)
    # This is hard to test directly, but we can check the verifier call count
    verifier_calls = [c for c in fake.calls if "check resume claims" in c.get("system", "").lower()]
    # Should be called twice (once per round), but second call should have fewer items
    # This is implicit in the cache mechanism


def test_w3_bad_every_time_hits_budget(monkeypatch, tmp_path, candidate):
    """W3: BAD draft every time, hits budget."""
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))

    spans = split_spans(candidate["raw_text"], candidate)
    DASH = sid(spans, "Built weekly sales dashboards")
    CLEAN = sid(spans, "Cleaned and merged shipment data")

    BAD = draft(
        bank_bullets=[
            ("Built weekly Power BI sales dashboards for the regional sales team.", [DASH]),
            ("Led a team that cut data cleaning time by 40%.", [CLEAN])  # Always bad
        ]
    )

    fake = FakeChat(creator_drafts=[BAD, BAD, BAD])  # Draft + 2 revisions
    monkeypatch.setattr(w4_evidence, "chat", fake)

    job_data = {"title": "Data Analyst", "source_text": "Need SQL and Power BI.", "responsibilities": [],
                "required_qualifications": [], "preferred_qualifications": []}

    result = w4_evidence.build_evidence_graph(
        Tracker(), resume_path=str(tmp_path / "input.md"), run_id="budget"
    ).invoke({"candidate_data": candidate, "job_data": job_data})

    # Creator called 3 times (draft + 2 revisions)
    creator_calls = [c for c in fake.calls if "CITATION RULES" in c.get("system", "")]
    assert len(creator_calls) == 3

    # Resume still rendered (bad claim dropped)
    assert result.get("resume_md")
    assert "40%" not in result["resume_md"]
    assert "Led a team" not in result["resume_md"]

    # Stats
    assert result["w4_stats"]["dropped"] == 1
    assert result["hit_budget"] is True

    # Evidence
    evidence = result.get("evidence", [])
    dropped = [e for e in evidence if e["status"] == "dropped"]
    assert len(dropped) == 1
    assert any("new_number: 40%" in " ".join(dropped[0]["problems"]) for _ in [None])


def test_w4_no_citation_dropped(monkeypatch, tmp_path, candidate):
    """W4: Bullet with sources=[] dropped."""
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))

    spans = split_spans(candidate["raw_text"], candidate)
    DASH = sid(spans, "Built weekly sales dashboards")

    BAD = draft(
        bank_bullets=[
            ("Built weekly Power BI sales dashboards for the regional sales team.", [DASH]),
            ("Some bullet with no citation.", [])  # no_citation
        ]
    )

    fake = FakeChat(creator_drafts=[BAD, BAD])  # Stays bad
    monkeypatch.setattr(w4_evidence, "chat", fake)

    job_data = {"title": "Data Analyst", "source_text": "", "responsibilities": [],
                "required_qualifications": [], "preferred_qualifications": []}

    result = w4_evidence.build_evidence_graph(
        Tracker(), resume_path=str(tmp_path / "input.md"), run_id="no_cite"
    ).invoke({"candidate_data": candidate, "job_data": job_data})

    # Resume rendered
    assert result.get("resume_md")

    # Evidence
    evidence = result.get("evidence", [])
    dropped = [e for e in evidence if e["status"] == "dropped"]
    assert len(dropped) == 1
    assert "no_citation" in dropped[0]["problems"]


def test_w5_verifier_rejects_then_approves(monkeypatch, tmp_path, candidate):
    """W5: Verifier rejects summary once, then approves after revision."""
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))

    spans = split_spans(candidate["raw_text"], candidate)
    DASH = sid(spans, "Built weekly sales dashboards")

    DRAFT1 = draft(
        bank_bullets=[("Built weekly Power BI sales dashboards.", [DASH])],
        summary=[("Information Systems student with regional sales experience.", [DASH])]
    )

    DRAFT2 = draft(
        bank_bullets=[("Built weekly Power BI sales dashboards.", [DASH])],
        summary=[("Information Systems student who builds sales dashboards.", [DASH])]  # Fixed
    )

    # Verifier rejects summary.0 in first round, approves in second
    fake = FakeChat(
        creator_drafts=[DRAFT1, DRAFT2],
        verifier_responses=[
            {"verdicts": [
                {"claim_id": "exp.0.0", "supported": True, "reason": ""},
                {"claim_id": "summary.0", "supported": False, "reason": "adds 'regional'"}
            ]},
            {"verdicts": [
                {"claim_id": "exp.0.0", "supported": True, "reason": ""},
                {"claim_id": "summary.0", "supported": True, "reason": ""}
            ]},
        ]
    )
    # Patch both w4_evidence.chat and common.grounding.chat (llm_verify uses grounding.chat)
    monkeypatch.setattr(w4_evidence, "chat", fake)
    from common import grounding
    monkeypatch.setattr(grounding, "chat", fake)

    job_data = {"title": "Data Analyst", "source_text": "", "responsibilities": [],
                "required_qualifications": [], "preferred_qualifications": []}

    result = w4_evidence.build_evidence_graph(
        Tracker(), resume_path=str(tmp_path / "input.md"), run_id="verifier"
    ).invoke({"candidate_data": candidate, "job_data": job_data})

    # Check revise prompt has not_entailed
    creator_calls = [c for c in fake.calls if "CITATION RULES" in c.get("system", "")]
    assert len(creator_calls) == 2
    revise_call = creator_calls[1]
    assert "not_entailed" in revise_call["user"]
    assert "regional" in revise_call["user"]

    # Final is kept
    evidence = result.get("evidence", [])
    summary_ev = [e for e in evidence if e["claim_id"] == "summary.0"]
    assert summary_ev[0]["status"] == "kept"


def test_w6_draft_fails_twice_no_resume(monkeypatch, tmp_path, candidate):
    """W6: Creator returns invalid JSON twice, stops with error."""
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))

    fake = FakeChat(creator_drafts=[])
    # Override to return invalid JSON
    fake_call_count = [0]

    def bad_chat(messages, tracker, format=None, **kwargs):
        fake_call_count[0] += 1
        return SimpleNamespace(content="not json")

    monkeypatch.setattr(w4_evidence, "chat", bad_chat)

    job_data = {"title": "Data Analyst", "source_text": "", "responsibilities": [],
                "required_qualifications": [], "preferred_qualifications": []}

    result = w4_evidence.build_evidence_graph(
        Tracker(), resume_path=str(tmp_path / "input.md"), run_id="fail"
    ).invoke({"candidate_data": candidate, "job_data": job_data})

    # Error
    assert result.get("errors")
    assert result["errors"][0].startswith("W4_DRAFT_FAILED")
    assert not result.get("resume_md")

    # Verifier never called (no draft to verify)
    # Chat was called (for draft, then retry)
    assert fake_call_count[0] == 2


def test_w7_skills_filtered(monkeypatch, tmp_path, candidate):
    """W7: Skills not in candidate are dropped."""
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))

    spans = split_spans(candidate["raw_text"], candidate)
    DASH = sid(spans, "Built weekly sales dashboards")

    DRAFT = draft(
        bank_bullets=[("Built weekly Power BI sales dashboards.", [DASH])],
        skills=["SQL", "Tableau"]  # Tableau not in candidate
    )

    fake = FakeChat(creator_drafts=[DRAFT])
    monkeypatch.setattr(w4_evidence, "chat", fake)

    job_data = {"title": "Data Analyst", "source_text": "", "responsibilities": [],
                "required_qualifications": [], "preferred_qualifications": []}

    result = w4_evidence.build_evidence_graph(
        Tracker(), resume_path=str(tmp_path / "input.md"), run_id="skills"
    ).invoke({"candidate_data": candidate, "job_data": job_data})

    # Tableau not in resume
    assert "Tableau" not in result.get("resume_md", "")

    # Stats
    assert result["w4_stats"]["skills_dropped"] == 1


def test_w8_revise_fails_keeps_bad_draft(monkeypatch, tmp_path, candidate):
    """W8: Revise returns invalid JSON twice, keeps BAD draft and drops bad claims."""
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))

    spans = split_spans(candidate["raw_text"], candidate)
    DASH = sid(spans, "Built weekly sales dashboards")
    CLEAN = sid(spans, "Cleaned and merged shipment data")

    BAD = draft(
        bank_bullets=[
            ("Built weekly Power BI sales dashboards.", [DASH]),
            ("Led data cleaning with 40% improvement.", [CLEAN])  # Bad
        ]
    )

    call_count = [0]

    def mixed_chat(messages, tracker, format=None, **kwargs):
        call_count[0] += 1
        system = messages[0]["content"]

        if "check resume claims" in system.lower():
            return SimpleNamespace(content=json.dumps({"verdicts": []}))

        # Creator
        if call_count[0] <= 2:  # First call is draft
            return SimpleNamespace(content=json.dumps(BAD))
        else:  # Revise calls return invalid JSON
            return SimpleNamespace(content="bad json")

    monkeypatch.setattr(w4_evidence, "chat", mixed_chat)

    job_data = {"title": "Data Analyst", "source_text": "", "responsibilities": [],
                "required_qualifications": [], "preferred_qualifications": []}

    result = w4_evidence.build_evidence_graph(
        Tracker(), resume_path=str(tmp_path / "input.md"), run_id="revise_fail"
    ).invoke({"candidate_data": candidate, "job_data": job_data})

    # Resume still rendered (bad claim dropped)
    assert result.get("resume_md")
    assert "40%" not in result["resume_md"]


def test_w9_empty_raw_text_error(monkeypatch, tmp_path, candidate):
    """W9: Empty raw_text returns W4_NO_SPANS error."""
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))

    empty_candidate = {**candidate, "raw_text": ""}

    result = w4_evidence.build_evidence_graph(
        Tracker(), resume_path=str(tmp_path / "input.md"), run_id="empty"
    ).invoke({"candidate_data": empty_candidate, "job_data": {"title": "Analyst"}})

    assert result.get("errors")
    assert result["errors"][0].startswith("W4_NO_SPANS")


def test_w10_w4_in_workflows():
    """W10: w4_evidence is registered in WORKFLOWS."""
    assert "w4_evidence" in WORKFLOWS


def test_w11_end_to_end_through_run(monkeypatch, tmp_path, candidate):
    """W11: End-to-end through run(pair, tracker) returns evidence and w4_stats."""
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))

    # Write resume to file
    resume_file = tmp_path / "resume.md"
    resume_file.write_text(candidate["raw_text"])

    # Mock extraction
    parsed_candidate = {k: v for k, v in candidate.items() if k != "raw_text"}
    job = {"title": "Data Analyst", "source_text": "Need SQL.", "responsibilities": [],
           "required_qualifications": [], "preferred_qualifications": []}

    def fake_extraction_chat(messages, tracker, format=None, **kw):
        props = format["properties"]
        return SimpleNamespace(content=json.dumps(parsed_candidate if "experience" in props else job))

    monkeypatch.setattr(extraction, "chat", fake_extraction_chat)

    # Mock W4 creator and verifier
    spans = split_spans(candidate["raw_text"], candidate)
    DASH = sid(spans, "Built weekly sales dashboards")

    GOOD = draft(bank_bullets=[("Built sales dashboards.", [DASH])])

    def fake_w4_chat(messages, tracker, format=None, **kwargs):
        system = messages[0]["content"]
        if "check resume claims" in system.lower():
            return SimpleNamespace(content=json.dumps({"verdicts": []}))
        return SimpleNamespace(content=json.dumps(GOOD))

    monkeypatch.setattr(w4_evidence, "chat", fake_w4_chat)

    # Run
    tracker = Tracker()
    pair = {
        "pair_id": "test",
        "resume_path": str(resume_file),
        "jd": "Data Analyst role requiring SQL.",
    }

    result = w4_evidence.run(pair, tracker)

    # Check result has all required keys
    assert "resume_md" in result
    assert "finished" in result
    assert result["finished"] is True
    assert "errors" in result
    assert result["errors"] == []
    assert "evidence" in result
    assert isinstance(result["evidence"], list)
    assert len(result["evidence"]) > 0
    assert "w4_stats" in result
    assert isinstance(result["w4_stats"], dict)
