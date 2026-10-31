"""Tests for W4 workflow (test plan §4)."""
import json
from types import SimpleNamespace

import pytest

import config
from common import extraction, grounding, llm
from common.llm import Tracker
from common.schemas import GroundedResume
from common.spans import split_spans
from workflows import WORKFLOWS, w4_evidence


@pytest.fixture(autouse=True)
def forbid_real_model_calls(monkeypatch):
    """Fail immediately if a workflow test accidentally reaches Ollama."""
    def unexpected_call(*args, **kwargs):
        pytest.fail("W4 tests must patch both the creator and verifier chat functions")

    monkeypatch.setattr(llm._client, "chat", unexpected_call)


def patch_chat(monkeypatch, fake):
    monkeypatch.setattr(w4_evidence, "chat", fake)
    monkeypatch.setattr(grounding, "chat", fake)


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
    patch_chat(monkeypatch, fake)

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
    patch_chat(monkeypatch, fake)

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

    verifier_calls = [c for c in fake.calls if "check resume claims" in c.get("system", "").lower()]
    assert len(verifier_calls) == 2
    assert [item["claim_id"] for item in json.loads(verifier_calls[1]["user"])] == ["exp.0.1"]
    rounds = [event for event in tracker.events if event["type"] == "w4_round"]
    assert [(event["llm_checked"], event["llm_cached"]) for event in rounds] == [(1, 0), (1, 1)]


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
    patch_chat(monkeypatch, fake)

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
    patch_chat(monkeypatch, fake)

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
    patch_chat(monkeypatch, fake)

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

    patch_chat(monkeypatch, bad_chat)

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
    patch_chat(monkeypatch, fake)

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
        system = messages[0]["content"]

        if "check resume claims" in system.lower():
            items = json.loads(messages[1]["content"])
            return SimpleNamespace(content=json.dumps({"verdicts": [
                {"claim_id": item["claim_id"], "supported": True} for item in items
            ]}))

        # Creator
        call_count[0] += 1
        if call_count[0] == 1:  # First call is draft
            return SimpleNamespace(content=json.dumps(BAD))
        else:  # Revise calls return invalid JSON
            return SimpleNamespace(content="bad json")

    patch_chat(monkeypatch, mixed_chat)

    job_data = {"title": "Data Analyst", "source_text": "", "responsibilities": [],
                "required_qualifications": [], "preferred_qualifications": []}

    result = w4_evidence.build_evidence_graph(
        Tracker(), resume_path=str(tmp_path / "input.md"), run_id="revise_fail"
    ).invoke({"candidate_data": candidate, "job_data": job_data})

    # Resume still rendered (bad claim dropped)
    assert result.get("resume_md")
    assert "40%" not in result["resume_md"]
    assert "Built weekly Power BI sales dashboards." in result["resume_md"]
    assert call_count[0] == 3  # Draft, failed revision, failed retry; then finalise.
    assert result["w4_stats"]["rounds_used"] == 1
    assert result["w4_stats"]["dropped"] == 1
    assert result["hit_budget"] is False


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


@pytest.mark.parametrize("revise", [False, True])
def test_education_fields_preserved_after_creator_reply(monkeypatch, tmp_path, candidate, revise):
    """A creator-added year label must not break protected-field validation."""
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    spans = split_spans(candidate["raw_text"], candidate)
    source = sid(spans, "Built weekly sales dashboards")
    changed = draft([("Built weekly sales dashboards.", [source])])
    changed["education"][0].update({
        "degree": "Bachelor of Science (Information Systems) (Year 3)",
        "start": "2023", "end": "2027",
    })
    replies = [changed]
    if revise:
        replies.insert(0, draft([("Increased sales by 40%.", [source])]))
    fake = FakeChat(replies)
    patch_chat(monkeypatch, fake)

    result = w4_evidence.build_evidence_graph(
        Tracker(), resume_path=str(tmp_path / "input.md"), run_id="education"
    ).invoke({"candidate_data": candidate, "job_data": {"title": "Analyst"}})

    assert not result.get("errors")
    assert result["w4_stats"]["rounds_used"] == int(revise)
    education = result["generated_resume"]["education"][0]
    for field in ("institution", "degree", "start", "end"):
        assert education[field] == candidate["education"][0][field]
    assert "Year 3" not in result["resume_md"]
    assert "Built weekly sales dashboards." in result["resume_md"]


def test_education_restoration_does_not_guess_between_degrees(candidate):
    candidate["education"].append({
        "institution": "Example University", "degree": "Master of Science",
        "start": "2027", "end": "2028", "details": [],
    })
    original = GroundedResume.model_validate(draft([]))
    original.education[0].degree = "Unknown degree (Year 3)"
    restored = w4_evidence.preserve_education_fields(original, candidate)
    assert restored == original  # Two entries at this school: no safe fallback.

    original.education[0].degree = "Master of Science"
    restored = w4_evidence.preserve_education_fields(original, candidate)
    assert restored.education[0].start == "2027"
    assert restored.education[0].end == "2028"
    assert original.education[0].end == "expected 2026"  # Input is not mutated.


def test_unknown_education_still_fails_harness(monkeypatch, tmp_path, candidate):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    changed = draft([])
    changed["education"][0]["institution"] = "Unknown University"
    patch_chat(monkeypatch, FakeChat([changed]))
    result = w4_evidence.build_evidence_graph(
        Tracker(), resume_path=str(tmp_path / "input.md"), run_id="unknown-school"
    ).invoke({"candidate_data": candidate, "job_data": {"title": "Analyst"}})
    assert result["errors"][0].startswith("W4_FINAL_VALIDATION_FAILED:")
    assert not result.get("resume_md")


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
            items = json.loads(messages[1]["content"])
            return SimpleNamespace(content=json.dumps({"verdicts": [
                {"claim_id": item["claim_id"], "supported": True} for item in items
            ]}))
        return SimpleNamespace(content=json.dumps(GOOD))

    patch_chat(monkeypatch, fake_w4_chat)

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
    assert "Built sales dashboards." in result["resume_md"]
    assert "errors" in result
    assert result["errors"] == []
    assert "evidence" in result
    assert isinstance(result["evidence"], list)
    assert len(result["evidence"]) > 0
    assert "w4_stats" in result
    assert isinstance(result["w4_stats"], dict)
@pytest.mark.parametrize("description,supported", [
    ("Led a team of 50 and increased revenue by 900% using Tableau.", False),
    ({"text": "Increased revenue by 900%.", "sources": "SOURCE"}, False),
    ({"text": "Built a report viewer in Python.", "sources": "SOURCE"}, True),
])
def test_project_descriptions_are_verified(monkeypatch, tmp_path, candidate, description, supported):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    candidate["projects"] = [{"name": "Reporting Tool", "description": "Report viewer", "bullets": []}]
    candidate["raw_text"] += "\n## Projects\n### Reporting Tool\n- Built a report viewer in Python.\n"
    source = sid(split_spans(candidate["raw_text"], candidate), "Built a report viewer")
    if isinstance(description, dict):
        description = {**description, "sources": [source]}
    proposed = draft([])
    proposed["projects"] = [{"name": "Reporting Tool", "description": description}]
    fake = FakeChat([proposed])
    patch_chat(monkeypatch, fake)
    result = w4_evidence.build_evidence_graph(Tracker(), run_id="project").invoke({
        "candidate_data": candidate, "job_data": {"title": "Analyst"},
    })
    assert not result.get("errors")
    assert "900%" not in result["resume_md"]
    row = next(e for e in result["evidence"] if e["claim_id"] == "proj.0.description")
    assert row["status"] == ("kept" if supported else "dropped")
    verifier_calls = [c for c in fake.calls if c["system"] == grounding.VERIFIER_SYSTEM]
    assert len(verifier_calls) == int(supported)
    if supported:
        assert "Built a report viewer in Python." in result["resume_md"]
        assert json.loads(verifier_calls[0]["user"])[0]["claim_id"] == "proj.0.description"
    else:
        assert result["generated_resume"]["projects"][0]["description"] == ""


def test_project_description_rejected_by_verifier_is_dropped(monkeypatch, tmp_path, candidate):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    candidate["projects"] = [{"name": "Reporting Tool"}]
    candidate["raw_text"] += "\n## Projects\n### Reporting Tool\n- Built reports.\n"
    source = sid(split_spans(candidate["raw_text"], candidate), "Built reports.")
    proposed = draft([])
    proposed["projects"] = [{"name": "Reporting Tool", "description": {
        "text": "Built reports for executives.", "sources": [source],
    }}]
    fake = FakeChat([proposed], verifier_responses=[{"verdicts": [{
        "claim_id": "proj.0.description", "supported": False, "reason": "adds audience",
    }]}])
    patch_chat(monkeypatch, fake)
    result = w4_evidence.build_evidence_graph(Tracker(), run_id="project-rejected").invoke({
        "candidate_data": candidate, "job_data": {"title": "Analyst"},
    })
    assert "executives" not in result["resume_md"]
    assert result["evidence"][0]["problems"] == ["not_entailed: adds audience"]
    assert result["evidence"][0]["status"] == "dropped"


@pytest.mark.parametrize("replace_claim", [False, True])
def test_first_pass_history_survives_deletion_and_replacement(monkeypatch, tmp_path, candidate, replace_claim):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "out"))
    source = sid(split_spans(candidate["raw_text"], candidate), "Built weekly sales dashboards")
    good = ("Built weekly sales dashboards.", [source])
    bad = ("Reduced time by 40%.", [source])
    if replace_claim:
        initial = draft([], summary=[good, bad])
        revised = draft([], summary=[("Built sales dashboards in Excel.", [source])])
    else:
        initial = draft([], summary=[bad, good])
        revised = draft([], summary=[good])
    patch_chat(monkeypatch, FakeChat([initial, revised]))
    result = w4_evidence.build_evidence_graph(Tracker(), run_id="history").invoke({
        "candidate_data": candidate, "job_data": {"title": "Analyst"},
    })
    assert result["w4_stats"]["rounds_used"] == 1
    assert result["evidence"][0]["rounds_to_pass"] == int(replace_claim)
