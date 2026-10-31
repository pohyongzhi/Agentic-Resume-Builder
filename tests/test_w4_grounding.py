"""Tests for W4 grounding checks (test plan §3)."""
import json
from types import SimpleNamespace

import pytest

from common import grounding
from common.grounding import (
    cache_key,
    check_citations,
    drop_unsupported,
    filter_skills,
    iter_claims,
    llm_verify,
    rule_check,
    skill_terms,
    to_generated,
)
from common.schemas import Claim, GroundedResume
from common.spans import split_spans

# Use the PDF candidate from test_w4_spans
PDF_RAW = (
    "Jordan Lim \n"
    " \n"
    "Singapore • jordan.lim@example.com • (+65) 9000 0000 \n"
    "About Me\n"
    "I am a final-year Business Analytics student who enjoys turning messy data into clear reports for\n"
    "teams. I like working with people and improving everyday processes. \n"
    "Experience \n"
    "Data Analyst Intern - Example Bank               May 2026 – Present \n"
    "• Built automated reporting workflows in Excel and Python, reducing manual effort for the\n"
    "operations team \n"
    "• Gathered requirements from business users and wrote user stories for dashboard\n"
    "enhancements \n"
    "Operations Assistant (National Service) - Example Agency      Apr 2023 – Aug 2024 \n"
    "• Processed approval requests through an internal SAP-based system.  \n"
    "• Designed an Excel -based tracker to streamline approvals, enabling user self -service \n"
    "Education \n"
    "Example University       Aug 2024 – Dec 2027 \n"
    "Bachelor of Science (Business Analytics), GPA: 3.6/4.0 \n"
    "Skills & Interests \n"
    "Skills: Python, Excel, SQL, SAP, Figma \n"
)

PDF_CANDIDATE = {
    "name": "Jordan Lim",
    "contact": {"email": "jordan.lim@example.com", "phone": "(+65) 9000 0000", "location": "Singapore", "links": []},
    "summary": "",
    "experience": [
        {"employer": "Example Bank", "title": "Data Analyst Intern", "start": "May 2026", "end": "Present", "bullets": []},
        {"employer": "Example Agency", "title": "Operations Assistant (National Service)",
         "start": "Apr 2023", "end": "Aug 2024", "bullets": []},
    ],
    "education": [{"institution": "Example University", "degree": "Bachelor of Science (Business Analytics)",
                   "start": "Aug 2024", "end": "Dec 2027", "details": []}],
    "projects": [],
    "skills": ["Python", "Excel", "SQL", "SAP", "Figma"],
    "certifications": [],
    "achievements": [],
    "raw_text": PDF_RAW,
}


def sid(spans, needle):
    """ID of the citeable span whose text contains `needle` (exactly one must match)."""
    hits = [s.id for s in spans if s.kind == "content" and needle in s.text]
    assert len(hits) == 1, (needle, hits)
    return hits[0]


def resume_with(bullets_bank, bullets_agency=(), summary=()):
    """Build a GroundedResume from Jordan Lim candidate with specified bullets."""
    return GroundedResume.model_validate({
        "name": "Jordan Lim",
        "contact": {"email": "jordan.lim@example.com"},
        "summary": [{"text": t, "sources": s} for t, s in summary],
        "experience": [
            {"employer": "Example Bank", "title": "Data Analyst Intern", "start": "May 2026", "end": "Present",
             "bullets": [{"text": t, "sources": s} for t, s in bullets_bank]},
            {"employer": "Example Agency", "title": "Operations Assistant (National Service)",
             "start": "Apr 2023", "end": "Aug 2024",
             "bullets": [{"text": t, "sources": s} for t, s in bullets_agency]},
        ],
        "education": [{"institution": "Example University", "degree": "Bachelor of Science (Business Analytics)",
                       "start": "Aug 2024", "end": "Dec 2027", "details": []}],
        "skills": ["Python", "Excel"],
    })


# ========== Section A: iter_claims ==========

def test_iter_claims_ids_and_order():
    """A1: iter_claims returns correct IDs in order."""
    spans = split_spans(PDF_RAW, PDF_CANDIDATE)
    BANK1 = sid(spans, "Built automated reporting")
    BANK2 = sid(spans, "Gathered requirements")
    AGENCY1 = sid(spans, "Processed approval")

    resume = resume_with(
        bullets_bank=[("Bank bullet 1", [BANK1]), ("Bank bullet 2", [BANK2])],
        bullets_agency=[("Agency bullet", [AGENCY1])],
        summary=[("Summary 1", [BANK1]), ("Summary 2", [BANK2])]
    )

    claims = iter_claims(resume)
    ids = [c["claim_id"] for c in claims]
    assert ids == ["summary.0", "summary.1", "exp.0.0", "exp.0.1", "exp.1.0"]

    # Check entry labels
    bank_claims = [c for c in claims if c["claim_id"].startswith("exp.0")]
    assert all(c["entry_label"] == "Data Analyst Intern, Example Bank" for c in bank_claims)


# ========== Section B: check_citations ==========

def test_citation_valid():
    """B1: Bank bullet citing Bank span passes."""
    spans = split_spans(PDF_RAW, PDF_CANDIDATE)
    BANK1 = sid(spans, "Built automated reporting")
    resume = resume_with(bullets_bank=[("Built reporting workflows", [BANK1])])

    problems = check_citations(resume, spans, PDF_CANDIDATE)
    assert problems["exp.0.0"] == []


def test_citation_no_sources():
    """B2: Bullet with no sources flagged."""
    spans = split_spans(PDF_RAW, PDF_CANDIDATE)
    resume = resume_with(bullets_bank=[("Some bullet", [])])

    problems = check_citations(resume, spans, PDF_CANDIDATE)
    assert "no_citation" in problems["exp.0.0"]


def test_citation_bad_id():
    """B3: Citing unknown span ID flagged."""
    spans = split_spans(PDF_RAW, PDF_CANDIDATE)
    resume = resume_with(bullets_bank=[("Some bullet", ["S999"])])

    problems = check_citations(resume, spans, PDF_CANDIDATE)
    assert any("bad_citation" in p for p in problems["exp.0.0"])


def test_citation_entry_header():
    """B4: Citing entry_header span flagged."""
    spans = split_spans(PDF_RAW, PDF_CANDIDATE)
    # Find an entry_header span
    entry_header = next(s for s in spans if s.kind == "entry_header")
    resume = resume_with(bullets_bank=[("Some bullet", [entry_header.id])])

    problems = check_citations(resume, spans, PDF_CANDIDATE)
    assert any("bad_citation" in p for p in problems["exp.0.0"])


def test_citation_wrong_entry():
    """B5: Bank bullet citing Agency span flagged."""
    spans = split_spans(PDF_RAW, PDF_CANDIDATE)
    AGENCY1 = sid(spans, "Processed approval")
    resume = resume_with(bullets_bank=[("Some bullet", [AGENCY1])])

    problems = check_citations(resume, spans, PDF_CANDIDATE)
    assert any("wrong_entry" in p for p in problems["exp.0.0"])


def test_citation_summary_can_cite_any():
    """B6: Summary can cite Agency span."""
    spans = split_spans(PDF_RAW, PDF_CANDIDATE)
    AGENCY1 = sid(spans, "Processed approval")
    resume = resume_with(bullets_bank=[], summary=[("Summary", [AGENCY1])])

    problems = check_citations(resume, spans, PDF_CANDIDATE)
    assert problems["summary.0"] == []


def test_citation_too_many_sources():
    """B7: Bullet citing 4 spans flagged."""
    spans = split_spans(PDF_RAW, PDF_CANDIDATE)
    BANK1 = sid(spans, "Built automated reporting")
    BANK2 = sid(spans, "Gathered requirements")
    AGENCY1 = sid(spans, "Processed approval")
    AGENCY2 = sid(spans, "Excel-based tracker")

    # Summary can cite any span, so use 4 valid Bank/summary spans
    resume = resume_with(
        bullets_bank=[("Bullet", [BANK1])],
        summary=[("Summary with 4 sources", [BANK1, BANK2, AGENCY1, AGENCY2])]
    )

    problems = check_citations(resume, spans, PDF_CANDIDATE, max_sources=3)
    assert "too_many_sources" in problems["summary.0"]


def test_citation_education_wrong_section():
    """B8: Education detail citing Bank span flagged."""
    spans = split_spans(PDF_RAW, PDF_CANDIDATE)
    BANK1 = sid(spans, "Built automated reporting")

    resume = GroundedResume.model_validate({
        "name": "Jordan Lim",
        "contact": {"email": "jordan.lim@example.com"},
        "experience": [],
        "education": [{
            "institution": "Example University",
            "degree": "Bachelor of Science (Business Analytics)",
            "details": [{"text": "Detail", "sources": [BANK1]}]
        }],
        "skills": [],
    })

    problems = check_citations(resume, spans, PDF_CANDIDATE)
    assert any("wrong_section" in p for p in problems["edu.0.0"])


def test_citation_unknown_employer_skips_wrong_entry():
    """B9: Draft with unknown employer skips wrong_entry check."""
    spans = split_spans(PDF_RAW, PDF_CANDIDATE)
    BANK1 = sid(spans, "Built automated reporting")

    resume = GroundedResume.model_validate({
        "name": "Jordan Lim",
        "contact": {"email": "jordan.lim@example.com"},
        "experience": [{
            "employer": "Other Corp",
            "title": "Analyst",
            "bullets": [{"text": "Bullet", "sources": [BANK1]}]
        }],
        "education": [],
        "skills": [],
    })

    problems = check_citations(resume, spans, PDF_CANDIDATE)
    # Should not have wrong_entry because the entry can't be matched
    assert not any("wrong_entry" in p for p in problems.get("exp.0.0", []))


def test_citation_all_claims_have_key():
    """B10: Every claim ID has a key in results."""
    spans = split_spans(PDF_RAW, PDF_CANDIDATE)
    BANK1 = sid(spans, "Built automated reporting")
    resume = resume_with(
        bullets_bank=[("B1", [BANK1]), ("B2", [])],
        summary=[("S1", [BANK1])]
    )

    problems = check_citations(resume, spans, PDF_CANDIDATE)
    all_ids = {c["claim_id"] for c in iter_claims(resume)}
    assert all_ids == set(problems.keys())


# ========== Section C: rule_check ==========

JOB_TEXT = "Experience with SQL, Tableau and Power BI. Familiarity with A/B testing, KPIs and Jira is a plus."


def test_rule_check_passes():
    """C1: Claim that rewords cited text passes."""
    cited = "Built automated reporting workflows in Excel and Python, reducing manual effort for the operations team"
    claim = "Built automated Excel and Python reporting workflows that cut manual effort for operations"
    problems = rule_check(claim, cited, JOB_TEXT)
    assert problems == []


def test_rule_check_new_number():
    """C2: Claim with new number flagged."""
    cited = "Built automated reporting workflows in Excel and Python, reducing manual effort for the operations team"
    claim = "Cut manual reporting effort by 40% with automated workflows"
    problems = rule_check(claim, cited, JOB_TEXT)
    assert any("new_number: 40%" in p for p in problems)


def test_rule_check_year_ignored():
    """C3: Year numbers ignored."""
    cited = "Built automated reporting workflows in Excel and Python, reducing manual effort for the operations team"
    claim = "Built reporting workflows in 2026"
    problems = rule_check(claim, cited, JOB_TEXT)
    # 2026 is a year, should not be flagged
    assert not any("new_number" in p for p in problems)


def test_rule_check_new_skill():
    """C4: Claim with skill from JD not in cited text flagged."""
    cited = "Built automated reporting workflows in Excel and Python, reducing manual effort for the operations team"
    claim = "Built automated reporting in Excel, Python and Tableau"
    problems = rule_check(claim, cited, JOB_TEXT)
    assert any("new_skill: Tableau" in p for p in problems)


def test_rule_check_new_skill_kpi():
    """C5: New skill KPI flagged."""
    cited = "Gathered requirements from business users and wrote user stories for dashboard enhancements"
    claim = "Gathered requirements and defined KPIs for dashboards"
    problems = rule_check(claim, cited, JOB_TEXT)
    assert any("new_skill: KPI" in p or "new_skill: KPIs" in p for p in problems)


@pytest.mark.parametrize("job,claim,cited,unsupported", [
    ("Need KPI.", "Tracked KPIs", "Tracked KPI", False),
    ("Need KPIs.", "Tracked KPI", "Tracked KPIs", False),
    ("Need KPI.", "Tracked KPIs", "Tracked reporting", True),
    ("Need KPIs.", "Tracked KPI", "Tracked reporting", True),
    ("Need AWS.", "Used AWS", "Used AW", True),
])
def test_rule_check_skill_plural_matching(job, claim, cited, unsupported):
    problems = rule_check(claim, cited, job)
    assert any(problem.startswith("new_skill:") for problem in problems) is unsupported


def test_rule_check_new_skill_ab_testing():
    """C6: New skill A/B testing flagged."""
    cited = "Gathered requirements from business users and wrote user stories for dashboard enhancements"
    claim = "Ran A/B testing on dashboard enhancements"
    problems = rule_check(claim, cited, JOB_TEXT)
    assert any("new_skill: A/B" in p or "A/B testing" in p.lower() for p in problems)


def test_rule_check_inflated_led():
    """C7: Inflated term 'led' flagged."""
    cited = "Built automated reporting workflows in Excel and Python, reducing manual effort for the operations team"
    claim = "Led the automation of reporting workflows"
    problems = rule_check(claim, cited, JOB_TEXT)
    assert any("inflated: led" in p for p in problems)


def test_rule_check_inflated_advanced():
    """C8: Inflated term 'advanced' flagged."""
    cited = "Built automated reporting workflows in Excel and Python, reducing manual effort for the operations team"
    claim = "Advanced Excel user who built automated reporting"
    problems = rule_check(claim, cited, JOB_TEXT)
    assert any("inflated: advanced" in p for p in problems)


def test_rule_check_led_vs_ledger():
    """C9: 'led' must not match 'ledger'."""
    cited = "Built reporting workflows and maintained ledger files"
    claim = "Built reporting workflows; maintained ledger files"
    problems = rule_check(claim, cited, JOB_TEXT)
    # Should not flag 'led' because 'ledger' in cited text has 'led' in it
    # Actually, the rule checks if a word STARTS with 'led', so 'ledger' starts with 'led'
    # Wait, the spec says "led" should not match "ledger". Let me re-read.
    # The spec says the rule should match on word STARTS, so "led" matches "ledger".
    # But test C9 says it should NOT flag. So the implementation needs to be word-boundary aware.
    # Actually, I need to check whole words only. Let me fix the rule_check implementation.
    assert problems == []


def test_rule_check_led_in_source():
    """C10: 'Led' in source text passes."""
    cited = "Led weekly stand-ups with the team"
    claim = "Led weekly stand-ups"
    problems = rule_check(claim, cited, JOB_TEXT)
    assert problems == []


def test_rule_check_skill_in_cited():
    """C11: SQL in cited text passes."""
    cited = "Cleaned data using SQL queries"
    claim = "Used SQL to clean data"
    problems = rule_check(claim, cited, JOB_TEXT)
    assert not any("new_skill: SQL" in p for p in problems)


def test_skill_terms_extraction():
    """C12: skill_terms extracts correct terms."""
    terms = skill_terms(JOB_TEXT)
    assert "SQL" in terms
    assert "Tableau" in terms
    assert "Power BI" in terms or "Power" in terms  # May extract as phrase or separate
    assert "Jira" in terms
    # Should NOT contain generic sentence-initial words
    assert "Experience" not in terms
    assert "Familiarity" not in terms


# ========== Section D: Finalise helpers ==========

def test_drop_unsupported_keeps_entry():
    """D1: drop_unsupported removes flagged claims but keeps entry."""
    spans = split_spans(PDF_RAW, PDF_CANDIDATE)
    BANK1 = sid(spans, "Built automated reporting")
    BANK2 = sid(spans, "Gathered requirements")

    resume = resume_with(bullets_bank=[("Bullet 1", [BANK1]), ("Bullet 2", [BANK2])])

    problems = {"exp.0.1": ["inflated: led"], "exp.0.0": []}
    result = drop_unsupported(resume, problems)

    assert len(result.experience[0].bullets) == 1
    assert result.experience[0].bullets[0].text == "Bullet 1"
    # Entry itself is kept
    assert len(result.experience) == 2


def test_drop_unsupported_all_bullets():
    """D2: Entry kept even if all bullets dropped."""
    spans = split_spans(PDF_RAW, PDF_CANDIDATE)
    BANK1 = sid(spans, "Built automated reporting")

    resume = resume_with(bullets_bank=[("Bullet 1", [BANK1])])

    problems = {"exp.0.0": ["inflated: led"]}
    result = drop_unsupported(resume, problems)

    assert len(result.experience) == 2
    assert result.experience[0].bullets == []


def test_filter_skills():
    """D3: filter_skills keeps only valid skills."""
    kept, dropped = filter_skills(
        ["Python", "Tableau", "excel"],
        parsed=["Python", "Excel", "SQL", "SAP", "Figma"],
        raw_text=PDF_RAW
    )
    assert "Python" in kept
    assert "excel" in kept  # case-insensitive match
    assert "Tableau" in dropped


def test_to_generated():
    """D4: to_generated removes citations."""
    spans = split_spans(PDF_RAW, PDF_CANDIDATE)
    BANK1 = sid(spans, "Built automated reporting")

    resume = resume_with(
        bullets_bank=[("Built reporting workflows", [BANK1])],
        summary=[("Information Systems student", [BANK1])]
    )

    generated = to_generated(resume)

    # Should be valid GeneratedResume structure
    assert "name" in generated
    assert "summary" in generated
    assert isinstance(generated["summary"], str)
    assert generated["summary"] == "Information Systems student"

    # Bullets should be strings
    assert isinstance(generated["experience"][0]["bullets"], list)
    assert generated["experience"][0]["bullets"][0] == "Built reporting workflows"

    # No "sources" key in the structure (check it's not a dict key)
    dump = json.dumps(generated)
    # The word "sources" shouldn't appear as a JSON key
    assert '"sources"' not in dump
    # No [S markers
    assert "[S" not in dump


# ========== Section E: llm_verify ==========

def test_llm_verify_all_supported(monkeypatch):
    """E1: 3 items, all supported."""
    from common.llm import Tracker

    calls = []

    def fake_chat(messages, tracker, format=None, **kwargs):
        calls.append((messages, format))
        return SimpleNamespace(content=json.dumps({
            "verdicts": [
                {"claim_id": "exp.0.0", "supported": True, "reason": ""},
                {"claim_id": "exp.0.1", "supported": True, "reason": ""},
                {"claim_id": "exp.0.2", "supported": True, "reason": ""},
            ]
        }))

    monkeypatch.setattr(grounding, "chat", fake_chat)

    items = [
        {"claim_id": "exp.0.0", "claim": Claim(text="A", sources=["S1"]), "cited_text": "A text"},
        {"claim_id": "exp.0.1", "claim": Claim(text="B", sources=["S2"]), "cited_text": "B text"},
        {"claim_id": "exp.0.2", "claim": Claim(text="C", sources=["S3"]), "cited_text": "C text"},
    ]

    cache = {}
    results = llm_verify(items, Tracker(), cache)

    assert len(calls) == 1
    assert all(results[f"exp.0.{i}"]["supported"] for i in range(3))
    assert len(cache) == 3


def test_llm_verify_cached(monkeypatch):
    """E2: Same items again, no calls."""
    from common.llm import Tracker

    calls = []

    def fake_chat(messages, tracker, format=None, **kwargs):
        calls.append((messages, format))
        return SimpleNamespace(content=json.dumps({"verdicts": []}))

    monkeypatch.setattr(grounding, "chat", fake_chat)

    items = [
        {"claim_id": "exp.0.0", "claim": Claim(text="A", sources=["S1"]), "cited_text": "A text"},
        {"claim_id": "exp.0.1", "claim": Claim(text="B", sources=["S2"]), "cited_text": "B text"},
        {"claim_id": "exp.0.2", "claim": Claim(text="C", sources=["S3"]), "cited_text": "C text"},
    ]

    # Pre-populate cache
    cache = {
        cache_key(Claim(text="A", sources=["S1"])): {"supported": True, "reason": ""},
        cache_key(Claim(text="B", sources=["S2"])): {"supported": True, "reason": ""},
        cache_key(Claim(text="C", sources=["S3"])): {"supported": True, "reason": ""},
    }

    results = llm_verify(items, Tracker(), cache)

    assert len(calls) == 0
    assert all(results[f"exp.0.{i}"]["supported"] for i in range(3))


def test_llm_verify_partial_cache(monkeypatch):
    """E3: 2 cached + 1 new, only new item sent."""
    from common.llm import Tracker

    calls = []

    def fake_chat(messages, tracker, format=None, **kwargs):
        calls.append((messages, format))
        user_msg = messages[1]["content"]
        # Check that only exp.0.2 is in the user message
        assert "exp.0.2" in user_msg
        assert "exp.0.0" not in user_msg
        assert "exp.0.1" not in user_msg

        return SimpleNamespace(content=json.dumps({
            "verdicts": [
                {"claim_id": "exp.0.2", "supported": True, "reason": ""},
            ]
        }))

    monkeypatch.setattr(grounding, "chat", fake_chat)

    items = [
        {"claim_id": "exp.0.0", "claim": Claim(text="A", sources=["S1"]), "cited_text": "A text"},
        {"claim_id": "exp.0.1", "claim": Claim(text="B", sources=["S2"]), "cited_text": "B text"},
        {"claim_id": "exp.0.2", "claim": Claim(text="C", sources=["S3"]), "cited_text": "C text"},
    ]

    cache = {
        cache_key(Claim(text="A", sources=["S1"])): {"supported": True, "reason": ""},
        cache_key(Claim(text="B", sources=["S2"])): {"supported": True, "reason": ""},
    }

    results = llm_verify(items, Tracker(), cache)

    assert len(calls) == 1


def test_llm_verify_missing_verdict(monkeypatch):
    """E4: Missing verdict in response."""
    from common.llm import Tracker

    def fake_chat(messages, tracker, format=None, **kwargs):
        return SimpleNamespace(content=json.dumps({
            "verdicts": [
                {"claim_id": "exp.0.0", "supported": True, "reason": ""},
                # exp.0.1 is missing
            ]
        }))

    monkeypatch.setattr(grounding, "chat", fake_chat)

    items = [
        {"claim_id": "exp.0.0", "claim": Claim(text="A", sources=["S1"]), "cited_text": "A text"},
        {"claim_id": "exp.0.1", "claim": Claim(text="B", sources=["S2"]), "cited_text": "B text"},
    ]

    cache = {}
    results = llm_verify(items, Tracker(), cache)

    assert results["exp.0.1"]["supported"] is False
    assert results["exp.0.1"]["reason"] == "verifier gave no verdict"


def test_llm_verify_invalid_json(monkeypatch):
    """E5: Invalid JSON from verifier."""
    from common.llm import Tracker

    def fake_chat(messages, tracker, format=None, **kwargs):
        return SimpleNamespace(content="not json")

    monkeypatch.setattr(grounding, "chat", fake_chat)

    items = [
        {"claim_id": "exp.0.0", "claim": Claim(text="A", sources=["S1"]), "cited_text": "A text"},
    ]

    cache = {}
    results = llm_verify(items, Tracker(), cache)

    # Should not crash, should mark as unsupported
    assert results["exp.0.0"]["supported"] is False


def test_llm_verify_reason_preserved(monkeypatch):
    """E6: Reason from verifier is preserved."""
    from common.llm import Tracker

    def fake_chat(messages, tracker, format=None, **kwargs):
        return SimpleNamespace(content=json.dumps({
            "verdicts": [
                {"claim_id": "exp.0.0", "supported": False, "reason": "adds 'for executives'"},
            ]
        }))

    monkeypatch.setattr(grounding, "chat", fake_chat)

    items = [
        {"claim_id": "exp.0.0", "claim": Claim(text="A", sources=["S1"]), "cited_text": "A text"},
    ]

    cache = {}
    results = llm_verify(items, Tracker(), cache)

    assert results["exp.0.0"]["reason"] == "adds 'for executives'"


def test_llm_verify_uses_correct_format(monkeypatch):
    """E7: Uses correct schema and system prompt."""
    from common.llm import Tracker
    from common.schemas import VerifyBatch, llm_schema

    calls = []

    def fake_chat(messages, tracker, format=None, **kwargs):
        calls.append({"messages": messages, "format": format})
        return SimpleNamespace(content=json.dumps({"verdicts": []}))

    monkeypatch.setattr(grounding, "chat", fake_chat)

    items = [
        {"claim_id": "exp.0.0", "claim": Claim(text="A", sources=["S1"]), "cited_text": "A text"},
    ]

    llm_verify(items, Tracker(), {})

    assert len(calls) == 1
    assert calls[0]["messages"][0]["content"] == grounding.VERIFIER_SYSTEM
    # Check format is the schema
    expected_schema = llm_schema(VerifyBatch)
    assert calls[0]["format"] == expected_schema
@pytest.mark.parametrize("claim,source", [
    ("Saved 40%", "Saved 140%"),
    ("Saved $40", "Saved $400"),
    ("Processed 4 records", "Processed 40 records"),
    ("Improved by 3.5%", "Improved by 13.5%"),
])
def test_number_check_rejects_partial_numeric_matches(claim, source):
    assert any(p.startswith("new_number:") for p in rule_check(claim, source, ""))
    assert rule_check(claim, claim, "") == []


@pytest.mark.parametrize("claim,source,flag", [
    ("Built world-class reports", "Built reports", True),
    ("Built cutting-edge reports", "Built reports", True),
    ("Led reporting", "Maintained ledger files", True),
    ("Maintained ledger files", "Maintained files", False),
    ("Spearheaded reporting", "Built reports", True),
    ("Built world-class reports", "Created world-class reports", False),
    ("Led reporting", "Led weekly reporting", False),
])
def test_inflation_matches_words_and_inflections(claim, source, flag):
    assert any(p.startswith("inflated:") for p in rule_check(claim, source, "")) is flag


def test_skill_sentence_position_is_per_occurrence():
    job = "Tableau experience required. Familiarity with Tableau is preferred."
    assert "Tableau" in skill_terms(job)
    assert "new_skill: Tableau" in rule_check("Used Tableau", "Used Excel", job)
    assert "Tableau" not in skill_terms("Tableau experience required.")
    terms = skill_terms("Need SQL, Tableau and Power BI. Use C#, C++ and Node.js.")
    assert {"SQL", "Tableau", "Power BI", "C#", "C++", "Node.js"} <= terms
    assert "SQL Tableau" not in terms


@pytest.mark.parametrize("source_section", ["experience", "skills", "education"])
def test_experience_citations_require_resolved_job(source_section):
    from common.spans import Span

    candidate = {"experience": [{"employer": "Alpha Company", "title": "Analyst"}]}
    resume = GroundedResume.model_validate({"name": "Alex Example", "experience": [{
        "employer": "Alpha Company", "title": "Analyst",
        "bullets": [{"text": "Built reports", "sources": ["S1"]}],
    }]})
    spans = [Span("S1", "Built reports", 0, 13, source_section, None, "content")]
    assert "wrong_entry: S1" in check_citations(resume, spans, candidate)["exp.0.0"]
