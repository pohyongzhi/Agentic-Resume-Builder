"""Tests for W4 span splitting (test plan §2)."""
import pytest

from common.harness import _norm
from common.spans import citeable, render_spans_for_prompt, split_spans, spans_by_id

# PDF-shaped fixture (test plan §2, fixture from spec)
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


# ========== Markdown resume tests ==========

def test_span_ids_sequential_no_gaps(candidate):
    """2.1: Span IDs are S1..Sn in order, no gaps."""
    spans = split_spans(candidate["raw_text"], candidate)
    ids = [s.id for s in spans]
    expected = [f"S{i}" for i in range(1, len(ids) + 1)]
    assert ids == expected


def test_experience_content_span(candidate):
    """2.2: Experience bullet is content, no bullet marker."""
    spans = split_spans(candidate["raw_text"], candidate)
    span = next((s for s in spans if "Cleaned and merged shipment data" in s.text), None)
    assert span is not None
    assert span.kind == "content"
    assert span.section == "experience"
    assert span.entry == 0
    assert not span.text.startswith("-")


def test_entry_header_not_citeable(candidate):
    """2.3: Entry header line is not citeable."""
    spans = split_spans(candidate["raw_text"], candidate)
    # The line "**Business Analyst Intern, Example Logistics Pte Ltd** (May 2025 - Aug 2025)"
    # Should be an entry_header
    entry_headers = [s for s in spans if s.kind == "entry_header"]
    assert len(entry_headers) > 0
    assert not any(s in citeable(spans) for s in entry_headers)


def test_headings_not_citeable(candidate):
    """2.4: Section headings are not citeable."""
    spans = split_spans(candidate["raw_text"], candidate)
    headings = [s for s in spans if s.kind == "heading"]
    assert len(headings) > 0
    assert not any(s in citeable(spans) for s in headings)


def test_skills_line_citeable(candidate):
    """2.5: Skills line is citeable."""
    spans = split_spans(candidate["raw_text"], candidate)
    skills_spans = [s for s in spans if s.section == "skills" and s.kind == "content"]
    assert len(skills_spans) > 0
    assert all(s in citeable(spans) for s in skills_spans)


def test_header_section_not_citeable(candidate):
    """2.6: Name + email lines are in header section, not citeable."""
    spans = split_spans(candidate["raw_text"], candidate)
    header_spans = [s for s in spans if s.section == "header"]
    assert len(header_spans) > 0
    assert not any(s in citeable(spans) for s in header_spans)


def test_offset_invariant(candidate):
    """2.7: Offset invariant - span text is contained in raw_text slice."""
    spans = split_spans(candidate["raw_text"], candidate)
    raw_text = candidate["raw_text"]
    for span in spans:
        # Extract the raw slice
        raw_slice = raw_text[span.start:span.end]
        # Normalize both (remove bullet markers, collapse whitespace)
        normalized_slice = _norm(" ".join(raw_slice.split()))
        normalized_text = _norm(span.text)
        # The span text should be contained in the slice
        assert normalized_text in normalized_slice or normalized_slice in normalized_text, \
            f"Span {span.id} text '{span.text}' not in raw slice '{raw_slice}'"


# ========== PDF-shaped resume tests ==========

def test_wrapped_bullet_first(PDF_CANDIDATE=PDF_CANDIDATE):
    """2.8: Wrapped bullet is joined into one span."""
    spans = split_spans(PDF_RAW, PDF_CANDIDATE)
    matches = [s for s in spans if "Built automated reporting workflows in Excel and Python" in s.text]
    assert len(matches) == 1
    span = matches[0]
    assert "reducing manual effort for the operations team" in span.text


def test_wrapped_bullet_second(PDF_CANDIDATE=PDF_CANDIDATE):
    """2.9: Second wrapped bullet is joined."""
    spans = split_spans(PDF_RAW, PDF_CANDIDATE)
    matches = [s for s in spans if "Gathered requirements from business users" in s.text]
    assert len(matches) == 1
    assert "wrote user stories for dashboard enhancements" in matches[0].text


def test_entry_assignment(PDF_CANDIDATE=PDF_CANDIDATE):
    """2.10: Bullets assigned to correct entries."""
    spans = split_spans(PDF_RAW, PDF_CANDIDATE)
    bank_bullets = [s for s in spans if s.section == "experience" and s.entry == 0 and s.kind == "content"]
    agency_bullets = [s for s in spans if s.section == "experience" and s.entry == 1 and s.kind == "content"]
    assert len(bank_bullets) == 2
    assert len(agency_bullets) == 2


def test_entry_headers_identified(PDF_CANDIDATE=PDF_CANDIDATE):
    """2.11: Entry headers are identified."""
    spans = split_spans(PDF_RAW, PDF_CANDIDATE)
    entry_headers = [s for s in spans if s.kind == "entry_header"]
    # Should have headers for: Data Analyst Intern, Operations Assistant, Example University
    assert len(entry_headers) >= 2
    # Check that they contain the expected text
    header_texts = [s.text for s in entry_headers]
    assert any("Example Bank" in t for t in header_texts)
    assert any("Example University" in t for t in header_texts)


def test_hyphen_artifacts_fixed(PDF_CANDIDATE=PDF_CANDIDATE):
    """2.12: Hyphen artifacts are fixed."""
    spans = split_spans(PDF_RAW, PDF_CANDIDATE)
    # Find the span with "Excel -based" in original -> should be "Excel-based"
    excel_spans = [s for s in spans if "Excel" in s.text and "tracker" in s.text]
    assert len(excel_spans) == 1
    assert "Excel-based" in excel_spans[0].text
    assert "self-service" in excel_spans[0].text


def test_about_me_paragraph(PDF_CANDIDATE=PDF_CANDIDATE):
    """2.13: About Me paragraph is one citeable summary span."""
    spans = split_spans(PDF_RAW, PDF_CANDIDATE)
    summary_spans = [s for s in spans if s.section == "summary" and s.kind == "content"]
    assert len(summary_spans) == 1
    span = summary_spans[0]
    assert "final-year Business Analytics student" in span.text
    assert "improving everyday processes" in span.text
    assert len(span.text) <= 300


def test_degree_line_citeable(PDF_CANDIDATE=PDF_CANDIDATE):
    """2.14: Degree line is citeable with correct section and entry."""
    spans = split_spans(PDF_RAW, PDF_CANDIDATE)
    degree_spans = [s for s in spans if "Bachelor of Science (Business Analytics)" in s.text and s.kind == "content"]
    assert len(degree_spans) == 1
    span = degree_spans[0]
    assert span.section == "education"
    assert span.entry == 0
    assert "GPA: 3.6/4.0" in span.text


def test_contact_line_not_bullet(PDF_CANDIDATE=PDF_CANDIDATE):
    """2.15: Contact line with • separators is header, not treated as bullet."""
    spans = split_spans(PDF_RAW, PDF_CANDIDATE)
    # Contact line: "Singapore • jordan.lim@example.com • (+65) 9000 0000"
    contact_spans = [s for s in spans if "jordan.lim@example.com" in s.text]
    if contact_spans:
        # Should be in header section
        assert all(s.section == "header" for s in contact_spans)
        assert not any(s in citeable(spans) for s in contact_spans)


def test_long_block_split():
    """2.16: Long paragraph (>300 chars) is split into sentences."""
    long_text = (
        "Heading\n"
        "Long sentence one that goes on and on with many words to exceed the character limit for testing purposes. "
        "Long sentence two that also has a lot of words and contributes to the total length being over three hundred characters. "
        "Long sentence three which is the final part of this very lengthy paragraph that definitely exceeds the maximum span character limit."
    )
    candidate_long = {"name": "Test", "experience": [], "education": [], "projects": [], "skills": []}
    spans = split_spans(long_text, candidate_long, max_chars=300)

    # The long paragraph should be split into multiple spans
    content_spans = [s for s in spans if s.kind == "content"]
    # Should have at least 2 spans (sentences)
    assert len(content_spans) >= 2
    # Each should respect the invariant
    for span in content_spans:
        normalized_slice = _norm(long_text[span.start:span.end])
        normalized_text = _norm(span.text)
        assert normalized_text in normalized_slice or normalized_slice in normalized_text


def test_empty_text():
    """2.17: Empty text returns empty list."""
    spans = split_spans("", {"name": "Test", "experience": [], "education": [], "projects": [], "skills": []})
    assert spans == []


# ========== Prompt rendering tests ==========

def test_render_spans_for_prompt_structure(PDF_CANDIDATE=PDF_CANDIDATE):
    """2.18: Rendered prompt has correct structure with entry headers."""
    spans = split_spans(PDF_RAW, PDF_CANDIDATE)
    rendered = render_spans_for_prompt(spans, PDF_CANDIDATE)
    assert "SOURCE SPANS (cite these by ID):" in rendered
    assert "== EXPERIENCE: Data Analyst Intern, Example Bank (May 2026 - Present) ==" in rendered
    # Check that span IDs are rendered
    assert "[S" in rendered
    # Check that the Bank spans appear before the next section
    lines = rendered.split("\n")
    bank_header_idx = next(i for i, line in enumerate(lines) if "Example Bank" in line)
    # Find spans under this header (before next ==)
    bank_spans = []
    for i in range(bank_header_idx + 1, len(lines)):
        if lines[i].startswith("=="):
            break
        if lines[i].startswith("[S"):
            bank_spans.append(lines[i])
    assert len(bank_spans) == 2  # Two Bank bullets


def test_render_spans_excludes_non_citeable(PDF_CANDIDATE=PDF_CANDIDATE):
    """2.19: Rendered prompt does not include entry_header or heading spans."""
    spans = split_spans(PDF_RAW, PDF_CANDIDATE)
    rendered = render_spans_for_prompt(spans, PDF_CANDIDATE)
    # Should not have any heading or entry_header span text as [S..] lines
    entry_headers = [s for s in spans if s.kind == "entry_header"]
    headings = [s for s in spans if s.kind == "heading"]
    for span in entry_headers + headings:
        # The span text might appear in section headers (==), but not as a [S..] citation line
        # Check that span.id doesn't appear in the rendered output
        assert f"[{span.id}]" not in rendered
