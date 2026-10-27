"""Span splitting for W4 evidence-grounded generation.

Converts raw resume text into numbered, citeable spans with section and entry metadata.
Handles PDF extraction artifacts: wrapped bullets, hyphenation, etc.
"""
import re
from dataclasses import asdict, dataclass

from common.harness import _in_text, _norm

SPAN_MAX_CHARS = 300

# Known section headings (case-insensitive, normalized)
HEADING_MAP = {
    "summary": "summary",
    "about me": "summary",
    "profile": "summary",
    "objective": "summary",
    "experience": "experience",
    "work experience": "experience",
    "employment": "experience",
    "professional experience": "experience",
    "education": "education",
    "projects": "projects",
    "skills": "skills",
    "skills & interests": "skills",
    "technical skills": "skills",
    "leadership": "other",
    "activities": "other",
    "awards": "other",
    "certifications": "other",
    "volunteering": "other",
}

# Generic capitalized words to ignore in skill detection
GENERIC_CAPS = {
    "i", "we", "the", "you", "our", "singapore",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}


@dataclass(frozen=True)
class Span:
    id: str
    text: str
    start: int
    end: int
    section: str
    entry: int | None
    kind: str  # "heading" | "entry_header" | "content"

    def to_dict(self) -> dict:
        return asdict(self)


def _is_bullet(line: str) -> bool:
    """Check if line starts with a bullet marker."""
    stripped = line.lstrip()
    if not stripped:
        return False
    # Bullet markers: •, -, *, –, ▪, ●, or numbered (1., 2), etc.)
    if stripped[0] in "•-*–▪●":
        return True
    # Numbered bullets: \d+[.)]
    return bool(re.match(r"^\d+[.)]", stripped))


def _strip_bullet(line: str) -> str:
    """Remove bullet marker from line, preserving other text."""
    stripped = line.lstrip()
    if not stripped:
        return line
    # Remove bullet character
    if stripped[0] in "•-*–▪●":
        return stripped[1:].lstrip()
    # Remove numbered bullet
    match = re.match(r"^(\d+[.)])\s*", stripped)
    if match:
        return stripped[len(match.group(1)):].lstrip()
    return line


def _heading_section(line: str) -> str | None:
    """Return section name if line is a known heading, else None."""
    text = line.strip()
    # Markdown headings
    if text.startswith("#"):
        text = text.lstrip("#").strip()
    # Short lines (≤ 4 words, no full stop)
    words = text.split()
    if len(words) > 4 or text.endswith("."):
        return None
    normalized = _norm(text)
    return HEADING_MAP.get(normalized)


def _has_date_range(line: str) -> bool:
    """Check if line contains a date range pattern."""
    # Pattern from spec: month + year, year ranges, "expected YYYY", "Present", etc.
    pattern = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{4}|\b(19|20)\d{2}\b\s*[-–—]\s*((19|20)\d{2}|present|now|current)|expected\s+(19|20)\d{2}"
    return bool(re.search(pattern, line, re.IGNORECASE))


def _match_entry(line: str, section: str, candidate: dict, current_entry: int | None) -> int | None:
    """Match line to a candidate entry by name/employer/institution.

    Returns the entry index, or current_entry + 1 if line has a date but no name match.
    """
    if section not in ("experience", "education", "projects"):
        return None

    raw_norm = _norm(line)

    if section == "experience":
        entries = candidate.get("experience", [])
        for i, entry in enumerate(entries):
            employer = entry.get("employer", "")
            title = entry.get("title", "")
            if employer and _in_text(employer, raw_norm):
                return i
        # Has date but no name match: advance to next entry
        if _has_date_range(line):
            return (current_entry + 1) if current_entry is not None else 0

    elif section == "education":
        entries = candidate.get("education", [])
        for i, entry in enumerate(entries):
            institution = entry.get("institution", "")
            if institution and _in_text(institution, raw_norm):
                return i
        if _has_date_range(line):
            return (current_entry + 1) if current_entry is not None else 0

    elif section == "projects":
        entries = candidate.get("projects", [])
        for i, entry in enumerate(entries):
            name = entry.get("name", "")
            if name and _in_text(name, raw_norm):
                return i
        if _has_date_range(line):
            return (current_entry + 1) if current_entry is not None else 0

    return None


def _clean_text(text: str) -> str:
    """Clean span text: collapse whitespace and fix PDF hyphen artifacts."""
    # Fix hyphen artifacts: "Excel -based" → "Excel-based"
    text = re.sub(r"(\w) -(\w)", r"\1-\2", text)
    # Collapse whitespace
    return re.sub(r"\s+", " ", text).strip()


def _sentences(text: str, start_offset: int) -> list[tuple[str, int, int]]:
    """Split text into sentences, preserving offsets.

    Returns list of (sentence_text, start, end) tuples.
    """
    # Split on ". ", "! ", "? " followed by capital letter or "I'"
    # This is a simple heuristic; could be refined
    sentences = []
    pattern = r"([.!?]\s+)(?=[A-Z]|I')"
    parts = re.split(pattern, text)

    offset = start_offset
    current = ""
    for part in parts:
        current += part
        # If this is a punctuation + space marker, the sentence is complete
        if re.match(r"^[.!?]\s+$", part):
            if current.strip():
                end = offset + len(current)
                sentences.append((current.strip(), offset, end))
                offset = end
                current = ""

    # Add remaining text
    if current.strip():
        sentences.append((current.strip(), offset, offset + len(current)))

    # If we couldn't split, return the whole text as one sentence
    if not sentences:
        sentences = [(text, start_offset, start_offset + len(text))]

    return sentences


def split_spans(raw_text: str, candidate: dict, max_chars: int = SPAN_MAX_CHARS) -> list[Span]:
    """Split raw resume text into numbered, citeable spans.

    Implements the 10 rules from spec §4.1.
    """
    if not raw_text.strip():
        return []

    # Extract lines with offsets
    lines = []
    for match in re.finditer(r"[^\n]*\n?", raw_text):
        line_text = match.group(0)
        if line_text or match.start() < len(raw_text):  # Include empty lines
            lines.append((line_text, match.start(), match.end()))

    spans = []
    current_section = "header"
    current_entry = None
    current_block = None  # (text, start, end, section, entry, kind)

    for line_text, line_start, line_end in lines:
        # Rule 2: Blank lines end the current block
        if not line_text.strip():
            if current_block:
                spans.append(current_block)
                current_block = None
            continue

        # Rule 4: Check for section heading
        section_name = _heading_section(line_text)
        is_markdown_heading = line_text.strip().startswith("#")

        if section_name:
            # Known heading: change section
            if current_block:
                spans.append(current_block)
                current_block = None
            current_section = section_name
            current_entry = None
            # The heading itself becomes a span
            spans.append((
                _clean_text(line_text.lstrip("#").strip() if is_markdown_heading else line_text.strip()),
                line_start, line_end, current_section, None, "heading"
            ))
            continue
        elif is_markdown_heading:
            # Markdown heading but not a known section (e.g., "# Alex Tan")
            # Still a heading span, but doesn't change section
            if current_block:
                spans.append(current_block)
                current_block = None
            spans.append((
                _clean_text(line_text.lstrip("#").strip()),
                line_start, line_end, current_section, None, "heading"
            ))
            continue

        # Rule 5: Check for entry header
        if current_section in ("experience", "education", "projects"):
            entry_match = _match_entry(line_text, current_section, candidate, current_entry)
            if entry_match is not None:
                if current_block:
                    spans.append(current_block)
                    current_block = None
                current_entry = entry_match
                # Entry header becomes its own span
                spans.append((
                    _clean_text(line_text.strip()),
                    line_start, line_end, current_section, current_entry, "entry_header"
                ))
                continue

        # Rule 3: Bullet line starts a new block
        if _is_bullet(line_text):
            if current_block:
                spans.append(current_block)
            cleaned = _clean_text(_strip_bullet(line_text))
            current_block = (cleaned, line_start, line_end, current_section, current_entry, "content")
            continue

        # Rule 6: Continuation line or new content
        if current_block and current_block[5] in ("content",):
            # Continuation: join to current block with a space
            text, start, end, sec, ent, kind = current_block
            joined_text = text + " " + _clean_text(line_text.strip())
            current_block = (joined_text, start, line_end, sec, ent, kind)
        else:
            # New content block
            if current_block:
                spans.append(current_block)
            current_block = (
                _clean_text(line_text.strip()),
                line_start, line_end, current_section, current_entry, "content"
            )

    # Add final block
    if current_block:
        spans.append(current_block)

    # Rule 8: Split long content blocks into sentences
    final_spans = []
    for text, start, end, section, entry, kind in spans:
        if kind == "content" and len(text) > max_chars:
            # Split into sentences
            sents = _sentences(text, start)
            for sent_text, sent_start, sent_end in sents:
                final_spans.append((sent_text, sent_start, sent_end, section, entry, kind))
        else:
            final_spans.append((text, start, end, section, entry, kind))

    # Rule 10: Assign IDs S1...Sn in document order
    result = []
    for i, (text, start, end, section, entry, kind) in enumerate(final_spans, start=1):
        result.append(Span(
            id=f"S{i}",
            text=text,
            start=start,
            end=end,
            section=section,
            entry=entry,
            kind=kind,
        ))

    return result


def citeable(spans: list[Span]) -> list[Span]:
    """Return only citeable spans (content, not header/heading/entry_header)."""
    return [s for s in spans if s.kind == "content" and s.section != "header"]


def spans_by_id(spans: list[Span]) -> dict[str, Span]:
    """Return a dict mapping span ID to Span."""
    return {s.id: s for s in spans}


def render_spans_for_prompt(spans: list[Span], candidate: dict) -> str:
    """Render citeable spans grouped by section and entry for the creator prompt."""
    cite = citeable(spans)
    if not cite:
        return ""

    lines = []
    lines.append("SOURCE SPANS (cite these by ID):")
    lines.append("")

    # Group by section and entry
    by_section = {}
    for span in cite:
        key = (span.section, span.entry)
        by_section.setdefault(key, []).append(span)

    # Render in order: experience entries, education entries, projects entries, then summary/other

    # Experience
    experience_entries = candidate.get("experience", [])
    for i, exp in enumerate(experience_entries):
        key = ("experience", i)
        if key in by_section:
            title = exp.get("title", "")
            employer = exp.get("employer", "")
            start = exp.get("start", "")
            end = exp.get("end", "")
            date_str = f" ({start} - {end})" if start or end else ""
            lines.append(f"== EXPERIENCE: {title}, {employer}{date_str} ==")
            for span in by_section[key]:
                lines.append(f"[{span.id}] {span.text}")
            lines.append("")

    # Unassigned experience spans
    key = ("experience", None)
    if key in by_section:
        lines.append("== EXPERIENCE (unassigned) ==")
        for span in by_section[key]:
            lines.append(f"[{span.id}] {span.text}")
        lines.append("")

    # Education
    education_entries = candidate.get("education", [])
    for i, edu in enumerate(education_entries):
        key = ("education", i)
        if key in by_section:
            institution = edu.get("institution", "")
            degree = edu.get("degree", "")
            start = edu.get("start", "")
            end = edu.get("end", "")
            date_str = f" ({start} - {end})" if start or end else ""
            lines.append(f"== EDUCATION: {degree}, {institution}{date_str} ==")
            for span in by_section[key]:
                lines.append(f"[{span.id}] {span.text}")
            lines.append("")

    key = ("education", None)
    if key in by_section:
        lines.append("== EDUCATION (unassigned) ==")
        for span in by_section[key]:
            lines.append(f"[{span.id}] {span.text}")
        lines.append("")

    # Projects
    project_entries = candidate.get("projects", [])
    for i, proj in enumerate(project_entries):
        key = ("projects", i)
        if key in by_section:
            name = proj.get("name", "")
            lines.append(f"== PROJECTS: {name} ==")
            for span in by_section[key]:
                lines.append(f"[{span.id}] {span.text}")
            lines.append("")

    key = ("projects", None)
    if key in by_section:
        lines.append("== PROJECTS (unassigned) ==")
        for span in by_section[key]:
            lines.append(f"[{span.id}] {span.text}")
        lines.append("")

    # Summary
    key = ("summary", None)
    if key in by_section:
        lines.append("== SUMMARY ==")
        for span in by_section[key]:
            lines.append(f"[{span.id}] {span.text}")
        lines.append("")

    # Skills
    key = ("skills", None)
    if key in by_section:
        lines.append("== SKILLS ==")
        for span in by_section[key]:
            lines.append(f"[{span.id}] {span.text}")
        lines.append("")

    # Other
    key = ("other", None)
    if key in by_section:
        lines.append("== OTHER ==")
        for span in by_section[key]:
            lines.append(f"[{span.id}] {span.text}")
        lines.append("")

    return "\n".join(lines).rstrip()
