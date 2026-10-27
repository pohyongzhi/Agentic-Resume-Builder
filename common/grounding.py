"""W4 evidence-grounding: citation checks, rule-based verification, and LLM verification."""
import json
import re

from common.harness import _in_text, _norm
from common.llm import chat
from common.schemas import Claim, GroundedResume, VerifyBatch, llm_schema
from common.spans import Span

MAX_SOURCES_PER_CLAIM = 3

INFLATION_TERMS = (
    "led", "lead", "spearhead", "head", "own", "expert", "advanced", "extensive",
    "proven", "significant", "substantial", "strategic", "senior", "award",
    "world-class", "cutting-edge", "drastically", "dramatically"
)

VERIFIER_SYSTEM = """You check resume claims against the source text they cite. For each item, decide if the claim is
fully supported by its cited source text alone.
Supported: the claim says the same thing in different words, shortens it, or combines the cited
sources without adding meaning.
NOT supported: the claim adds any tool, skill, number, scope, seniority, outcome, purpose,
audience or result that the cited text does not state.
Judge only against the cited text, not general knowledge or the job description.
Return one verdict per claim_id with a one-sentence reason when unsupported."""


def iter_claims(resume: GroundedResume) -> list[dict]:
    """Return all claims with their IDs and metadata.

    Returns list of {"claim_id", "section", "entry_index", "entry_label", "claim": Claim}
    in document order (spec §3.3).
    """
    claims = []

    # Summary
    for i, claim in enumerate(resume.summary):
        claims.append({
            "claim_id": f"summary.{i}",
            "section": "summary",
            "entry_index": None,
            "entry_label": "",
            "claim": claim,
        })

    # Experience
    for exp_idx, exp in enumerate(resume.experience):
        entry_label = f"{exp.title}, {exp.employer}"
        for bullet_idx, claim in enumerate(exp.bullets):
            claims.append({
                "claim_id": f"exp.{exp_idx}.{bullet_idx}",
                "section": "experience",
                "entry_index": exp_idx,
                "entry_label": entry_label,
                "claim": claim,
            })

    # Education
    for edu_idx, edu in enumerate(resume.education):
        entry_label = f"{edu.degree}, {edu.institution}"
        for detail_idx, claim in enumerate(edu.details):
            claims.append({
                "claim_id": f"edu.{edu_idx}.{detail_idx}",
                "section": "education",
                "entry_index": edu_idx,
                "entry_label": entry_label,
                "claim": claim,
            })

    # Projects
    for proj_idx, proj in enumerate(resume.projects):
        entry_label = proj.name
        for bullet_idx, claim in enumerate(proj.bullets):
            claims.append({
                "claim_id": f"proj.{proj_idx}.{bullet_idx}",
                "section": "projects",
                "entry_index": proj_idx,
                "entry_label": entry_label,
                "claim": claim,
            })

    return claims


def match_entry(section: str, draft_entry, candidate: dict) -> int | None:
    """Match a draft entry to a candidate entry by normalized name/employer/institution.

    Returns the index of the matching candidate entry, or None if no match.
    """
    if section == "experience":
        draft_employer = _norm(draft_entry.employer)
        draft_title = _norm(draft_entry.title)
        for i, entry in enumerate(candidate.get("experience", [])):
            cand_employer = _norm(entry.get("employer", ""))
            cand_title = _norm(entry.get("title", ""))
            # Try employer + title first
            if draft_employer == cand_employer and draft_title == cand_title:
                return i
        # Fall back to employer only
        for i, entry in enumerate(candidate.get("experience", [])):
            if draft_employer == _norm(entry.get("employer", "")):
                return i

    elif section == "education":
        draft_institution = _norm(draft_entry.institution)
        draft_degree = _norm(draft_entry.degree)
        for i, entry in enumerate(candidate.get("education", [])):
            cand_institution = _norm(entry.get("institution", ""))
            cand_degree = _norm(entry.get("degree", ""))
            # Try institution + degree first
            if draft_institution == cand_institution and draft_degree == cand_degree:
                return i
        # Fall back to institution only
        for i, entry in enumerate(candidate.get("education", [])):
            if draft_institution == _norm(entry.get("institution", "")):
                return i

    elif section == "projects":
        draft_name = _norm(draft_entry.name)
        for i, entry in enumerate(candidate.get("projects", [])):
            if draft_name == _norm(entry.get("name", "")):
                return i

    return None


def check_citations(
    resume: GroundedResume,
    spans: list[Span],
    candidate: dict,
    max_sources: int = MAX_SOURCES_PER_CLAIM
) -> dict[str, list[str]]:
    """Check citation validity for all claims.

    Returns dict of claim_id -> list of problem strings (empty = passed).
    Problem codes: no_citation, bad_citation, too_many_sources, wrong_entry, wrong_section.
    """
    problems = {}
    all_claims = iter_claims(resume)
    span_by_id = {s.id: s for s in spans}

    for item in all_claims:
        claim_id = item["claim_id"]
        claim = item["claim"]
        section = item["section"]
        entry_index = item["entry_index"]

        issues = []

        # Rule: no_citation
        if not claim.sources:
            issues.append("no_citation")

        # Rule: too_many_sources
        if len(claim.sources) > max_sources:
            issues.append("too_many_sources")

        # Check each cited span
        for source_id in claim.sources:
            span = span_by_id.get(source_id)
            # Rule: bad_citation (unknown ID or non-citeable span)
            if span is None:
                issues.append(f"bad_citation: {source_id}")
                continue
            if span.kind in ("heading", "entry_header") or span.section == "header":
                issues.append(f"bad_citation: {source_id}")
                continue

            # Rule: wrong_section (education/projects only; summary is exempt)
            if section in ("education", "projects"):
                if span.section != section:
                    issues.append(f"wrong_section: {source_id}")

            # Rule: wrong_entry (experience/education/projects)
            if section == "experience" and entry_index is not None:
                # Match the draft entry to a candidate entry
                draft_entry = resume.experience[entry_index]
                matched_index = match_entry("experience", draft_entry, candidate)
                if matched_index is not None:
                    # If the span belongs to a different entry, flag it
                    if span.section == "experience" and span.entry is not None and span.entry != matched_index:
                        issues.append(f"wrong_entry: {source_id}")

            elif section == "education" and entry_index is not None:
                draft_entry = resume.education[entry_index]
                matched_index = match_entry("education", draft_entry, candidate)
                if matched_index is not None:
                    if span.section == "education" and span.entry is not None and span.entry != matched_index:
                        issues.append(f"wrong_entry: {source_id}")

            elif section == "projects" and entry_index is not None:
                draft_entry = resume.projects[entry_index]
                matched_index = match_entry("projects", draft_entry, candidate)
                if matched_index is not None:
                    if span.section == "projects" and span.entry is not None and span.entry != matched_index:
                        issues.append(f"wrong_entry: {source_id}")

        problems[claim_id] = issues

    return problems


def skill_terms(job_text: str) -> set[str]:
    """Extract skill-like terms from job description text.

    Returns set of terms that look like skills: ALL-CAPS, contains /.+#., or Capitalized.
    Ignores generic capitalized words and sentence-initial capitals.
    """
    terms = set()

    # Split into sentences to detect sentence-initial words
    sentences = re.split(r'[.!?]+\s+', job_text)
    sentence_starts = set()
    for sent in sentences:
        # Split on whitespace and punctuation, take first word
        words = re.findall(r'\b[A-Za-z]+', sent)
        if words:
            sentence_starts.add(words[0])

    # Better tokenization: split on whitespace and most punctuation, but preserve /, +, #, .
    # This will capture "Tableau" even after a comma
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#./]*", job_text)

    generic_lower = {"i", "we", "the", "you", "our", "singapore", "and", "or", "with",
                     "january", "february", "march", "april", "may", "june",
                     "july", "august", "september", "october", "november", "december",
                     "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}

    i = 0
    while i < len(tokens):
        token = tokens[i]

        # ALL-CAPS (≥ 2 letters)
        if token.isupper() and len(token) >= 2:
            # Add this token
            terms.add(token)
            # Try to build a phrase with following skill-like tokens
            phrase_tokens = [token]
            j = i + 1
            while j < len(tokens):
                next_tok = tokens[j]
                # Check if next token is also skill-like
                is_caps = next_tok.isupper() and len(next_tok) >= 2
                is_special = re.search(r'[+#./]', next_tok[:-1]) or '/' in next_tok
                is_capitalized = (next_tok[0].isupper() and next_tok not in sentence_starts and
                                  _norm(next_tok) not in generic_lower)
                if is_caps or is_special or is_capitalized:
                    phrase_tokens.append(next_tok)
                    # Also add this individual token
                    if is_capitalized or is_caps:
                        terms.add(next_tok)
                    j += 1
                else:
                    break
            # Add the phrase if we merged multiple tokens
            if len(phrase_tokens) > 1:
                terms.add(" ".join(phrase_tokens))
            i = j if j > i + 1 else i + 1
            continue

        # Contains special chars inside word (not just at end)
        if re.search(r'[+#./]', token[:-1]) or '/' in token:
            terms.add(token)
            # Try to build a phrase
            phrase_tokens = [token]
            j = i + 1
            while j < len(tokens) and (re.search(r'[+#./]', tokens[j][:-1]) or '/' in tokens[j]):
                phrase_tokens.append(tokens[j])
                terms.add(tokens[j])
                j += 1
            if len(phrase_tokens) > 1:
                terms.add(" ".join(phrase_tokens))
            i = j if j > i + 1 else i + 1
            continue

        # Capitalized (but not sentence-initial or generic)
        if token[0].isupper() and token not in sentence_starts:
            norm_token = _norm(token)
            if norm_token not in generic_lower:
                terms.add(token)
                # Try to build a phrase with consecutive capitalized tokens
                phrase_tokens = [token]
                j = i + 1
                while j < len(tokens):
                    next_tok = tokens[j]
                    if next_tok[0].isupper() and next_tok not in sentence_starts:
                        norm_next = _norm(next_tok)
                        if norm_next not in generic_lower:
                            phrase_tokens.append(next_tok)
                            terms.add(next_tok)
                            j += 1
                        else:
                            break
                    else:
                        break
                if len(phrase_tokens) > 1:
                    terms.add(" ".join(phrase_tokens))
                i = j if j > i + 1 else i + 1
                continue

        i += 1

    return terms


def rule_check(
    claim_text: str,
    cited_text: str,
    job_text: str,
    inflation_terms=INFLATION_TERMS
) -> list[str]:
    """Run deterministic rule checks on a claim.

    Returns list of problem strings: "new_number: X", "new_skill: Y", "inflated: Z".
    """
    problems = []

    # Rule: new_number
    # Extract all numbers from claim (ignore 4-digit years 1950-2035)
    claim_numbers = re.findall(r'\$?\d[\d,]*(?:\.\d+)?%?', claim_text)
    claim_numbers = [n for n in claim_numbers if not (re.match(r'^\d{4}$', n) and 1950 <= int(n) <= 2035)]

    for num in claim_numbers:
        if num not in cited_text:
            problems.append(f"new_number: {num}")

    # Rule: new_skill
    skills = skill_terms(job_text)
    claim_norm = _norm(claim_text)
    cited_norm = _norm(cited_text)

    for skill in skills:
        # Check with plural tolerance (s at end)
        skill_base = skill.rstrip('s').rstrip('S')
        skill_pattern_claim = skill_base.lower()
        skill_pattern_cited = skill_base.lower()

        # If skill appears in claim but not in cited text
        if _in_text(skill, claim_norm) and not _in_text(skill, cited_norm):
            problems.append(f"new_skill: {skill}")

    # Rule: inflated
    for stem in inflation_terms:
        # Check if any word in claim starts with this stem
        claim_words = re.findall(r'\b\w+', claim_text.lower())
        cited_words = re.findall(r'\b\w+', cited_text.lower())

        for word in claim_words:
            if word.startswith(stem):
                # Check if any word in cited_text starts with this stem
                if not any(cw.startswith(stem) for cw in cited_words):
                    problems.append(f"inflated: {word}")
                    break  # Only flag once per stem

    return problems


def drop_unsupported(resume: GroundedResume, problems: dict[str, list[str]]) -> GroundedResume:
    """Remove claims that have problems, returning a new GroundedResume."""
    # Build a new resume with only supported claims
    new_summary = [claim for i, claim in enumerate(resume.summary)
                   if not problems.get(f"summary.{i}")]

    new_experience = []
    for exp_idx, exp in enumerate(resume.experience):
        new_bullets = [claim for bullet_idx, claim in enumerate(exp.bullets)
                       if not problems.get(f"exp.{exp_idx}.{bullet_idx}")]
        # Keep the entry even if all bullets are dropped
        new_experience.append(type(exp)(
            employer=exp.employer,
            title=exp.title,
            start=exp.start,
            end=exp.end,
            bullets=new_bullets,
        ))

    new_education = []
    for edu_idx, edu in enumerate(resume.education):
        new_details = [claim for detail_idx, claim in enumerate(edu.details)
                       if not problems.get(f"edu.{edu_idx}.{detail_idx}")]
        new_education.append(type(edu)(
            institution=edu.institution,
            degree=edu.degree,
            start=edu.start,
            end=edu.end,
            details=new_details,
        ))

    new_projects = []
    for proj_idx, proj in enumerate(resume.projects):
        new_bullets = [claim for bullet_idx, claim in enumerate(proj.bullets)
                       if not problems.get(f"proj.{proj_idx}.{bullet_idx}")]
        new_projects.append(type(proj)(
            name=proj.name,
            description=proj.description,
            bullets=new_bullets,
        ))

    return GroundedResume(
        name=resume.name,
        contact=resume.contact,
        summary=new_summary,
        experience=new_experience,
        education=new_education,
        projects=new_projects,
        skills=resume.skills,
        certifications=resume.certifications,
    )


def filter_skills(items: list[str], parsed: list[str], raw_text: str) -> tuple[list[str], list[str]]:
    """Filter skills/certifications, keeping only those in parsed list or raw text.

    Returns (kept, dropped) tuples.
    """
    kept = []
    dropped = []
    parsed_norm = {_norm(s) for s in parsed}
    raw_norm = _norm(raw_text)

    for item in items:
        if _norm(item) in parsed_norm or _in_text(item, raw_norm):
            kept.append(item)
        else:
            dropped.append(item)

    return kept, dropped


def to_generated(resume: GroundedResume) -> dict:
    """Convert GroundedResume to GeneratedResume dict, stripping citations."""
    return {
        "name": resume.name,
        "contact": resume.contact.model_dump() if hasattr(resume.contact, 'model_dump') else resume.contact,
        "summary": " ".join(claim.text for claim in resume.summary),
        "experience": [
            {
                "employer": exp.employer,
                "title": exp.title,
                "start": exp.start,
                "end": exp.end,
                "bullets": [claim.text for claim in exp.bullets],
            }
            for exp in resume.experience
        ],
        "education": [
            {
                "institution": edu.institution,
                "degree": edu.degree,
                "start": edu.start,
                "end": edu.end,
                "details": [claim.text for claim in edu.details],
            }
            for edu in resume.education
        ],
        "projects": [
            {
                "name": proj.name,
                "description": proj.description,
                "bullets": [claim.text for claim in proj.bullets],
            }
            for proj in resume.projects
        ],
        "skills": resume.skills,
        "certifications": resume.certifications,
    }


# ========== LLM Verifier ==========

def cache_key(claim: Claim) -> str:
    """Generate cache key for a claim (text + sorted sources)."""
    return json.dumps([claim.text.strip(), sorted(claim.sources)])


def llm_verify(items: list[dict], tracker, cache: dict) -> dict[str, dict]:
    """Verify claims using the LLM verifier.

    items: [{"claim_id", "claim": Claim, "cited_text"}]
    cache: dict mapping cache_key -> verdict dict

    Returns dict of claim_id -> {"supported": bool, "reason": str}.
    """
    # Separate cached and uncached items
    uncached = []
    results = {}

    for item in items:
        claim_id = item["claim_id"]
        claim = item["claim"]
        key = cache_key(claim)

        if key in cache:
            results[claim_id] = cache[key]
        else:
            uncached.append(item)

    # If all cached, return early
    if not uncached:
        return results

    # Prepare batch for LLM
    batch_input = [
        {
            "claim_id": item["claim_id"],
            "claim": item["claim"].text,
            "cited_text": item["cited_text"],
        }
        for item in uncached
    ]

    # Call LLM
    try:
        msg = chat(
            [
                {"role": "system", "content": VERIFIER_SYSTEM},
                {"role": "user", "content": json.dumps(batch_input, indent=2, ensure_ascii=False)},
            ],
            tracker,
            format=llm_schema(VerifyBatch),
        )

        batch = VerifyBatch.model_validate_json(msg.content or "{}")
        verdicts_by_id = {v.claim_id: {"supported": v.supported, "reason": v.reason}
                          for v in batch.verdicts}
    except Exception:
        # On error, treat all uncached items as unsupported
        verdicts_by_id = {}

    # Process uncached items
    for item in uncached:
        claim_id = item["claim_id"]
        claim = item["claim"]

        if claim_id in verdicts_by_id:
            verdict = verdicts_by_id[claim_id]
        else:
            # Missing or malformed verdict
            verdict = {"supported": False, "reason": "verifier gave no verdict"}

        # Cache and store
        key = cache_key(claim)
        cache[key] = verdict
        results[claim_id] = verdict

    return results
