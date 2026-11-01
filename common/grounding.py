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
        if proj.description.text.strip():
            claims.append({
                "claim_id": f"proj.{proj_idx}.description",
                "section": "projects",
                "entry_index": proj_idx,
                "entry_label": entry_label,
                "claim": proj.description,
            })
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
                    if span.section != "experience" or span.entry != matched_index:
                        issues.append(f"wrong_entry: {source_id}")

            elif section == "education" and entry_index is not None:
                draft_entry = resume.education[entry_index]
                matched_index = match_entry("education", draft_entry, candidate)
                if matched_index is not None:
                    if span.section == "education" and span.entry != matched_index:
                        issues.append(f"wrong_entry: {source_id}")

            elif section == "projects" and entry_index is not None:
                draft_entry = resume.projects[entry_index]
                matched_index = match_entry("projects", draft_entry, candidate)
                if matched_index is not None:
                    if span.section == "projects" and span.entry != matched_index:
                        issues.append(f"wrong_entry: {source_id}")

        problems[claim_id] = issues

    return problems


def skill_terms(job_text: str) -> set[str]:
    """Find skill terms by occurrence, retaining technical punctuation and phrases."""
    generic = {
        "i", "we", "the", "you", "our", "singapore", "and", "or", "with",
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    }
    terms = set()
    phrase = []
    previous_end = 0
    previous_ended_sentence = False

    def flush():
        if len(phrase) > 1:
            terms.add(" ".join(phrase))
        phrase.clear()

    for match in re.finditer(r"[A-Za-z][A-Za-z0-9+#./]*", job_text):
        raw_token = match.group()
        token = raw_token.rstrip(".")
        gap = job_text[previous_end:match.start()]
        sentence_start = previous_end == 0 or previous_ended_sentence or bool(re.search(r"[.!?]", gap))
        # Punctuation separates listed skills; only whitespace can join a phrase.
        if sentence_start or gap.strip():
            flush()
        letters = re.sub(r"[^A-Za-z]", "", token)
        acronym = len(letters) >= 2 and (letters.isupper() or
                   (letters.endswith("s") and len(letters) > 2 and letters[:-1].isupper()))
        technical = bool(re.search(r"[+#/]", token) or "." in token)
        capitalized = token[0].isupper() and not sentence_start
        if token.lower() not in generic and (acronym or technical or capitalized):
            terms.add(token)
            phrase.append(token)
        else:
            flush()
        previous_end = match.end()
        previous_ended_sentence = raw_token.endswith(".")
    flush()
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

    cited_numbers = set(re.findall(r'\$?\d[\d,]*(?:\.\d+)?%?', cited_text))
    for num in claim_numbers:
        if num not in cited_numbers:
            problems.append(f"new_number: {num}")

    # Rule: new_skill
    skills = skill_terms(job_text)
    claim_norm = _norm(claim_text)
    cited_norm = _norm(cited_text)

    for skill in sorted(skills):
        # Strip a plural suffix, preserving acronym endings such as AWS.
        base = skill[:-1] if skill.endswith("s") and not skill.endswith("ss") else skill
        variants = {base, base + "s", skill}
        in_claim = any(_in_text(term, claim_norm) for term in variants)
        in_source = any(_in_text(term, cited_norm) for term in variants)
        if in_claim and not in_source:
            problems.append(f"new_skill: {skill}")

    # Rule: inflated
    inflections = {
        "led": {"led"},
        "lead": {"lead", "leads", "leading", "leader", "leaders", "leadership"},
        "spearhead": {"spearhead", "spearheads", "spearheaded", "spearheading"},
        "head": {"head", "heads", "headed", "heading"},
        "own": {"own", "owns", "owned", "owning", "owner", "owners", "ownership"},
        "expert": {"expert", "experts", "expertise", "expertly"},
        "award": {"award", "awards", "awarded", "awarding", "award-winning"},
        "senior": {"senior", "seniors", "seniority"},
    }
    claim_words = re.findall(r"[a-z]+(?:-[a-z]+)*", claim_text.lower())
    cited_words = set(re.findall(r"[a-z]+(?:-[a-z]+)*", cited_text.lower()))
    for stem in inflation_terms:
        forms = inflections.get(stem, {stem, stem + "ly"})
        added = next((word for word in claim_words if word in forms), None)
        if added and not cited_words.intersection(forms):
            problems.append(f"inflated: {added}")

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
            description=Claim(text="") if problems.get(f"proj.{proj_idx}.description") else proj.description,
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
                "description": proj.description.text,
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
