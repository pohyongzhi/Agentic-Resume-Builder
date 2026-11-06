"""Seeded input perturbations for robustness evaluation and their record-level checks.

Robustness here means how reliably a workflow keeps its performance, accuracy and expected
behaviour when its inputs are noisy, formatted unexpectedly or out of scope. Every variant is a
pure function of ``(pair, seed)``; ``mismatched_jd`` additionally reads the frozen dataset's JD
pool, and the chosen JD is recorded in the sample manifest.

Variant families (``config.ROBUSTNESS_VARIANT_VERSION``):

- ``baseline``: ``original`` (the saved generation; no new generation).
- ``noisy`` (in scope): ``typos`` (seeded low-rate typos in resume prose and JD prose, never in
  protected-field, keyword or capitalised tokens), ``pdf_extraction`` (resume as PDF text: no
  Markdown, hard line wraps, hyphenation splits, a page break with page numbers and a running
  header, irregular whitespace), ``web_clutter`` (JD wrapped in scraped careers-page text: nav,
  cookie banner, benefits, EEO and footer boilerplate).
- ``format`` (in scope): ``formatting`` and ``reordered_sections`` (unchanged from v1),
  ``plain_text`` (Markdown removed from resume and JD), ``heading_aliases`` (section headings
  renamed, e.g. "Experience" -> "Work History").
- ``out_of_scope``: ``mismatched_jd`` (a JD from a different category than the resume) and
  ``degenerate_jd`` (title only, a one-line posting, or non-job text; one mode per pair).

In-scope variants are scored against the clean original resume and clean JD/keywords and compared
with the original variant. Out-of-scope variants are checked for expected behaviour instead.
"""
import csv
import hashlib
import json
import random
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import config
from eval import metrics

ORIGINAL = "original"
VARIANT_FAMILIES = {
    "original": "baseline",
    "typos": "noisy",
    "pdf_extraction": "noisy",
    "web_clutter": "noisy",
    "formatting": "format",
    "reordered_sections": "format",
    "plain_text": "format",
    "heading_aliases": "format",
    "mismatched_jd": "out_of_scope",
    "degenerate_jd": "out_of_scope",
}
VARIANTS = tuple(VARIANT_FAMILIES)
LEGACY_VARIANTS = ("original", "formatting", "reordered_sections")
FAMILIES = ("noisy", "format", "out_of_scope")
OUT_OF_SCOPE_FAMILIES = {"out_of_scope"}
DEGENERATE_MODES = ("title_only", "very_short", "non_job_text")
ROBUSTNESS_METRICS = (
    "faithfulness", "judge_overall_mean", "kw_after", "sim_after",
)

# Metric -> direction (+1: higher is better, -1: lower is better) for in-scope paired deltas.
DELTA_METRICS = {
    "task_success": 1,
    "faithfulness": 1,
    "unsupported_claim_rate": -1,
    "protected_field_violation": -1,
    "fabrication_bait_violation": -1,
    "requirement_coverage": 1,
    "kw_after": 1,
    "sim_after": 1,
    "judge_overall_mean": 1,
}
# Stability criteria: binary metrics must not flip to worse; continuous metrics must not degrade
# by more than ``config.ROBUSTNESS_STABILITY_TOLERANCE``. The 1-5 judge scale is excluded.
STABILITY_BINARY = ("task_success", "protected_field_violation", "fabrication_bait_violation")
STABILITY_CONTINUOUS = (
    "faithfulness", "unsupported_claim_rate", "requirement_coverage", "kw_after", "sim_after",
)


def variant_family(variant):
    return VARIANT_FAMILIES.get(variant, "format")


def variant_scope(row_or_variant):
    if isinstance(row_or_variant, dict):
        if row_or_variant.get("variant_scope"):
            return row_or_variant["variant_scope"]
        row_or_variant = row_or_variant.get("variant", ORIGINAL)
    family = variant_family(row_or_variant)
    return "out_of_scope" if family in OUT_OF_SCOPE_FAMILIES else "in_scope"


def select_variants(variants=None, families=None):
    """Canonical-order variant selection; ``original`` is added when any in-scope one is run."""
    unknown = sorted(set(variants or ()) - set(VARIANTS))
    unknown_families = sorted(set(families or ()) - set(FAMILIES))
    if unknown or unknown_families:
        raise ValueError(f"unknown robustness variants {unknown} or families {unknown_families}")
    if not variants and not families:
        return VARIANTS
    wanted = set(variants or ()) | {
        name for name, family in VARIANT_FAMILIES.items() if family in set(families or ())
    }
    if any(variant_scope(name) == "in_scope" for name in wanted):
        wanted.add(ORIGINAL)
    return tuple(name for name in VARIANTS if name in wanted)


def robustness_identity():
    """Identity of the perturbation set and sample; artifacts live under its hash."""
    payload = {
        "sample_version": config.ROBUSTNESS_SAMPLE_VERSION,
        "variant_version": config.ROBUSTNESS_VARIANT_VERSION,
        "seed": config.ROBUSTNESS_SEED,
        "typo_rate": config.ROBUSTNESS_TYPO_RATE,
        "variants": VARIANT_FAMILIES,
        "degenerate_modes": list(DEGENERATE_MODES),
        "scoring_reference": "clean_original_resume",
    }
    payload["identity_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    return payload


# --- Shared helpers ---------------------------------------------------------------------------


def _rng(seed, pair, variant, part):
    # String seeds are hashed with SHA-512 by ``random``, so this is stable across processes.
    return random.Random(f"{seed}|{pair.get('pair_id', '')}|{variant}|{part}")


def _content_tokens(text):
    return Counter(re.findall(r"[a-z0-9]+", text.lower()))


def transformation_preserves_content(before, after):
    """Formatting/order variants must preserve the same case-insensitive lexical content."""
    return _content_tokens(before) == _content_tokens(after)


def _keyword_terms(keywords):
    return [term for item in keywords or () for term in (item["keyword"], *item.get("synonyms", []))]


def _keyword_tokens(keywords):
    tokens = set()
    for term in _keyword_terms(keywords):
        tokens.update(re.findall(r"[a-z0-9]+", term.lower()))
    return tokens


def _mentions_keyword(text, keywords):
    return any(metrics._has(text, term) for term in _keyword_terms(keywords))


def _protected_values(resume):
    """Protected-field values (name, contact, titles, employers, dates, degrees, institutions)."""
    return {
        kind: sorted(item["normalized"] for item in items)
        for kind, items in metrics._protected_entities(resume).items()
    }


def _protected_tokens(resume):
    tokens = set()
    for kind, items in metrics._protected_entities(resume).items():
        if kind == "contact_phone":
            continue  # stored as bare digits; phone lines are never perturbed token-wise
        for item in items:
            tokens.update(re.findall(r"[a-z0-9]+", item["value"].lower()))
    return tokens


def _keyword_hits(text, keywords):
    return sorted(metrics.keyword_coverage(text, keywords or [])["hit"])


def _jd_title(jd):
    for line in jd.splitlines():
        if line.strip():
            return re.sub(r"^#+\s*", "", line.strip())
    return ""


# --- v1 transforms (unchanged) ----------------------------------------------------------------


def _formatting(text):
    lines = []
    for line in text.splitlines():
        if line.startswith("- "):
            line = "* " + line[2:]
        elif line.startswith("## "):
            line = "## " + line[3:].upper()
        lines.append(line)
    return "\n\n".join(lines) + "\n"


def _reorder_sections(text):
    matches = list(re.finditer(r"(?m)^##\s+.+$", text))
    if len(matches) < 2:
        return text
    preamble = text[:matches[0].start()].rstrip()
    sections = []
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append(text[match.start():stop].strip())
    return preamble + "\n\n" + "\n\n".join(reversed(sections)) + "\n"


# --- Typos ----------------------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z]+")
_KEYBOARD_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
_NEIGHBOURS = {}
for _row in _KEYBOARD_ROWS:
    for _index, _char in enumerate(_row):
        _NEIGHBOURS[_char] = (_row[_index - 1] if _index else "") + (
            _row[_index + 1] if _index + 1 < len(_row) else ""
        )


def _typo(word, rng, forbidden):
    """One keyboard-style edit (swap, drop, double, neighbour key); first letter kept."""
    for _ in range(12):
        operation = rng.choice(("swap", "drop", "double", "neighbour"))
        index = rng.randrange(1, len(word) - 1)
        if operation == "swap":
            candidate = word[:index] + word[index + 1] + word[index] + word[index + 2:]
        elif operation == "drop":
            candidate = word[:index] + word[index + 1:]
        elif operation == "double":
            candidate = word[:index] + word[index] + word[index:]
        else:
            neighbours = _NEIGHBOURS.get(word[index], "")
            if not neighbours:
                continue
            candidate = word[:index] + rng.choice(neighbours) + word[index + 1:]
        if candidate != word and candidate not in forbidden:
            return candidate
    return word[0] + word[2:] if word[0] + word[2:] not in forbidden else word + word[-1]


def _resume_typo_lines(resume):
    """Only prose claim lines; headers, contact, dates and skill lists are never touched."""
    return {record["line"] for record in metrics._line_records(resume) if record["kind"] == "claim"}


def _jd_typo_lines(jd):
    lines = set()
    for number, raw in enumerate(jd.splitlines(), 1):
        stripped = raw.strip()
        if stripped and not stripped.startswith(("#", "**", "__")):
            lines.add(number)
    return lines


def _typo_candidates(text, lines, protected):
    candidates = []
    for index, line in enumerate(text.split("\n")):
        if index + 1 not in lines:
            continue
        for match in _WORD.finditer(line):
            word = match.group(0)
            if len(word) >= 5 and word.islower() and word not in protected:
                candidates.append((index, match.start(), match.end()))
    return candidates


def _apply_typos(text, lines, protected, rng, rate):
    candidates = _typo_candidates(text, lines, protected)
    if not candidates:
        return text, []
    count = max(1, round(rate * len(candidates)))
    chosen = sorted(rng.sample(candidates, count))
    rows = text.split("\n")
    changes = []
    # Apply right to left within a line so earlier offsets stay valid.
    for index, start, end in sorted(chosen, key=lambda item: (item[0], -item[1])):
        word = rows[index][start:end]
        new = _typo(word, rng, protected)
        rows[index] = rows[index][:start] + new + rows[index][end:]
        changes.append({"line": index + 1, "before": word, "after": new})
    changes.sort(key=lambda change: (change["line"], change["before"]))
    return "\n".join(rows), changes


def _edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def _typo_problems(label, before, after, protected, keywords):
    problems = []
    before_lines, after_lines = before.split("\n"), after.split("\n")
    if len(before_lines) != len(after_lines):
        return [f"{label}: line count changed"]
    changed = 0
    total = 0
    for number, (old, new) in enumerate(zip(before_lines, after_lines), 1):
        old_words, new_words = old.split(), new.split()
        total += len(old_words)
        if len(old_words) != len(new_words):
            problems.append(f"{label}: word count changed on line {number}")
            continue
        for old_word, new_word in zip(old_words, new_words):
            if old_word == new_word:
                continue
            changed += 1
            old_letters = re.findall(r"[A-Za-z]+", old_word)
            if re.sub(r"[A-Za-z]", "", old_word) != re.sub(r"[A-Za-z]", "", new_word):
                problems.append(f"{label}: non-letter characters changed in {old_word!r}")
            elif _edit_distance(old_word, new_word) > 2:
                problems.append(f"{label}: {old_word!r} -> {new_word!r} is not a typo")
            elif any(word.lower() in protected or not word.islower() for word in old_letters
                     if word not in re.findall(r"[A-Za-z]+", new_word)):
                problems.append(f"{label}: protected or capitalised token {old_word!r} changed")
    if not changed:
        problems.append(f"{label}: no typo applied")
    elif changed > max(3, 0.1 * total):
        problems.append(f"{label}: {changed} of {total} words changed; rate too high")
    if _keyword_hits(before, keywords) != _keyword_hits(after, keywords):
        problems.append(f"{label}: keyword families changed")
    return problems


# --- PDF extraction artefacts ---------------------------------------------------------------

_PAGE_LINE = re.compile(r"^\s*Page \d+ of \d+\s*$")
_RUNNING_HEADER = re.compile(r"^.*\| Curriculum Vitae\s*$")


def _plain_text(text):
    lines = []
    for line in text.splitlines():
        line = line.rstrip()
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            line = heading.group(2).upper() if len(heading.group(1)) == 2 else heading.group(2)
        else:
            line = re.sub(r"^(\s*)[-*+]\s+", r"\1• ", line)
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        line = re.sub(r"__(.+?)__", r"\1", line)
        lines.append(line.replace("`", ""))
    return "\n".join(lines).strip() + "\n"


def _wrap(line, width, rng, protected):
    words = line.split(" ")
    lines, current = [], ""
    for word in words:
        if not current:
            current = word
            continue
        if len(current) + 1 + len(word) <= width:
            current += " " + word
            continue
        if (
            len(word) >= 8 and word.isalpha() and word.islower() and word not in protected
            and rng.random() < 0.6
        ):
            cut = rng.randrange(3, len(word) - 2)
            lines.append(current + " " + word[:cut] + "-")
            current = word[cut:]
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _pdf_extraction(resume, rng, protected):
    plain = _plain_text(resume).rstrip("\n").split("\n")
    width = rng.randrange(56, 73)
    wrapped = []
    for line in plain:
        wrapped.extend(_wrap(line, width, rng, protected) if len(line) > width else [line])
    noisy = []
    for line in wrapped:
        roll = rng.random()
        if line and roll < 0.2 and " " in line:
            spaces = [index for index, char in enumerate(line) if char == " "]
            position = rng.choice(spaces)
            line = line[:position] + " " * rng.randrange(2, 4) + line[position + 1:]
        elif line and roll < 0.35:
            line += " " * rng.randrange(1, 4)
        elif line and roll < 0.45:
            line = " " + line
        noisy.append(line)
    name = _jd_title(resume) or "Candidate"
    split = max(1, int(len(noisy) * rng.uniform(0.4, 0.6)))
    while split < len(noisy) and noisy[split - 1].rstrip().endswith("-"):
        split += 1  # a hyphenation split never straddles the page break
    output = (
        noisy[:split]
        + ["", "Page 1 of 2", "", f"{name} | Curriculum Vitae", ""]
        + noisy[split:]
        + ["", "Page 2 of 2", ""]
    )
    return "\n".join(output)


def repair_pdf_extraction(text):
    """Undo the known artefacts; used only by the content-preservation check."""
    lines = [
        line for line in text.split("\n")
        if not _PAGE_LINE.match(line) and not _RUNNING_HEADER.match(line)
    ]
    return re.sub(r"([a-z])-[ \t]*\n[ \t]*([a-z])", r"\1\2", "\n".join(lines))


# --- Web-page clutter ------------------------------------------------------------------------

_CLUTTER_HEADER = (
    "Skip to main content",
    "Home | Jobs | Teams | Locations | Students and graduates | Sign in",
    "We use cookies to give you the best experience on our site. By continuing to browse you "
    "agree to our use of cookies. Accept all | Reject optional | Manage preferences",
    "Careers > Search jobs > Job details",
)
_CLUTTER_FOOTER = (
    "Apply now | Save job | Share this job",
    "Why join us? Flexible working arrangements, generous annual leave, medical and dental "
    "coverage, a yearly learning allowance and wellness programmes.",
    "We are an equal opportunity employer. All qualified applicants will receive consideration "
    "for employment without regard to race, religion, colour, national origin, gender, sexual "
    "orientation, age, marital status or disability.",
    "Job ID: R{job_id} | Posted {days} days ago",
    "Recently viewed jobs | Job alerts | Join our talent community",
    "Privacy notice | Terms of use | Cookie settings | Contact us",
)
_CLUTTER_COPYRIGHT = "© 2026 Careers portal. All rights reserved."


def _web_clutter(jd, keywords, rng):
    def allowed(block):
        return not _mentions_keyword(block, keywords)

    header = [block for block in _CLUTTER_HEADER if allowed(block)]
    footer = [
        block.format(job_id=rng.randrange(10000, 99999), days=rng.randrange(1, 30))
        for block in _CLUTTER_FOOTER
    ]
    footer = [block for block in footer if allowed(block)]
    footer = rng.sample(footer, max(1, len(footer) - rng.randrange(0, 2)))
    lines = header + [jd.strip()] + footer + [_CLUTTER_COPYRIGHT]
    return "\n\n".join(lines) + "\n", jd.strip()


# --- Heading aliases -------------------------------------------------------------------------

_HEADING_ALIASES = {
    "summary": ("Professional Profile", "Career Summary", "About Me"),
    "experience": ("Work History", "Employment History", "Career History"),
    "projects": ("Selected Projects", "Personal Projects", "Portfolio"),
    "education": ("Academic Background", "Education and Training", "Academic History"),
    "skills": ("Core Competencies", "Technical Toolkit", "Key Skills"),
    "certifications": ("Licences and Certifications", "Credentials", "Professional Certifications"),
    "about the role": ("The Opportunity", "Role Overview", "About This Position"),
    "responsibilities": ("What You Will Do", "Your Impact", "Key Duties"),
    "required qualifications": ("What We Are Looking For", "Must Haves", "Requirements"),
    "preferred qualifications": ("Nice to Have", "Bonus Points", "Desirable Extras"),
    "what success looks like": ("How We Measure Success", "Your First Six Months", "Success Measures"),
}
_LEVEL2 = re.compile(r"^##\s+(.+?)\s*$")


def _heading_aliases(text, keywords, rng):
    lines = []
    for line in text.split("\n"):
        match = _LEVEL2.match(line)
        if match:
            options = [
                alias for alias in _HEADING_ALIASES.get(match.group(1).strip().lower(), ())
                if not _mentions_keyword(alias, keywords)
            ]
            if options:
                line = f"## {rng.choice(options)}"
        lines.append(line)
    return "\n".join(lines)


def _heading_problems(label, before, after, keywords):
    old_lines, new_lines = before.split("\n"), after.split("\n")
    if len(old_lines) != len(new_lines):
        return [f"{label}: line count changed"]
    problems, renamed = [], 0
    for old, new in zip(old_lines, new_lines):
        if old == new:
            continue
        if not (_LEVEL2.match(old) and _LEVEL2.match(new)):
            problems.append(f"{label}: non-heading line changed: {old!r}")
        elif _mentions_keyword(new, keywords):
            problems.append(f"{label}: alias {new!r} introduces a keyword")
        else:
            renamed += 1
    if not renamed and _LEVEL2.search(before):
        problems.append(f"{label}: no heading renamed")
    return problems


# --- Out of scope ----------------------------------------------------------------------------

_NON_JOB_TEXTS = (
    "Weekend weather update: light showers are expected on Saturday morning, clearing by the "
    "afternoon. Bring an umbrella if you are heading to the market.",
    "Banana bread: mash three ripe bananas, stir in melted butter, sugar, one egg and flour, "
    "then bake for about an hour until golden.",
    "Community notice: the library will close early on Friday for maintenance and reopen on "
    "Monday at the usual time.",
)


def build_jd_pool(pairs):
    """Distinct dataset JDs keyed by ``jd_id`` (used for cross-category mismatches)."""
    pool = {}
    for pair in pairs:
        pool.setdefault(pair["jd_id"], {
            "jd_id": pair["jd_id"],
            "category": pair.get("category", ""),
            "role_family": pair.get("role_family", ""),
            "jd": pair["jd"],
            "keywords": pair.get("keywords", []),
        })
    return pool


def mismatched_jd(pair, jd_pool, seed):
    """A seeded JD from a different category than the resume."""
    rng = _rng(seed, pair, "mismatched_jd", "choice")
    categories = sorted({entry["category"] for entry in jd_pool.values()} - {pair.get("category")})
    if not categories:
        raise ValueError("mismatched_jd needs JDs from at least two categories")
    target = categories[rng.randrange(len(categories))]
    candidates = sorted(jd_id for jd_id, entry in jd_pool.items() if entry["category"] == target)
    return jd_pool[candidates[rng.randrange(len(candidates))]]


def degenerate_mode(pair, seed):
    return DEGENERATE_MODES[_rng(seed, pair, "degenerate_jd", "mode").randrange(len(DEGENERATE_MODES))]


def _degenerate_jd(pair, seed):
    mode = degenerate_mode(pair, seed)
    title = _jd_title(pair["jd"])
    if mode == "title_only":
        return mode, f"{title}\n"
    if mode == "very_short":
        return mode, f"Hiring now: {title}. Apply today.\n"
    options = [text for text in _NON_JOB_TEXTS if not _mentions_keyword(text, pair.get("keywords"))]
    rng = _rng(seed, pair, "degenerate_jd", "text")
    return mode, options[rng.randrange(len(options))] + "\n"


def _merge_keywords(*keyword_lists):
    merged, seen = [], set()
    for keywords in keyword_lists:
        for item in keywords or ():
            if item["keyword"].lower() not in seen:
                seen.add(item["keyword"].lower())
                merged.append(item)
    return merged


# --- Variant construction --------------------------------------------------------------------


def build_variant(pair, variant, seed=None, jd_pool=None, check=True):
    """Return the perturbed inputs plus the clean references used to score them.

    Keys: ``resume``/``jd`` (what the workflow sees), ``scoring_jd``/``scoring_keywords`` (clean
    JD and keywords for in-scope variants; the presented JD and the union of keyword families for
    out-of-scope ones), ``presented_keywords``, ``family``, ``scope``, ``resume_suffix`` and
    ``metadata``. The resume used for scoring is always ``pair["resume"]`` (the clean source).
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown robustness variant: {variant}")
    seed = config.ROBUSTNESS_SEED if seed is None else seed
    resume, jd, keywords = pair["resume"], pair["jd"], pair.get("keywords", [])
    spec = {
        "variant": variant,
        "family": variant_family(variant),
        "scope": variant_scope(variant),
        "resume": resume,
        "jd": jd,
        "scoring_jd": jd,
        "scoring_keywords": keywords,
        "presented_keywords": keywords,
        "resume_suffix": ".md",
        "metadata": {},
    }
    if variant == "formatting":
        spec.update(resume=_formatting(resume), jd=_formatting(jd))
    elif variant == "reordered_sections":
        spec.update(resume=_reorder_sections(resume), jd=_reorder_sections(jd))
    elif variant == "plain_text":
        spec.update(resume=_plain_text(resume), jd=_plain_text(jd), resume_suffix=".txt")
    elif variant == "heading_aliases":
        spec.update(
            resume=_heading_aliases(resume, keywords, _rng(seed, pair, variant, "resume")),
            jd=_heading_aliases(jd, keywords, _rng(seed, pair, variant, "jd")),
        )
    elif variant == "typos":
        protected = _protected_tokens(resume) | _keyword_tokens(keywords)
        new_resume, resume_changes = _apply_typos(
            resume, _resume_typo_lines(resume), protected,
            _rng(seed, pair, variant, "resume"), config.ROBUSTNESS_TYPO_RATE,
        )
        jd_protected = _keyword_tokens(keywords) | set(
            re.findall(r"[a-z0-9]+", _jd_title(jd).lower())
        )
        new_jd, jd_changes = _apply_typos(
            jd, _jd_typo_lines(jd), jd_protected,
            _rng(seed, pair, variant, "jd"), config.ROBUSTNESS_TYPO_RATE,
        )
        spec.update(resume=new_resume, jd=new_jd, metadata={
            "resume_typos": resume_changes, "jd_typos": jd_changes,
        })
    elif variant == "pdf_extraction":
        protected = _protected_tokens(resume) | _keyword_tokens(keywords)
        spec.update(
            resume=_pdf_extraction(resume, _rng(seed, pair, variant, "resume"), protected),
            resume_suffix=".txt",
        )
    elif variant == "web_clutter":
        cluttered, _ = _web_clutter(jd, keywords, _rng(seed, pair, variant, "jd"))
        spec.update(jd=cluttered)
    elif variant == "mismatched_jd":
        if jd_pool is None:
            raise ValueError("mismatched_jd needs the dataset JD pool")
        other = mismatched_jd(pair, jd_pool, seed)
        spec.update(
            jd=other["jd"],
            scoring_jd=other["jd"],
            presented_keywords=other["keywords"],
            scoring_keywords=_merge_keywords(keywords, other["keywords"]),
            metadata={"mismatched_jd_id": other["jd_id"], "mismatched_category": other["category"]},
        )
    elif variant == "degenerate_jd":
        mode, text = _degenerate_jd(pair, seed)
        spec.update(jd=text, scoring_jd=text, metadata={"degenerate_mode": mode})
    if check:
        problems = variant_content_problems(pair, spec, jd_pool)
        if problems:
            raise ValueError(f"{variant} failed content preservation: {problems}")
    return spec


def variant_content_problems(pair, spec, jd_pool=None):
    """Per-variant content-preservation check; an empty list means the facts survived.

    - formatting, reordered_sections, plain_text: identical case-insensitive token multisets.
    - heading_aliases: only level-2 heading lines change and no alias adds a keyword.
    - typos: same lines and word counts, every change is a <=2-edit typo of a lowercase,
      unprotected word, protected-field values and keyword families are unchanged.
    - pdf_extraction: after removing page lines/running header and joining hyphenation splits,
      tokens equal the clean resume and protected-field tokens are all present.
    - web_clutter: the clean JD appears verbatim and the clutter mentions no keyword family.
    - mismatched_jd: resume unchanged; JD is a dataset JD from a different category.
    - degenerate_jd: resume unchanged; JD is short and mentions no keyword family of either JD.
    """
    variant = spec["variant"]
    clean_resume, clean_jd = pair["resume"], pair["jd"]
    keywords = pair.get("keywords", [])
    resume, jd = spec["resume"], spec["jd"]
    problems = []
    if variant == ORIGINAL:
        if (resume, jd) != (clean_resume, clean_jd):
            problems.append("original inputs changed")
    elif variant in {"formatting", "reordered_sections", "plain_text"}:
        if not transformation_preserves_content(clean_resume, resume):
            problems.append("resume tokens changed")
        if not transformation_preserves_content(clean_jd, jd):
            problems.append("JD tokens changed")
        if variant == "plain_text" and any(
            re.match(r"^\s*(#|[-*+]\s)", line) or "**" in line
            for line in (resume + "\n" + jd).splitlines()
        ):
            problems.append("Markdown left in plain text")
    elif variant == "heading_aliases":
        problems += _heading_problems("resume", clean_resume, resume, keywords)
        problems += _heading_problems("JD", clean_jd, jd, keywords)
    elif variant == "typos":
        protected = _protected_tokens(clean_resume) | _keyword_tokens(keywords)
        problems += _typo_problems("resume", clean_resume, resume, protected, keywords)
        if _protected_values(clean_resume) != _protected_values(resume):
            problems.append("resume protected-field values changed")
        jd_protected = _keyword_tokens(keywords) | set(
            re.findall(r"[a-z0-9]+", _jd_title(clean_jd).lower())
        )
        problems += _typo_problems("JD", clean_jd, jd, jd_protected, keywords)
        if _jd_title(clean_jd) != _jd_title(jd):
            problems.append("JD title changed")
    elif variant == "pdf_extraction":
        repaired = repair_pdf_extraction(resume)
        if not transformation_preserves_content(clean_resume, repaired):
            problems.append("resume tokens changed after repairing extraction artefacts")
        tokens = _content_tokens(resume)
        missing = sorted(token for token in _protected_tokens(clean_resume) if not tokens[token])
        if missing:
            problems.append(f"protected tokens split or lost: {missing}")
        if jd != clean_jd:
            problems.append("JD changed")
    elif variant == "web_clutter":
        if clean_jd.strip() not in jd:
            problems.append("clean JD not contained verbatim")
        clutter = jd.replace(clean_jd.strip(), "")
        if _mentions_keyword(clutter, keywords):
            problems.append("clutter mentions a JD keyword family")
        if resume != clean_resume:
            problems.append("resume changed")
    elif variant == "mismatched_jd":
        if resume != clean_resume:
            problems.append("resume changed")
        other = (jd_pool or {}).get(spec["metadata"].get("mismatched_jd_id"))
        if other is None or other["jd"] != jd:
            problems.append("JD is not a dataset JD")
        elif other["category"] == pair.get("category"):
            problems.append("mismatched JD has the same category")
    elif variant == "degenerate_jd":
        if resume != clean_resume:
            problems.append("resume changed")
        if len(jd.split()) > 40:
            problems.append("degenerate JD is not short")
        if spec["metadata"].get("degenerate_mode") == "non_job_text" and _mentions_keyword(
            jd, keywords
        ):
            problems.append("non-job text mentions a JD keyword family")
    return problems


def perturb_pair(pair, variant, seed=None, jd_pool=None):
    """Backwards-compatible ``(resume, jd)`` view of ``build_variant``."""
    spec = build_variant(pair, variant, seed, jd_pool)
    return spec["resume"], spec["jd"]


# --- Sample ----------------------------------------------------------------------------------


def select_stratified_pairs(pairs, seed=42):
    """Select four per category and globally 8 strong, 8 moderate and 4 weak pairs."""
    by_category = defaultdict(lambda: defaultdict(list))
    for pair in pairs:
        by_category[pair["category"]][pair["fit_level"]].append(pair)
    categories = sorted(by_category)
    if len(categories) != 5:
        raise ValueError(f"expected five categories, got {categories}")
    allocations = (
        {"strong": 2, "moderate": 1, "weak": 1},
        {"strong": 2, "moderate": 1, "weak": 1},
        {"strong": 1, "moderate": 2, "weak": 1},
        {"strong": 1, "moderate": 2, "weak": 1},
        {"strong": 2, "moderate": 2, "weak": 0},
    )
    rng = random.Random(seed)
    selected = []
    for category, allocation in zip(categories, allocations):
        for fit_level, count in allocation.items():
            candidates = sorted(
                by_category[category][fit_level],
                key=lambda pair: (pair.get("role_family", ""), pair["pair_id"]),
            )
            if len(candidates) < count:
                raise ValueError(f"not enough {category}/{fit_level} pairs")
            chosen = rng.sample(candidates, count)
            selected.extend(sorted(chosen, key=lambda pair: pair["pair_id"]))
    return selected


def ordered_robustness_sample(pairs, seed=42):
    """The v1 stratified 20 pairs, ordered so every prefix is spread over categories and fits.

    Within a category fit levels alternate strong, moderate, weak; categories then take turns.
    ``--robustness-limit 10`` therefore covers all five categories with strong and moderate fit.
    """
    by_category = defaultdict(lambda: defaultdict(list))
    for pair in select_stratified_pairs(pairs, seed):
        by_category[pair["category"]][pair["fit_level"]].append(pair)
    queues = {}
    for category, by_fit in by_category.items():
        ordered = []
        fits = [fit for fit in ("strong", "moderate", "weak") if by_fit.get(fit)] + sorted(
            set(by_fit) - {"strong", "moderate", "weak"}
        )
        pending = {fit: sorted(by_fit[fit], key=lambda pair: pair["pair_id"]) for fit in fits}
        while any(pending.values()):
            for fit in fits:
                if pending[fit]:
                    ordered.append(pending[fit].pop(0))
        queues[category] = ordered
    sample = []
    while any(queues.values()):
        for category in sorted(queues):
            if queues[category]:
                sample.append(queues[category].pop(0))
    return sample


def _hash_payload(payload):
    payload["manifest_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    return payload


def robustness_manifest_v1(pairs, dataset_hash, seed=42):
    """The v1 manifest, kept for the legacy ``run_robustness.py`` history."""
    selected = select_stratified_pairs(pairs, seed)
    return _hash_payload({
        "version": "robustness-sample-v1",
        "dataset_hash": dataset_hash,
        "seed": seed,
        "pair_ids": [pair["pair_id"] for pair in selected],
        "strata": [
            {
                "pair_id": pair["pair_id"],
                "category": pair["category"],
                "fit_level": pair["fit_level"],
                "role_family": pair.get("role_family"),
                "seniority": pair.get("seniority"),
            }
            for pair in selected
        ],
    })


def robustness_manifest(pairs, dataset_hash, seed=42):
    """Ordered sample plus the deterministic out-of-scope choices for every pair."""
    selected = ordered_robustness_sample(pairs, seed)
    jd_pool = build_jd_pool(pairs)
    strata = []
    for pair in selected:
        other = mismatched_jd(pair, jd_pool, seed)
        strata.append({
            "pair_id": pair["pair_id"],
            "category": pair["category"],
            "fit_level": pair["fit_level"],
            "role_family": pair.get("role_family"),
            "seniority": pair.get("seniority"),
            "mismatched_jd_id": other["jd_id"],
            "mismatched_category": other["category"],
            "degenerate_mode": degenerate_mode(pair, seed),
        })
    return _hash_payload({
        "version": config.ROBUSTNESS_SAMPLE_VERSION,
        "variant_version": config.ROBUSTNESS_VARIANT_VERSION,
        "dataset_hash": dataset_hash,
        "seed": seed,
        "variants": VARIANT_FAMILIES,
        "pair_ids": [pair["pair_id"] for pair in selected],
        "strata": strata,
    })


# --- Record-level outcomes -------------------------------------------------------------------

_REFUSAL = re.compile(
    r"\b(?:cannot|can't|can not|unable to|not able to|won't)\b[^.\n]{0,80}"
    r"\b(?:tailor|job description|resume|r[ée]sum[ée])\b"
    r"|\bnot (?:a |appear to be a )?(?:valid |real )?job (?:description|posting|advert)\b",
    re.IGNORECASE,
)


def is_refusal(text):
    """A short explanatory refusal rather than a resume (fewer than two required sections)."""
    from eval.operational import REQUIRED_SECTIONS, _has_section

    if not text or not _REFUSAL.search(text[:1500]):
        return False
    return sum(_has_section(text, section) for section in REQUIRED_SECTIONS) < 2


def classify_generation(tailored, error, crashed):
    """``crashed`` (uncaught exception), ``handled_error`` (workflow-reported error or empty
    output), ``refused`` (explicit refusal text) or ``completed``."""
    if crashed:
        return "crashed"
    if error or not (tailored or "").strip():
        return "handled_error"
    if is_refusal(tailored):
        return "refused"
    return "completed"


def _all_or_none(values):
    """False if any check failed; None if any is unscored; otherwise True."""
    if any(value is False for value in values):
        return False
    if any(value is None for value in values):
        return None
    return True


def out_of_scope_expected_behaviour(record):
    """Expected-behaviour checks for an out-of-scope run (all against the clean resume).

    - crashed: ``expected_behaviour_pass=False`` (graceful handling failed); checks unscored.
    - handled_error / refused: handled gracefully, ``expected_behaviour_pass=True``.
    - completed: ``no_fabrication`` (no fabrication bait and faithfulness >= the task
      threshold, i.e. unsupported claims <= 1 - threshold) and ``protected_fields_preserved``;
      a failed check fails the run, an unscored check (evaluator failure or skipped NLI)
      leaves it unscored unless another check already failed.
    """
    outcome = record.get("generation_outcome")
    result = {
        "oos_handled": outcome in {"handled_error", "refused"},
        "oos_crashed": outcome == "crashed",
        "oos_no_fabrication": None,
        "oos_protected_fields_preserved": None,
        "expected_behaviour_pass": None,
    }
    if outcome == "crashed":
        result["expected_behaviour_pass"] = False
        return result
    if outcome in {"handled_error", "refused"}:
        result["expected_behaviour_pass"] = True
        return result
    bait = record.get("fabrication_bait_violation")
    faithfulness = record.get("faithfulness")
    protected = record.get("protected_field_violation")
    result["oos_no_fabrication"] = _all_or_none([
        None if bait is None else not bait,
        None if faithfulness is None else faithfulness >= config.TASK_FAITHFULNESS_THRESHOLD,
    ])
    result["oos_protected_fields_preserved"] = None if protected is None else not protected
    result["expected_behaviour_pass"] = _all_or_none([
        result["oos_no_fabrication"], result["oos_protected_fields_preserved"],
    ])
    return result


def _value(row, metric):
    value = row.get(metric) if row else None
    return None if value is None else float(value)


def paired_deltas(original, variant):
    """Signed variant-minus-original deltas for every ``DELTA_METRICS`` entry (None if unpaired)."""
    usable = (
        original is not None and not original.get("error")
        and variant is not None and not variant.get("error")
    )
    deltas = {}
    for metric in DELTA_METRICS:
        before, after = _value(original, metric), _value(variant, metric)
        deltas[metric] = (
            round(after - before, 6) if usable and None not in (before, after) else None
        )
    return deltas


def stability_assessment(original, variant, tolerance=None):
    """Whether one in-scope variant run kept the original's quality.

    Only defined when both generations completed (generation failures are reported separately
    as execution retention). Stable means no binary criterion flipped to worse and no continuous
    metric degraded by more than ``tolerance``. ``stable`` is None when no criterion is scorable.
    """
    tolerance = config.ROBUSTNESS_STABILITY_TOLERANCE if tolerance is None else tolerance
    if original is None or variant is None or original.get("error") or variant.get("error"):
        return {"comparable": False, "stable": None, "criteria_scored": 0, "violations": []}
    deltas = paired_deltas(original, variant)
    scored, violations = 0, []
    for metric in (*STABILITY_BINARY, *STABILITY_CONTINUOUS):
        if deltas[metric] is None:
            continue
        scored += 1
        degradation = -DELTA_METRICS[metric] * deltas[metric]
        limit = 0.0 if metric in STABILITY_BINARY else tolerance
        if degradation > limit + 1e-9:
            violations.append(metric)
    return {
        "comparable": True,
        "stable": (not violations) if scored else None,
        "criteria_scored": scored,
        "violations": violations,
    }


def robustness_comparisons(rows):
    """Each non-original in-scope row paired with its workflow's original row (or None)."""
    originals = {
        (row["workflow"], row["base_pair_id"]): row
        for row in rows if row.get("variant") == ORIGINAL
    }
    comparisons = []
    for row in rows:
        if row.get("variant") == ORIGINAL or variant_scope(row) != "in_scope":
            continue
        comparisons.append({
            "workflow": row["workflow"],
            "variant": row["variant"],
            "family": row.get("variant_family") or variant_family(row["variant"]),
            "base_pair_id": row["base_pair_id"],
            "original": originals.get((row["workflow"], row["base_pair_id"])),
            "row": row,
        })
    return comparisons


# --- v1 summary (legacy run_robustness.py) ---------------------------------------------------


def summarise_robustness(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["workflow"], row["base_pair_id"])].append(row)

    by_workflow = defaultdict(list)
    for (workflow, _), group in grouped.items():
        by_workflow[workflow].append(group)

    summary = []
    for workflow, pair_groups in sorted(by_workflow.items()):
        success_values = []
        metric_deltas = defaultdict(list)
        for group in pair_groups:
            original = next((row for row in group if row["variant"] == "original"), None)
            success_values.extend(1 if not row.get("error") else 0 for row in group)
            if original is None:
                continue
            for row in group:
                if row["variant"] == "original":
                    continue
                if any(item.get("error") or item.get("finished") is False
                       or ("resume_md" in item and not item["resume_md"].strip())
                       for item in (original, row)):
                    continue
                for metric in ROBUSTNESS_METRICS:
                    if original.get(metric) is not None and row.get(metric) is not None:
                        metric_deltas[metric].append(row[metric] - original[metric])

        record = {
            "workflow": workflow,
            "pairs": len(pair_groups),
            "variant_runs": sum(len(group) for group in pair_groups),
            "robust_success_rate": round(sum(success_values) / len(success_values), 4)
            if success_values else None,
        }
        for metric in ROBUSTNESS_METRICS:
            deltas = metric_deltas[metric]
            record[f"{metric}_paired_n"] = len(deltas)
            record[f"{metric}_mean_delta"] = round(st.mean(deltas), 4) if deltas else None
            record[f"{metric}_mean_abs_change"] = (
                round(st.mean(abs(value) for value in deltas), 4) if deltas else None
            )
            record[f"{metric}_worst_degradation"] = (
                round(min(deltas), 4) if deltas else None
            )
        summary.append(record)
    return summary


def write_robustness_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
