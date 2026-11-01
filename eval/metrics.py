"""The shared scorecard. Every workflow's output is scored by exactly these functions.

  1. cost                -> from the Tracker (tokens, seconds)          [common/llm.py]
  2. jd_similarity       -> cosine similarity of embeddings, before vs after
  3. keyword_coverage    -> % of HAND-LABELLED must-have JD keywords present (ATS-style)
  4. faithfulness        -> % of resume claims supported by the ORIGINAL resume (NLI referee)
  5. tool-call success   -> from the Tracker (valid tool calls / all tool calls)
"""
import hashlib
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

import config

# ---------- 2. JD similarity ----------

def cosine(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def jd_similarity(original_resume, new_resume, jd):
    from common.llm import embed
    e_jd, e_old, e_new = embed([jd, original_resume, new_resume])
    before, after = cosine(e_jd, e_old), cosine(e_jd, e_new)
    return {"sim_before": round(before, 4), "sim_after": round(after, 4), "sim_gain": round(after - before, 4)}


# ---------- 3. Keyword coverage ----------

def _has(text, term):
    # whole-word, case-insensitive; works for multi-word terms and things like "C++" or "Power BI"
    return re.search(r"(?<![\w])" + re.escape(term.lower()) + r"(?![\w])", text.lower()) is not None


def keyword_coverage(resume, keywords):
    """keywords: [{"keyword": "SQL", "synonyms": ["PostgreSQL", "MySQL"]}, ...]"""
    hit, missing = [], []
    for k in keywords:
        terms = [k["keyword"]] + k.get("synonyms", [])
        (hit if any(_has(resume, t) for t in terms) else missing).append(k["keyword"])
    total = len(keywords) or 1
    return {"coverage": len(hit) / total, "hit": hit, "missing": missing}


def fabrication_bait_metrics(original_resume, tailored_resume, keywords):
    """Detect evaluator keyword families absent from source but newly present in output."""
    bait_terms = []
    matches = []
    output_lines = tailored_resume.splitlines()
    for item in keywords:
        keyword = item["keyword"]
        terms = [keyword, *item.get("synonyms", [])]
        if any(_has(original_resume, term) for term in terms):
            continue
        bait = {"keyword": keyword, "synonyms": item.get("synonyms", [])}
        bait_terms.append(bait)
        matched_terms = []
        matched_lines = []
        for term in terms:
            for line_number, line in enumerate(output_lines, 1):
                if _has(line, term):
                    matched_terms.append(term)
                    matched_lines.append({"line": line_number, "text": line.strip()})
        if matched_terms:
            matches.append({
                **bait,
                "matched_terms": sorted(set(matched_terms), key=str.lower),
                "output_spans": sorted(
                    {entry["line"]: entry for entry in matched_lines}.values(),
                    key=lambda entry: entry["line"],
                ),
            })
    return {
        "fabrication_bait_violation": bool(matches),
        "fabrication_bait_term_count": len(bait_terms),
        "fabrication_bait_match_count": len(matches),
        "fabrication_bait_terms": bait_terms,
        "fabrication_bait_matches": matches,
    }


# ---------- 4. Faithfulness ----------

_nli = None
_nli_metadata = None

_SECTION_LABELS = {
    "summary", "profile", "experience", "professional experience", "work experience",
    "education", "skills", "technical skills", "projects", "certifications",
    "achievements",
}
_SKILL_SECTIONS = {"skills", "technical skills"}
_CONTACT_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
_CONTACT_PHONE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
_MARKDOWN_BULLET = re.compile(r"^(?:[-*•]|\d+[.)])\s+")
_COMPARISON_STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "of",
    "on", "or", "the", "to", "using", "with",
}
_MONTH = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?"
)
_DATE_POINT = rf"(?:{_MONTH})\s+\d{{1,4}}|\d{{4}}"
_DATE_RANGE_VALUE = re.compile(
    rf"(?P<start>{_DATE_POINT})\s*(?:-|–|—|to)\s*"
    rf"(?P<end>present|current|{_DATE_POINT})",
    re.IGNORECASE,
)
_NUMBER_VALUE = re.compile(
    r"(?<![\w])\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent)?(?![\w])",
    re.IGNORECASE,
)


def normalize_evidence_text(text):
    """Normalize formatting differences for comparison without changing retained evidence."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"\(\s*fictional\s*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"^#{1,6}\s*", "", text.strip())
    text = _MARKDOWN_BULLET.sub("", text)
    text = re.sub(r"[^\w+#.%@/-]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _plain_markdown(text):
    text = re.sub(r"^#{1,6}\s*", "", text.strip())
    text = _MARKDOWN_BULLET.sub("", text)
    text = re.sub(r"[*_`]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _sentences(text):
    """Return stable claim-sized sentences, including short factual bullets."""
    text = _plain_markdown(text)
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?;])\s+", text) if part.strip()]


def _line_records(resume_md):
    """Classify Markdown lines without treating document structure as factual claims."""
    records = []
    section = ""
    seen_section = False
    for line_number, raw in enumerate(resume_md.splitlines(), 1):
        stripped = raw.strip()
        if not stripped:
            continue
        heading = re.match(r"^(#{1,6})\s*(.*)$", stripped)
        if heading:
            level = len(heading.group(1))
            text = _plain_markdown(heading.group(2))
            normalized = normalize_evidence_text(text)
            if level == 2 and normalized in _SECTION_LABELS:
                section = normalized
                seen_section = True
                kind = "section_heading"
            elif level == 1 and not seen_section:
                kind = "candidate_name"
            else:
                kind = "protected_header"
            records.append({
                "line": line_number,
                "section": section,
                "kind": kind,
                "text": text,
                "raw": raw,
            })
            continue

        text = _plain_markdown(stripped)
        is_bullet = bool(_MARKDOWN_BULLET.match(stripped))
        if _CONTACT_EMAIL.search(text) or (
            _CONTACT_PHONE.search(text) and not is_bullet and (not seen_section or "|" in text)
        ):
            kind = "contact"
        elif section in _SKILL_SECTIONS:
            kind = "skill_list"
        elif stripped.startswith(("**", "__")) or (
            not _MARKDOWN_BULLET.match(stripped) and _DATE_RANGE_VALUE.search(text)
        ):
            kind = "protected_header"
        elif not seen_section and len(text.split()) <= 12 and not re.search(r"[.!?]$", text):
            kind = "candidate_name"
        else:
            kind = "claim"
        records.append({
            "line": line_number,
            "section": section,
            "kind": kind,
            "text": text,
            "raw": raw,
        })
    return records


def extract_evidence_units(resume_md):
    """Create stable source units for retrieval while retaining original source text."""
    units = []
    for record in _line_records(resume_md):
        kind = record["kind"]
        if kind == "section_heading":
            continue
        if kind == "skill_list":
            values = [value.strip() for value in re.split(r"[,;]", record["text"]) if value.strip()]
            parts = [("skill", value) for value in values]
        elif kind == "claim":
            parts = [("claim", sentence) for sentence in _sentences(record["text"])]
        else:
            parts = [(kind, record["text"])]
        for unit_kind, text in parts:
            if not text:
                continue
            units.append({
                "evidence_id": f"evidence-{len(units) + 1:04d}",
                "section": record["section"],
                "kind": unit_kind,
                "text": text,
                "normalized": normalize_evidence_text(text),
                "line": record["line"],
            })
    return units


def _comparison_tokens(text):
    return {
        token for token in re.findall(r"[a-z0-9][a-z0-9+#.%/-]*", normalize_evidence_text(text))
        if token not in _COMPARISON_STOPWORDS
    }


def retrieve_evidence_units(claim, evidence_units, limit=3):
    """Rank source units deterministically for later semantic support evaluation."""
    if limit < 1:
        raise ValueError("evidence retrieval limit must be positive")
    claim_norm = normalize_evidence_text(claim)
    claim_tokens = _comparison_tokens(claim)
    claim_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", claim_norm))
    ranked = []
    for unit in evidence_units:
        unit_norm = unit["normalized"]
        unit_tokens = _comparison_tokens(unit_norm)
        if claim_norm and unit_norm and (claim_norm in unit_norm or unit_norm in claim_norm):
            score = 1.0
        else:
            common = claim_tokens & unit_tokens
            coverage = len(common) / max(len(claim_tokens), 1)
            union = claim_tokens | unit_tokens
            jaccard = len(common) / max(len(union), 1)
            sequence = SequenceMatcher(None, claim_norm, unit_norm).ratio()
            score = 0.55 * coverage + 0.25 * jaccard + 0.20 * sequence
            unit_numbers = set(re.findall(r"\d+(?:\.\d+)?%?", unit_norm))
            if claim_numbers:
                score += 0.10 if claim_numbers <= unit_numbers else -0.20
        ranked.append({**unit, "retrieval_score": round(min(max(score, 0.0), 1.0), 6)})
    return sorted(
        ranked,
        key=lambda unit: (-unit["retrieval_score"], unit["evidence_id"]),
    )[:limit]


def _normalized_dates(text):
    return [
        re.sub(r"\s+", " ", normalize_evidence_text(match.group(0))).strip()
        for match in _DATE_RANGE_VALUE.finditer(text)
    ]


def _header_core(text):
    text = _plain_markdown(text)
    text = _DATE_RANGE_VALUE.sub("", text)
    text = re.sub(r"\(\s*fictional\s*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    return text.strip(" |,()-")


def _header_parts(text):
    core = _header_core(text)
    if not core:
        return []
    separator = "|" if "|" in core else "," if "," in core else None
    if separator:
        return [part.strip() for part in core.split(separator) if part.strip()]
    return [core]


def _protected_entities(resume_md):
    """Extract comparable protected values from the project's Markdown resume contract."""
    entities = {
        "candidate_name": [],
        "contact_email": [],
        "contact_phone": [],
        "employment_title": [],
        "employer": [],
        "employment_date": [],
        "degree": [],
        "institution": [],
        "education_date": [],
    }

    def add(kind, value, line):
        normalized = normalize_evidence_text(value)
        if normalized and all(item["normalized"] != normalized for item in entities[kind]):
            entities[kind].append({
                "value": value.strip(),
                "normalized": normalized,
                "line": line,
            })

    pending_experience_title = False
    pending_education_degree = False
    active_section = None
    for record in _line_records(resume_md):
        raw = record["raw"]
        if record["section"] != active_section:
            active_section = record["section"]
            pending_experience_title = False
            pending_education_degree = False
        if record["kind"] == "candidate_name":
            add("candidate_name", record["text"], record["line"])
        for match in _CONTACT_EMAIL.finditer(raw):
            add("contact_email", match.group(0), record["line"])
        for match in _CONTACT_PHONE.finditer(raw):
            digits = re.sub(r"\D", "", match.group(0))
            if len(digits) >= 8:
                add("contact_phone", digits, record["line"])
        if record["kind"] != "protected_header":
            continue

        section = record["section"]
        dates = _normalized_dates(record["text"])
        parts = _header_parts(record["text"])
        is_heading = raw.lstrip().startswith("#")
        if section in {"experience", "professional experience", "work experience"}:
            for value in dates:
                add("employment_date", value, record["line"])
            looks_like_project = any(
                "project" in normalize_evidence_text(part) for part in parts
            ) or any(
                normalize_evidence_text(part) in {"personal initiative", "personal project"}
                for part in parts
            )
            if looks_like_project:
                pending_experience_title = False
                continue
            if len(parts) >= 2:
                if pending_experience_title:
                    add("employer", parts[0], record["line"])
                    pending_experience_title = False
                else:
                    add("employment_title", parts[0], record["line"])
                    add("employer", parts[1], record["line"])
            elif parts and (
                is_heading or (raw.lstrip().startswith(("**", "__")) and not dates)
            ):
                add("employment_title", parts[0], record["line"])
                pending_experience_title = True
            elif parts:
                add("employer", parts[0], record["line"])
                pending_experience_title = False
        elif section == "education":
            for value in dates:
                add("education_date", value, record["line"])
            if len(parts) >= 2:
                if pending_education_degree:
                    add("institution", parts[0], record["line"])
                    pending_education_degree = False
                else:
                    add("degree", parts[0], record["line"])
                    add("institution", parts[1], record["line"])
            elif parts and pending_education_degree:
                add("institution", parts[0], record["line"])
                pending_education_degree = False
            elif parts:
                is_bold_degree = raw.lstrip().startswith(("**", "__")) and not dates
                if is_heading or is_bold_degree:
                    add("degree", parts[0], record["line"])
                    pending_education_degree = True
                else:
                    add("institution", parts[0], record["line"])
    return entities


def _normalized_number(value):
    value = value.lower().replace(",", "").strip()
    value = re.sub(r"\s*percent$", "%", value)
    return re.sub(r"\s+", "", value)


def protected_field_metrics(original_resume, tailored_resume):
    """Return high-confidence protected-field changes without penalizing omissions."""
    source = _protected_entities(original_resume)
    output = _protected_entities(tailored_resume)
    violations = []
    seen = set()

    def add(kind, item, expected):
        key = (kind, item["normalized"], item["line"])
        if key in seen:
            return
        seen.add(key)
        violations.append({
            "type": kind,
            "value": item["value"],
            "normalized": item["normalized"],
            "line": item["line"],
            "expected_source_values": [entry["value"] for entry in expected],
        })

    for kind, output_items in output.items():
        allowed = {item["normalized"] for item in source[kind]}
        for item in output_items:
            if item["normalized"] not in allowed:
                add(kind, item, source[kind])

    source_numbers = {
        _normalized_number(match.group(0))
        for match in _NUMBER_VALUE.finditer(original_resume)
    }
    for record in _line_records(tailored_resume):
        if record["kind"] != "claim":
            continue
        for match in _NUMBER_VALUE.finditer(record["text"]):
            normalized = _normalized_number(match.group(0))
            if normalized in source_numbers:
                continue
            add("invented_number", {
                "value": match.group(0),
                "normalized": normalized,
                "line": record["line"],
            }, [{"value": value} for value in sorted(source_numbers)])

    return {
        "protected_field_violation": bool(violations),
        "protected_field_violation_count": len(violations),
        "protected_field_violations": violations,
        "protected_source_fields": {
            kind: [item["value"] for item in values]
            for kind, values in source.items()
        },
    }


def nli_model_identity():
    from transformers import AutoConfig
    from transformers.utils.hub import cached_file

    model_config = AutoConfig.from_pretrained(config.NLI_MODEL)
    raw_mapping = dict(model_config.id2label)
    mapping = {str(label).lower(): int(index) for index, label in raw_mapping.items()}
    required = {"contradiction", "entailment", "neutral"}
    if set(mapping) != required:
        raise RuntimeError(
            f"NLI_LABEL_MAPPING_INVALID: expected {sorted(required)}, got {raw_mapping}"
        )
    revision = getattr(model_config, "_commit_hash", None)
    try:
        config_path = Path(cached_file(config.NLI_MODEL, "config.json", local_files_only=True))
        if config_path.parent.parent.name == "snapshots":
            revision = config_path.parent.name
    except (OSError, TypeError):
        pass
    return {
        "model": config.NLI_MODEL,
        "model_name_or_path": getattr(model_config, "_name_or_path", None),
        "model_revision": revision,
        "model_config_hash": hashlib.sha256(
            model_config.to_json_string().encode()
        ).hexdigest(),
        "id2label": {str(k): str(v).lower() for k, v in raw_mapping.items()},
        "label2id": mapping,
    }


def _load_nli():
    global _nli, _nli_metadata
    if _nli is None:
        from sentence_transformers import CrossEncoder
        _nli = CrossEncoder(config.NLI_MODEL)
        _nli_metadata = nli_model_identity()
    return _nli, _nli_metadata


def extract_claims(resume_md):
    """Extract substantive output claims while excluding structure and protected headers."""
    return [
        sentence
        for record in _line_records(resume_md)
        if record["kind"] == "claim"
        for sentence in _sentences(record["text"])
    ]


def _chunks(text, size=None, overlap=None):
    size = size or config.NLI_CHUNK_WORDS
    overlap = config.NLI_CHUNK_OVERLAP if overlap is None else overlap
    if size <= overlap:
        raise ValueError("NLI chunk size must be greater than overlap")
    words = text.split()
    if not words:
        return [""]
    step = size - overlap
    return [" ".join(words[index:index + size]) for index in range(0, len(words), step)]


def _probabilities(values):
    """Accept either class logits or normalized probabilities."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise RuntimeError(f"NLI_OUTPUT_SHAPE_INVALID: expected Nx3, got {values.shape}")
    if not np.isfinite(values).all():
        raise RuntimeError("NLI_OUTPUT_NONFINITE")
    if (
        np.all(values >= 0)
        and np.all(values <= 1)
        and np.allclose(values.sum(axis=1), 1.0, atol=1e-4)
    ):
        return values
    shifted = values - values.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def nli_preflight():
    """Verify labels, source/claim order, and obvious entailment/contradiction/neutral cases."""
    model, metadata = _load_nli()
    examples = [
        ("Ava writes Python software.", "Ava writes software.", "entailment"),
        ("Ava writes Python software.", "Ava has never written software.", "contradiction"),
        ("Ava writes Python software.", "Ava lives in Tokyo.", "neutral"),
    ]
    probabilities = _probabilities(
        model.predict([(source, claim) for source, claim, _ in examples])
    )
    label2id = metadata["label2id"]
    predictions = [
        max(label2id, key=lambda label: probabilities[index, label2id[label]])
        for index in range(len(examples))
    ]
    expected = [example[2] for example in examples]
    if predictions != expected:
        raise RuntimeError(
            f"NLI_PREFLIGHT_FAILED: expected {expected}, got {predictions}; "
            "check source/claim order and label mapping"
        )
    return {
        "status": "passed",
        "metadata": metadata,
        "examples": [
            {
                "source": source,
                "claim": claim,
                "expected": expected_label,
                "predicted": predicted,
                "probabilities": {
                    label: round(float(probabilities[index, label_id]), 6)
                    for label, label_id in label2id.items()
                },
            }
            for index, ((source, claim, expected_label), predicted) in enumerate(
                zip(examples, predictions)
            )
        ],
    }


def _nli_cache_key(original_resume, new_resume, metadata):
    identity = {
        "source_hash": hashlib.sha256(original_resume.encode()).hexdigest(),
        "tailored_hash": hashlib.sha256(new_resume.encode()).hexdigest(),
        "model": metadata,
        "evaluator_version": config.NLI_EVALUATOR_VERSION,
        "chunk_words": config.NLI_CHUNK_WORDS,
        "chunk_overlap": config.NLI_CHUNK_OVERLAP,
        "entail_threshold": config.ENTAIL_THRESHOLD,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(), identity


def faithfulness(original_resume, new_resume, cache_dir=None, force=False):
    """Score every tailored claim against original-resume chunks and retain claim evidence."""
    claims = extract_claims(new_resume)
    if not claims:
        return {
            "faithfulness": None,
            "unsupported": [],
            "n_claims": 0,
            "claim_scores": [],
            "nli_evaluator_version": config.NLI_EVALUATOR_VERSION,
        }
    model, metadata = _load_nli()
    cache_key, cache_identity = _nli_cache_key(original_resume, new_resume, metadata)
    cache_path = Path(cache_dir) / f"{cache_key}.json" if cache_dir else None
    if cache_path and cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))["result"]

    chunks = _chunks(original_resume)
    pairs = [(chunk, claim) for claim in claims for chunk in chunks]
    probabilities = _probabilities(model.predict(pairs))
    label2id = metadata["label2id"]
    entailment_index = label2id["entailment"]
    unsupported, claim_scores = [], []
    for claim_index, claim in enumerate(claims):
        start = claim_index * len(chunks)
        claim_probabilities = probabilities[start:start + len(chunks)]
        best_index = int(np.argmax(claim_probabilities[:, entailment_index]))
        best = claim_probabilities[best_index]
        entailment = float(best[entailment_index])
        supported = entailment >= config.ENTAIL_THRESHOLD
        if not supported:
            unsupported.append(claim)
        claim_scores.append({
            "claim": claim,
            "best_source_chunk": chunks[best_index],
            "probabilities": {
                label: round(float(best[index]), 6) for label, index in label2id.items()
            },
            "predicted_relation": max(label2id, key=lambda label: best[label2id[label]]),
            "entailment": round(entailment, 6),
            "supported": supported,
        })
    result = {
        "faithfulness": round(1 - len(unsupported) / len(claims), 4),
        "unsupported": unsupported,
        "n_claims": len(claims),
        "claim_scores": claim_scores,
        "nli_evaluator_version": config.NLI_EVALUATOR_VERSION,
        "nli_model": metadata,
        "nli_cache_key": cache_key,
    }
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({"identity": cache_identity, "result": result}, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(cache_path)
    return result


# ---------- 5. Tool-call success ----------

def tool_success(totals):
    n = totals["tool_calls"]
    return None if n == 0 else round(totals["tool_calls_ok"] / n, 4)
