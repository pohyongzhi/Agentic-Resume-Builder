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


# ---------- 4. Faithfulness ----------

_nli = None
_nli_metadata = None


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
    """Extract substantive claims while excluding section labels and contact-only lines."""
    claims = []
    section_labels = {
        "summary", "experience", "professional experience", "education", "skills",
        "projects", "certifications", "achievements",
    }
    for raw_line in resume_md.splitlines():
        raw_line = raw_line.strip()
        if not raw_line or re.search(r"[\w.+-]+@[\w.-]+\.\w+", raw_line):
            continue
        if re.search(r"(?:\+?\d[\d\s().-]{7,}\d)", raw_line) and "|" in raw_line:
            continue
        line = re.sub(r"^#+\s*", "", raw_line)
        if line.lower().strip("*_: ") in section_labels or len(line) < 25:
            continue
        line = re.sub(r"^[-*•\d.)\s]+", "", line).strip()
        for sentence in re.split(r"(?<=[.;])\s+", line):
            if len(sentence) >= 25:
                claims.append(sentence)
    return claims


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
