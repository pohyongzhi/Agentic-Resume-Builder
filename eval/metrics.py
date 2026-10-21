"""The shared scorecard. Every workflow's output is scored by exactly these functions.

  1. cost                -> from the Tracker (tokens, seconds)          [common/llm.py]
  2. jd_similarity       -> cosine similarity of embeddings, before vs after
  3. keyword_coverage    -> % of HAND-LABELLED must-have JD keywords present (ATS-style)
  4. faithfulness        -> % of resume claims supported by the ORIGINAL resume (NLI referee)
  5. tool-call success   -> from the Tracker (valid tool calls / all tool calls)
"""
import re

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

def _load_nli():
    global _nli
    if _nli is None:
        from sentence_transformers import CrossEncoder
        _nli = CrossEncoder(config.NLI_MODEL)
    return _nli


def extract_claims(resume_md):
    """Claims = bullet points, summary sentences and entry headings such as
    '### Data Analyst, Example Ltd (2022 - 2024)', which carry titles, employers and dates.
    Section headings, names and contact lines are too short to count and are skipped."""
    claims = []
    for line in resume_md.splitlines():
        line = re.sub(r"^#+\s*", "", line.strip())
        if len(line) < 25:
            continue
        line = re.sub(r"^[-*•\d.)\s]+", "", line).strip()
        for sent in re.split(r"(?<=[.;])\s+", line):
            if len(sent) >= 25:
                claims.append(sent)
    return claims


def _chunks(text, size=60, overlap=20):
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def faithfulness(original_resume, new_resume):
    """For each claim in the new resume, take the best entailment score against any chunk of the
    original. Supported if >= threshold. This is a separate model from the writer, so it is an
    independent referee (a model grading its own work tends to be too lenient)."""
    claims = extract_claims(new_resume)
    if not claims:
        return {"faithfulness": None, "unsupported": [], "n_claims": 0}
    model = _load_nli()
    chunks = _chunks(original_resume)
    unsupported = []
    for c in claims:
        logits = model.predict([(ch, c) for ch in chunks])
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        entail = probs[:, 1].max()    # label order for this model: contradiction, entailment, neutral
        if entail < config.ENTAIL_THRESHOLD:
            unsupported.append(c)
    return {"faithfulness": round(1 - len(unsupported) / len(claims), 4),
            "unsupported": unsupported, "n_claims": len(claims)}


# ---------- 5. Tool-call success ----------

def tool_success(totals):
    n = totals["tool_calls"]
    return None if n == 0 else round(totals["tool_calls_ok"] / n, 4)
