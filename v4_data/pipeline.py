"""Offline request preparation and bounded fictional fixture rendering.

There is deliberately no network client, model loader, trainer or acceptance
export here. Real teacher responses enter through the archival case contract.
"""

import re

from .contracts import ROOT, DataError, digest, index_unique, validate
from .holdout import check_holdout
from .splitting import check_assignments, pair_sources

PROMPT_VERSION = "teacher-v1"
PROMPT = (ROOT / "prompts/v4/teacher-v1.txt").read_text(encoding="utf-8")
SECTIONS = ("Summary", "Experience", "Education", "Skills")
CATEGORIES = dict(zip(SECTIONS, ("summary", "experience", "education", "skill")))


def check_span(span, text):
    return 0 <= span["start"] < span["end"] <= len(text) and text[span["start"]:span["end"]] == span["text"]


def check_ledger(ledger, candidate):
    validate(ledger, "ledger")
    if candidate["kind"] != "candidate" or ledger["candidate_id"] != candidate["id"] or ledger["candidate_sha256"] != candidate["text_sha256"]:
        raise DataError("Fact ledger is bound to a different candidate or source version")
    index_unique(ledger["facts"])
    for fact in ledger["facts"]:
        if not check_span(fact["evidence"], candidate["text"]):
            raise DataError("Fact evidence is not a valid verbatim source span")
        # The foundation permits extractive facts only. Semantic normalization
        # needs its own independent review before extending this contract.
        if fact["claim"] != fact["evidence"]["text"]:
            raise DataError("Fact claim must match its visible evidence exactly")


def build_request(pair, sources, ledger, assignments):
    records = index_unique(sources)
    valid_pair = pair_sources(pair["candidate_id"], pair["jd_id"], sources, assignments)
    if pair != valid_pair:
        raise DataError("Pair ID does not match its stable source IDs")
    candidate, jd = records[pair["candidate_id"]], records[pair["jd_id"]]
    check_ledger(ledger, candidate)
    assigned = index_unique(assignments, "source_id")
    payload = {"pair": pair, "split": assigned[candidate["id"]]["split"],
               "prompt_version": PROMPT_VERSION, "prompt_sha256": digest(PROMPT),
               "system": PROMPT, "input": {"original_resume": candidate["text"], "target_jd": jd["text"]},
               "ledger": ledger, "ledger_sha256": digest(ledger),
               "assignments_sha256": digest(sorted(assignments, key=lambda a: a["source_id"]))}
    return validate({"schema_version": "1.0", "id": "request-" + digest(payload)[:24], **payload}, "request")


def prepare_requests(sources, ledgers, assignments, pairs, policy, existing_pairs=(), holdout=None):
    """Validate the complete inventory before constructing any teacher request."""
    check_assignments(sources, assignments, policy, existing_pairs)
    if any(a["split"] == "final_benchmark" for a in assignments):
        raise DataError("Generator inventory must not contain final benchmark sources")
    check_holdout(sources, holdout, policy)
    if any(privacy_findings(source["text"]) for source in sources):
        raise DataError("Contact or identifier pattern in a source; sanitize and review before request preparation")
    index_unique(ledgers)
    by_candidate = index_unique(ledgers, "candidate_id")
    records = index_unique(sources)
    for ledger in ledgers:
        if ledger["candidate_id"] not in records:
            raise DataError("Ledger references an unknown candidate")
        check_ledger(ledger, records[ledger["candidate_id"]])
    index_unique(pairs)
    requests = []
    for pair in sorted(pairs, key=lambda row: row["id"]):
        validate(pair, "pair")
        if pair["candidate_id"] not in by_candidate:
            raise DataError("Pair has no candidate fact ledger")
        requests.append(build_request(pair, sources, by_candidate[pair["candidate_id"]], assignments))
    return requests


def render_fixture(request, sources, run_id, created_at, code_revision):
    """Copy visible facts into sections for software testing, not SFT quality."""
    validate(request, "request")
    records = index_unique(sources)
    pair = request["pair"]
    candidate, jd = records[pair["candidate_id"]], records[pair["jd_id"]]
    for source in (candidate, jd):
        if not source["source"]["fictional"] or source["source"]["use_decision"] != "fixture-only":
            raise DataError("Fixture renderer accepts only fictional test sources")
    target, mappings = "", []
    for section in SECTIONS:
        target += f"## {section}\n"
        facts = [fact for fact in request["ledger"]["facts"] if fact["category"] == CATEGORIES[section]]
        for fact in facts:
            start = len(target)
            target += fact["claim"] + "\n"
            mappings.append({"output": {"start": start, "end": len(target) - 1, "text": fact["claim"]},
                             "fact_ids": [fact["id"]], "relation": "verbatim"})
        if not facts:
            target += "Not provided\n"
        target += "\n"
    generation = {"schema_version": "1.0", "run_id": run_id, "backend": "fixture",
                  "model_id": "extractive-fixture-renderer", "model_revision": "1",
                  "prompt_version": PROMPT_VERSION, "prompt_sha256": digest(PROMPT),
                  "request_sha256": digest(request), "parameters": {"temperature": 0, "seed": 0},
                  "attempt": 1, "parent_case_ids": [], "started_at": created_at, "completed_at": created_at,
                  "input_tokens": 0, "output_tokens": 0, "usage_status": "not-applicable", "cost_usd": 0,
                  "code_revision": code_revision, "source_ids": [candidate["id"], jd["id"]]}
    return validate({"schema_version": "1.0", "id": "case-" + digest([request["id"], target, generation])[:24],
                     "generation_family_id": "generation-" + digest([candidate["family_id"], jd["family_id"]])[:24],
                     "candidate_id": candidate["id"], "jd_id": jd["id"],
                     "candidate_family_id": candidate["family_id"], "jd_family_id": jd["family_id"],
                     "split": request["split"], "created_at": created_at,
                     "ledger_id": request["ledger"]["id"], "ledger_sha256": request["ledger_sha256"],
                     "input": request["input"], "target": {"resume_md": target},
                     "hashes": {"candidate_sha256": candidate["text_sha256"], "jd_sha256": jd["text_sha256"],
                                "target_sha256": digest(target), "assignments_sha256": request["assignments_sha256"]},
                     "target_evidence": mappings, "generation": generation}, "case")


def privacy_findings(text):
    """Signals for review, not a complete de-identification system."""
    patterns = (r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
                r"(?<!\w)(?:\+\d{1,3}[ -]?)?(?:\d[ ()-]?){8,15}(?!\w)",
                r"(?i)\b[STFGM]\d{7}[A-Z]\b")
    return any(re.search(pattern, text) for pattern in patterns)
