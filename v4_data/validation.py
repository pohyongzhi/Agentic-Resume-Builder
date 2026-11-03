"""Deterministic rejection gates; independent review is always outstanding."""

import re
from itertools import combinations

from .contracts import DataError, digest, index_unique, validate
from .pipeline import CATEGORIES, SECTIONS, check_span, prepare_requests, privacy_findings
from .splitting import similar

VALIDATOR_VERSION = "evidence-gates-v1"


def validate_cases(cases, sources, ledgers, assignments, pairs, policy, validated_at,
                   existing_pairs=(), holdout=None):
    requests = prepare_requests(sources, ledgers, assignments, pairs, policy, existing_pairs, holdout)
    by_pair = {(r["pair"]["candidate_id"], r["pair"]["jd_id"]): r for r in requests}
    records = index_unique(sources)
    case_index = index_unique(cases)
    reports = []
    structurally_valid = []
    for case in cases:
        reasons = []
        def reject(code, path, message):
            reasons.append({"code": code, "path": path, "message": message})
        try:
            validate(case, "case")
        except DataError as exc:
            reject("schema", "case", str(exc))
        else:
            structurally_valid.append(case)
            request = by_pair.get((case["candidate_id"], case["jd_id"]))
            if request is None:
                reject("provenance", "case", "Case is not part of the prepared pair plan")
            else:
                candidate, jd = records[case["candidate_id"]], records[case["jd_id"]]
                expected_family = "generation-" + digest([candidate["family_id"], jd["family_id"]])[:24]
                if (case["candidate_family_id"] != candidate["family_id"] or case["jd_family_id"] != jd["family_id"]
                        or case["generation_family_id"] != expected_family or case["split"] != request["split"]):
                    reject("split_leakage", "split", "Case split or family differs from the frozen source assignments")
                if case["input"] != request["input"]:
                    reject("hash_mismatch", "input", "Case input differs from the admitted source text")
                if case["ledger_id"] != request["ledger"]["id"] or case["ledger_sha256"] != request["ledger_sha256"]:
                    reject("ledger", "ledger_id", "Case refers to a different ledger version")
                target = case["target"]["resume_md"]
                expected_hashes = {"candidate_sha256": candidate["text_sha256"], "jd_sha256": jd["text_sha256"],
                                   "target_sha256": digest(target), "assignments_sha256": request["assignments_sha256"]}
                if case["hashes"] != expected_hashes:
                    reject("hash_mismatch", "hashes", "Input, target or assignment hash mismatch")
                generation = case["generation"]
                if (generation["prompt_version"] != request["prompt_version"]
                        or generation["prompt_sha256"] != request["prompt_sha256"]
                        or generation["request_sha256"] != digest(request)
                        or generation["source_ids"] != [candidate["id"], jd["id"]]):
                    reject("provenance", "generation", "Generation provenance differs from the prepared request")
                expected_id = "case-" + digest([request["id"], target, generation])[:24]
                if case["id"] != expected_id:
                    reject("hash_mismatch", "id", "Case ID does not match request, target and provenance")
                if generation["started_at"] > generation["completed_at"]:
                    reject("provenance", "generation", "Generation timestamps are reversed")
                if generation["backend"] == "fixture":
                    if (generation["model_id"] != "extractive-fixture-renderer" or generation["model_revision"] != "1"
                            or generation["cost_usd"] != 0 or generation["usage_status"] != "not-applicable"
                            or generation["input_tokens"] != 0 or generation["output_tokens"] != 0
                            or any(not s["source"]["fictional"] or s["source"]["use_decision"] != "fixture-only" for s in (candidate, jd))):
                        reject("provenance", "generation", "Invalid fixture provenance")
                elif generation["usage_status"] != "reported":
                    reject("provenance", "generation", "Teacher token usage must be reported")
                parent_ids = generation["parent_case_ids"]
                if generation["attempt"] == 1 and parent_ids:
                    reject("provenance", "generation", "First attempt cannot have a retry parent")
                if generation["attempt"] == 2:
                    parent = case_index.get(parent_ids[0]) if len(parent_ids) == 1 else None
                    try:
                        validate(parent, "case")
                    except DataError:
                        parent = None
                    if (parent is None or parent["generation"]["attempt"] != 1
                            or parent["generation"]["run_id"] != generation["run_id"]
                            or (parent["candidate_id"], parent["jd_id"]) != (candidate["id"], jd["id"])):
                        reject("provenance", "generation", "Retry requires its first attempt in the same batch and pair")

                if any(privacy_findings(text) for text in (target, candidate["text"], jd["text"])):
                    reject("privacy", "text", "Contact or identifier pattern requires removal and review")
                facts = index_unique(request["ledger"]["facts"])
                covered = set()
                for mapping in case["target_evidence"]:
                    span = mapping["output"]
                    if not check_span(span, target) or set(mapping["fact_ids"]) - facts.keys():
                        reject("evidence", "target_evidence", "Invalid output span or unknown fact ID")
                        continue
                    if mapping["relation"] == "verbatim" and (len(mapping["fact_ids"]) != 1 or
                            span["text"] != facts[mapping["fact_ids"][0]]["claim"]):
                        reject("unsupported_claim", "target_evidence", "Verbatim claim differs from its cited fact")
                    numbers = lambda text: set(re.findall(r"(?<!\w)\d+(?:[.,]\d+)*(?:%|\b)", text))
                    cited = " ".join(facts[fact_id]["claim"] for fact_id in mapping["fact_ids"])
                    if numbers(span["text"]) - numbers(cited):
                        reject("unsupported_claim", "target_evidence", "Number or date absent from the cited facts")
                    positions = set(range(span["start"], span["end"]))
                    if positions & covered:
                        reject("evidence", "target_evidence", "Overlapping evidence mappings")
                    covered.update(positions)
                headings = re.findall(r"^## (.+)$", target, flags=re.MULTILINE)
                if headings != list(SECTIONS):
                    reject("format", "target.resume_md", "Expected exactly Summary, Experience, Education and Skills in order")
                offset, section = 0, None
                section_lines = {name: 0 for name in SECTIONS}
                for line in target.splitlines(keepends=True):
                    content = line.strip()
                    if content.startswith("## ") and content[3:] in SECTIONS:
                        section = content[3:]
                    elif content == "Not provided":
                        if section is not None:
                            section_lines[section] += 1
                        if section is None or any(f["category"] == CATEGORIES[section] for f in facts.values()):
                            reject("unsupported_claim", "target.resume_md", "Not provided hides evidence supplied for this section")
                    elif content:
                        if section is None:
                            reject("format", "target.resume_md", "Content appears before the Summary heading")
                        else:
                            section_lines[section] += 1
                        required = {offset + i for i, char in enumerate(line) if not char.isspace()}
                        if not required <= covered:
                            reject("evidence", "target.resume_md", "Unmapped output claim, including short lines or skills")
                    offset += len(line)
                if any(count == 0 for count in section_lines.values()):
                    reject("format", "target.resume_md", "Empty section must explicitly state Not provided")
        reports.append({"schema_version": "1.0", "case_id": case["id"], "case_sha256": digest(case),
                        "validator_version": VALIDATOR_VERSION, "validated_at": validated_at,
                        "status": "rejected" if reasons else "needs_review", "reasons": reasons,
                        "checks": ["schema", "source_admission", "source_splits", "hashes", "provenance", "evidence", "format", "privacy_patterns"]})

    reports_by_id = {r["case_id"]: r for r in reports}
    for left, right in combinations(structurally_valid, 2):
        if left["split"] != right["split"] and (left["generation_family_id"] == right["generation_family_id"] or
                similar(left["target"]["resume_md"], right["target"]["resume_md"], policy["near_duplicate_threshold"])):
            for case in (left, right):
                report = reports_by_id[case["id"]]
                report["status"] = "rejected"
                report["reasons"].append({"code": "split_leakage", "path": "target.resume_md",
                                          "message": "Target or generation family overlaps another split"})
    for report in reports:
        if report["status"] == "needs_review":
            report["reasons"].append({"code": "review_required", "path": "case",
                                      "message": "Independent semantic, relevance and human review required; not eligible for training"})
        # Repeated failures may have the same explanation; JSONL uses unique reasons.
        report["reasons"] = list({digest(reason): reason for reason in report["reasons"]}.values())
        validate(report, "validation")
    return reports
