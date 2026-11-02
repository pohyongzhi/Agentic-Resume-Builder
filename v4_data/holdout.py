"""Evaluator-created exclusion fingerprints; no benchmark text in requests."""

from .contracts import DataError, digest, index_unique, validate
from .splitting import check_assignments, normalize


def shingles(text):
    words = normalize(text).split()
    if len(words) < 3:
        return {digest(words)}
    return {digest(words[i:i + 3]) for i in range(len(words) - 2)}


def family_ids(source):
    return {source["family_id"], source["template_family_id"], *source["related_family_ids"]}


def freeze_holdout(sources, assignments, policy, created_at, protocol_sha256, existing_pairs=()):
    check_assignments(sources, assignments, policy, existing_pairs)
    records = index_unique(sources)
    entries = []
    for assignment in sorted(assignments, key=lambda a: a["source_id"]):
        if assignment["split"] != "final_benchmark":
            continue
        source = records[assignment["source_id"]]
        if source["source"]["use_decision"] == "fixture-only":
            raise DataError("Software fixtures cannot become a final benchmark")
        entries.append({"source_id": source["id"], "family_ids": sorted(family_ids(source)),
                        "raw_sha256": source["source"]["raw_sha256"], "text_sha256": source["text_sha256"],
                        "normalized_sha256": source["normalized_sha256"], "shingles": sorted(shingles(source["text"]))})
    if not entries:
        raise DataError("No final benchmark sources to freeze")
    payload = {"schema_version": "1.0", "created_at": created_at, "protocol_sha256": protocol_sha256,
               "policy_sha256": digest(policy), "assignments_sha256": digest(sorted(assignments, key=lambda a: a["source_id"])),
               "fingerprint_version": "word-trigrams-v1", "entries": entries}
    return validate({"id": "holdout-" + digest(payload)[:24], **payload}, "holdout")


def check_holdout(sources, holdout, policy):
    if holdout is None:
        if any(s["source"]["use_decision"] != "fixture-only" for s in sources):
            raise DataError("Non-fixture preparation requires a frozen holdout exclusion manifest")
        return
    validate(holdout, "holdout")
    payload = {k: v for k, v in holdout.items() if k != "id"}
    if holdout["id"] != "holdout-" + digest(payload)[:24] or holdout["policy_sha256"] != digest(policy):
        raise DataError("Holdout manifest hash or policy mismatch")
    index_unique(holdout["entries"], "source_id")
    for source in sources:
        hashes = {source["text_sha256"], source["normalized_sha256"], source["source"]["raw_sha256"]}
        current = shingles(source["text"])
        for entry in holdout["entries"]:
            known = set(entry["shingles"])
            overlap = len(current & known)
            lexical_match = overlap / min(len(current), len(known)) >= 0.75
            if (source["id"] == entry["source_id"] or entry["source_id"] in source["parent_ids"]
                    or family_ids(source) & set(entry["family_ids"])
                    or hashes & {entry["raw_sha256"], entry["text_sha256"], entry["normalized_sha256"]}
                    or lexical_match):
                raise DataError("Source collides with the frozen final benchmark")
