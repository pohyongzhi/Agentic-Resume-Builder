"""Source admission and deterministic, append-safe component splitting."""

import hashlib
import json
import re
import unicodedata
from collections import deque
from difflib import SequenceMatcher
from itertools import combinations
from urllib.parse import unquote

from .contracts import ROOT, DataError, canonical, digest, index_unique, validate


def normalize(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def _prepare_text(text):
    norm = normalize(text)
    words = norm.split()
    shingles = frozenset(tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2)))
    g4 = frozenset(norm[i:i + 4] for i in range(max(0, len(norm) - 3)))
    return norm, len(norm), shingles, len(shingles), g4, len(g4)


def _similar_prepared(left_prep, right_prep, threshold):
    a, len_a, sa, len_sa, g4_a, len_g4a = left_prep
    b, len_b, sb, len_sb, g4_b, len_g4b = right_prep
    if a == b:
        return True
    if min(len_a, len_b) >= 30 and (a in b or b in a):
        return True
    if len_sa and len_sb and min(len_sa, len_sb) >= threshold * max(len_sa, len_sb):
        inter = len(sa & sb)
        if inter / (len_sa + len_sb - inter) >= threshold:
            return True
    if not len_a or not len_b:
        return False
    if (2.0 * min(len_a, len_b)) / (len_a + len_b) < threshold:
        return False
    if min(len_a, len_b) >= 80 and len_g4a and len_g4b:
        if (2.0 * len(g4_a & g4_b)) / (len_g4a + len_g4b) < threshold * 0.45:
            return False
    # SequenceMatcher's autojunk would hide repeated template words.
    return SequenceMatcher(None, a, b, autojunk=False).ratio() >= threshold


def similar(left, right, threshold):
    """Conservative lexical check; declared families cover semantic variants."""
    return _similar_prepared(_prepare_text(left), _prepare_text(right), threshold)


class Exclusions:
    """Pinned legacy fixture paths, hashes, IDs and normalized content."""

    def __init__(self):
        config = json.loads((ROOT / "configs/v4/legacy-exclusions.json").read_text())
        self.paths = config["files"]
        self.ids = set(config["source_ids"])
        self.hashes = set(self.paths.values())
        self.texts = []
        for relative, expected in self.paths.items():
            path = ROOT / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise DataError(f"Legacy exclusion inventory changed: {relative}; review its exclusion manifest")
            self.texts.append(path.read_text(encoding="utf-8"))
        self.prepared = [_prepare_text(text) for text in self.texts]

    def check(self, source):
        metadata = canonical({k: v for k, v in source.items() if k != "text"})
        decoded = unquote(metadata).replace("\\\\", "/").casefold()
        if any(path.casefold() in decoded for path in self.paths):
            raise DataError("Excluded legacy fixture path or lineage")
        ids = {source["id"], source["source"]["source_id"], source["family_id"],
               source["template_family_id"], *source["parent_ids"], *source["related_family_ids"]}
        if ids & self.ids or any(h in metadata for h in self.hashes):
            raise DataError("Excluded legacy fixture ID or hash")
        src_prep = _prepare_text(source["text"])
        if any(_similar_prepared(src_prep, prep, 0.75) for prep in self.prepared):
            raise DataError("Excluded legacy fixture content or near-duplicate")


def admit_sources(sources):
    records = index_unique(sources)
    exclusions = Exclusions()
    for source in sources:
        validate(source, "source")
        if source["source"]["use_decision"] not in {"fixture-only", "local-approved"}:
            raise DataError(f"Source use not admitted: {source['id']}")
        if source["source"]["privacy_review"] != "passed":
            raise DataError(f"Source privacy review pending: {source['id']}")
        if source["text_sha256"] != digest(source["text"]) or source["normalized_sha256"] != digest(normalize(source["text"])):
            raise DataError(f"Source hash mismatch: {source['id']}")
        if not normalize(source["text"]):
            raise DataError("Empty normalized source")
        if set(source["parent_ids"]) - records.keys():
            raise DataError(f"Unresolved source lineage: {source['id']}")
        exclusions.check(source)
    children = {key: [] for key in records}
    remaining = {key: len(source["parent_ids"]) for key, source in records.items()}
    for key, source in records.items():
        for ancestor in source["parent_ids"]:
            children[ancestor].append(key)
    ready = deque(key for key, count in remaining.items() if count == 0)
    visited = 0
    while ready:
        visited += 1
        for child in children[ready.popleft()]:
            remaining[child] -= 1
            if remaining[child] == 0:
                ready.append(child)
    if visited != len(records):
        raise DataError("Source parent lineage contains a cycle")
    return records


def assign_splits(sources, policy, created_at, existing_pairs=(), previous=(), reservations=None):
    """Union all known relations before allocation; never move an old source.

    Reservations explicitly select evaluator-owned final benchmark families.
    Existing pairs are graph edges, never a randomly split table of targets.
    """
    validate(policy, "policy")
    if policy["train_fraction"] + policy["validation_fraction"] > 1:
        raise DataError("Split fractions exceed one")
    records = admit_sources(sources)
    parent = {key: key for key in records}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    def union(a, b):
        x, y = find(a), find(b)
        parent[max(x, y)] = min(x, y)

    families, originals = {}, {}
    for source in sources:
        metadata = source["source"]
        for identity in (("raw", metadata["raw_sha256"]), ("source", metadata["dataset"], metadata["source_id"])):
            if identity in originals:
                union(source["id"], originals[identity])
            originals[identity] = source["id"]
        for family in {source["family_id"], source["template_family_id"], *source["related_family_ids"]}:
            # Globally namespaced family IDs also allow declared cross-kind links.
            if family in families:
                union(source["id"], families[family])
            families[family] = source["id"]
        for ancestor in source["parent_ids"]:
            union(source["id"], ancestor)
    prepared = {key: _prepare_text(records[key]["text"]) for key in records}
    for left, right in combinations(sorted(records), 2):
        if find(left) != find(right) and _similar_prepared(
                prepared[left], prepared[right], policy["near_duplicate_threshold"]):
            union(left, right)
    index_unique(existing_pairs)
    for pair in existing_pairs:
        validate(pair, "pair")
        c, j = pair["candidate_id"], pair["jd_id"]
        if c not in records or j not in records or records[c]["kind"] != "candidate" or records[j]["kind"] != "jd":
            raise DataError("Invalid existing pair reference")
        union(c, j)

    old = index_unique(previous, "source_id")
    if old.keys() - records.keys():
        raise DataError("Previous assignments require the complete source inventory")
    pins = {}
    policy_hash = digest(policy)
    for key, assignment in old.items():
        validate(assignment, "assignment")
        if assignment["source_sha256"] != records[key]["text_sha256"] or assignment["family_id"] != records[key]["family_id"]:
            raise DataError("Previously assigned source changed; register a new version with lineage")
        if assignment["policy_sha256"] != policy_hash:
            raise DataError("Split policy changed; explicit migration required")
        pins.setdefault(find(key), set()).add(assignment["split"])
    for key, split in (reservations or {}).items():
        if key not in records:
            raise DataError("Unknown reserved source")
        validate(split, "split")
        pins.setdefault(find(key), set()).add(split)
    if any(len(values) > 1 for values in pins.values()):
        raise DataError("Connected component crosses frozen split assignments")

    splits = {}
    for key in sorted(records):
        component = find(key)
        if component not in splits:
            number = int(digest([policy["seed"], component])[:16], 16) / 2**64
            default = ("train" if number < policy["train_fraction"] else
                       "validation" if number < policy["train_fraction"] + policy["validation_fraction"] else "synthetic_test")
            splits[component] = next(iter(pins[component])) if component in pins else default
    return [validate({"schema_version": "1.0", "source_id": key,
                      "source_sha256": records[key]["text_sha256"],
                      "family_id": records[key]["family_id"],
                      "component_id": "component-" + digest(find(key))[:24],
                      "split": splits[find(key)], "policy_sha256": policy_hash,
                      "created_at": old[key]["created_at"] if key in old else created_at}, "assignment")
            for key in sorted(records)]


def check_assignments(sources, assignments, policy, existing_pairs=()):
    """Rebuild the graph and detect tampered memberships or split collisions."""
    if {row["source_id"] for row in assignments} != {row["id"] for row in sources}:
        raise DataError("Assignments must cover exactly the supplied source inventory")
    rebuilt = assign_splits(sources, policy, "2000-01-01T00:00:00Z", existing_pairs, assignments)
    if sorted(assignments, key=lambda r: r["source_id"]) != rebuilt:
        raise DataError("Assignment component metadata mismatch")


def pair_sources(candidate_id, jd_id, sources, assignments):
    records = index_unique(sources)
    assigned = index_unique(assignments, "source_id")
    if candidate_id not in records or jd_id not in records or candidate_id not in assigned or jd_id not in assigned:
        raise DataError("Unknown pair source")
    if records[candidate_id]["kind"] != "candidate" or records[jd_id]["kind"] != "jd":
        raise DataError("Pair must contain one candidate and one JD")
    split = assigned[candidate_id]["split"]
    if split != assigned[jd_id]["split"]:
        raise DataError("Cross-split pairing is forbidden")
    if split == "final_benchmark":
        raise DataError("Final benchmark is unavailable to generation")
    return {"schema_version": "1.0", "id": "pair-" + digest([candidate_id, jd_id])[:24],
            "candidate_id": candidate_id, "jd_id": jd_id}
