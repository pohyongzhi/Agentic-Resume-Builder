import copy
import json
import unittest

from v4_data.contracts import ROOT, DataError, digest, read_jsonl
from v4_data.splitting import (
    _prepare_text,
    _similar_prepared,
    admit_sources,
    assign_splits,
    check_assignments,
    normalize,
    pair_sources,
    similar,
)

STAMP = "2026-10-24T00:00:00Z"


def set_text(source, text):
    source["text"] = text
    source["text_sha256"] = digest(text)
    source["normalized_sha256"] = digest(normalize(text))
    source["source"]["raw_sha256"] = digest(text)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.sources = read_jsonl(ROOT / "data/fixtures/v4/sources.jsonl", "source")
        self.policy = json.loads((ROOT / "configs/v4/split-policy.json").read_text())

    def assign(self, **kwargs):
        return assign_splits(self.sources, self.policy, STAMP, **kwargs)

    def test_determinism_under_input_reordering(self):
        actual = self.assign()
        self.sources.reverse()
        self.assertEqual(self.assign(), actual)
        check_assignments(self.sources, actual, self.policy)

    def test_families_templates_and_lineage_group(self):
        for relation in ("family", "template", "parent", "related"):
            with self.subTest(relation=relation):
                self.setUp()
                a, b = self.sources[:2]
                if relation == "family":
                    b["family_id"] = a["family_id"]
                elif relation == "template":
                    b["template_family_id"] = a["template_family_id"]
                elif relation == "parent":
                    b["parent_ids"] = [a["id"]]
                else:
                    b["related_family_ids"] = [a["family_id"]]
                assigned = self.assign()
                self.assertEqual(assigned[0]["component_id"], assigned[1]["component_id"])

    def test_exact_and_near_duplicate_group_across_sources(self):
        for text in (self.sources[0]["text"].upper(), self.sources[0]["text"] + " Available for shifts."):
            with self.subTest(text=text):
                set_text(self.sources[1], text)
                assigned = self.assign()
                self.assertEqual(assigned[0]["component_id"], assigned[1]["component_id"])

    def test_same_raw_source_or_upstream_identity_groups_different_cleaned_text(self):
        for identity in ("raw", "source"):
            with self.subTest(identity=identity):
                self.setUp()
                if identity == "raw":
                    self.sources[1]["source"]["raw_sha256"] = self.sources[0]["source"]["raw_sha256"]
                else:
                    self.sources[1]["source"]["source_id"] = self.sources[0]["source"]["source_id"]
                assigned = self.assign()
                self.assertEqual(assigned[0]["component_id"], assigned[1]["component_id"])

    def test_connected_paired_components_are_transitive(self):
        c1, c2, j1, j2 = [source["id"] for source in self.sources]
        pairs = [{"schema_version": "1.0", "id": f"edge-{i}", "candidate_id": c, "jd_id": j}
                 for i, (c, j) in enumerate(((c1, j1), (c2, j1), (c2, j2)))]
        assigned = self.assign(existing_pairs=pairs)
        self.assertEqual(len({row["component_id"] for row in assigned}), 1)
        check_assignments(self.sources, assigned, self.policy, pairs)
        with self.assertRaises(DataError):
            check_assignments(self.sources, assigned, self.policy)

    def test_append_preserves_splits_and_blocks_new_crossing_links(self):
        a, b = self.sources[:2]
        old = self.assign(reservations={a["id"]: "train", b["id"]: "validation"})
        extra = copy.deepcopy(a)
        extra["id"] = "candidate-added-version"
        extra["parent_ids"] = [a["id"]]
        self.sources.append(extra)
        new = self.assign(previous=old)
        self.assertEqual({r["source_id"]: r["split"] for r in old},
                         {r["source_id"]: r["split"] for r in new if r["source_id"] != extra["id"]})
        b["related_family_ids"] = [a["family_id"]]
        with self.assertRaisesRegex(DataError, "crosses frozen"):
            self.assign(previous=new)

    def test_source_mutation_and_policy_drift_are_blocked(self):
        old = self.assign()
        self.policy["seed"] = "changed"
        with self.assertRaises(DataError):
            self.assign(previous=old)
        self.setUp()
        set_text(self.sources[0], self.sources[0]["text"] + " Changed.")
        with self.assertRaises(DataError):
            self.assign(previous=old)

    def test_cross_split_pairing_and_final_benchmark_blocked(self):
        candidate, jd = self.sources[0]["id"], self.sources[2]["id"]
        assigned = self.assign(reservations={candidate: "train", jd: "validation"})
        with self.assertRaisesRegex(DataError, "Cross-split"):
            pair_sources(candidate, jd, self.sources, assigned)
        assigned = self.assign(reservations={candidate: "final_benchmark", jd: "final_benchmark"})
        with self.assertRaisesRegex(DataError, "Final benchmark"):
            pair_sources(candidate, jd, self.sources, assigned)

    def test_unknown_and_duplicate_references_fail_closed(self):
        self.sources[0]["parent_ids"] = ["missing-parent"]
        with self.assertRaises(DataError):
            self.assign()
        self.setUp()
        self.sources.append(copy.deepcopy(self.sources[0]))
        with self.assertRaises(DataError):
            self.assign()

    def test_cyclic_lineage_is_rejected(self):
        self.sources[0]["parent_ids"] = [self.sources[1]["id"]]
        self.sources[1]["parent_ids"] = [self.sources[0]["id"]]
        with self.assertRaisesRegex(DataError, "cycle"):
            self.assign()

    def test_unadmitted_and_unreviewed_sources_fail_closed(self):
        for field, value in (("use_decision", "pending"), ("privacy_review", "pending")):
            self.setUp()
            self.sources[0]["source"][field] = value
            with self.assertRaises(DataError):
                self.assign()

    def test_legacy_paths_ids_hashes_and_disguised_content_blocked(self):
        exclusion = json.loads((ROOT / "configs/v4/legacy-exclusions.json").read_text())
        for path, sha in exclusion["files"].items():
            with self.subTest(path=path):
                self.setUp()
                self.sources[0]["source"]["raw_sha256"] = sha
                with self.assertRaises(DataError):
                    admit_sources(self.sources)
                self.setUp()
                self.sources[0]["source"]["uri"] = "file:///renamed/" + path
                with self.assertRaises(DataError):
                    admit_sources(self.sources)
                self.setUp()
                text = (ROOT / path).read_text().upper()
                set_text(self.sources[0], "\n  " + text + "\n")
                with self.assertRaises(DataError):
                    admit_sources(self.sources)
        self.setUp()
        self.sources[0]["related_family_ids"] = ["cand_01"]
        with self.assertRaises(DataError):
            admit_sources(self.sources)

    def test_tampered_assignment_is_detected(self):
        rows = self.assign()
        rows[0]["component_id"] = "forged-component"
        with self.assertRaises(DataError):
            check_assignments(self.sources, rows, self.policy)

    def test_precomputed_similarity_matches_direct_similarity(self):
        samples = [s["text"] for s in self.sources] + [
            self.sources[0]["text"] + " Extra note.",
            "short text",
            "two words",
            "",
        ]
        prepared = [_prepare_text(t) for t in samples]
        for threshold in (0.5, 0.75, 0.9):
            for i, left in enumerate(samples):
                for j, right in enumerate(samples):
                    self.assertEqual(
                        _similar_prepared(prepared[i], prepared[j], threshold),
                        similar(left, right, threshold),
                    )


if __name__ == "__main__":
    unittest.main()
