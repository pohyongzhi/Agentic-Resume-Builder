import copy
import json
import unittest

from v4_data.contracts import ROOT, DataError, digest, parse_json, read_jsonl, validate
from v4_data.holdout import check_holdout, freeze_holdout
from v4_data.pipeline import check_ledger, prepare_requests, render_fixture
from v4_data.splitting import assign_splits, normalize, pair_sources
from v4_data.validation import validate_cases

STAMP = "2026-10-24T00:00:00Z"


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.sources = read_jsonl(ROOT / "data/fixtures/v4/sources.jsonl", "source")
        self.ledgers = read_jsonl(ROOT / "data/fixtures/v4/ledgers.jsonl", "ledger")
        self.policy = json.loads((ROOT / "configs/v4/split-policy.json").read_text())
        self.assignments = assign_splits(self.sources, self.policy, STAMP,
                                         reservations={s["id"]: "train" for s in self.sources})
        self.pairs = [pair_sources(self.sources[c]["id"], self.sources[j]["id"], self.sources, self.assignments)
                      for c, j in ((0, 2), (1, 3))]
        self.requests = prepare_requests(self.sources, self.ledgers, self.assignments, self.pairs, self.policy)
        self.cases = [render_fixture(r, self.sources, "test-run", STAMP, "test-revision") for r in self.requests]

    def reports(self):
        return validate_cases(self.cases, self.sources, self.ledgers, self.assignments, self.pairs, self.policy, STAMP)

    def rehash(self, case):
        request = next(r for r in self.requests if r["pair"]["candidate_id"] == case["candidate_id"])
        case["hashes"]["target_sha256"] = digest(case["target"]["resume_md"])
        case["id"] = "case-" + digest([request["id"], case["target"]["resume_md"], case["generation"]])[:24]

    def test_reproducible_fixture_rendering_needs_independent_review(self):
        again = [render_fixture(r, self.sources, "test-run", STAMP, "test-revision") for r in self.requests]
        self.assertEqual(again, self.cases)
        for report in self.reports():
            self.assertEqual(report["status"], "needs_review")
            self.assertEqual([r["code"] for r in report["reasons"]], ["review_required"])
        sparse = next(c for c in self.cases if c["candidate_id"] == "candidate-orbit-v1")
        self.assertIn("## Education\nNot provided", sparse["target"]["resume_md"])

    def test_hidden_or_fabricated_ledger_facts_rejected(self):
        ledger = copy.deepcopy(self.ledgers[0])
        ledger["facts"][0]["claim"] = "Expert in Kubernetes"
        with self.assertRaises(DataError):
            check_ledger(ledger, self.sources[0])
        ledger["facts"][0]["evidence"]["text"] = "Expert in Kubernetes"
        with self.assertRaises(DataError):
            check_ledger(ledger, self.sources[0])

    def test_forged_ledger_hash_or_wrong_candidate_rejected(self):
        ledger = copy.deepcopy(self.ledgers[0])
        ledger["candidate_sha256"] = "0" * 64
        with self.assertRaises(DataError):
            check_ledger(ledger, self.sources[0])
        with self.assertRaises(DataError):
            check_ledger(self.ledgers[0], self.sources[1])

    def test_invalid_case_schema_reported(self):
        self.cases[0]["unexpected"] = True
        self.assertEqual(self.reports()[0]["reasons"][0]["code"], "schema")

    def test_unmapped_short_skill_and_contact_rejected(self):
        self.cases[0]["target"]["resume_md"] += "Kubernetes\nfictional@example.com\n"
        self.rehash(self.cases[0])
        codes = {r["code"] for r in self.reports()[0]["reasons"]}
        self.assertTrue({"evidence", "privacy"} <= codes)

    def test_invalid_spans_unknown_facts_and_overlap_rejected(self):
        for mode in ("span", "fact", "overlap"):
            with self.subTest(mode=mode):
                self.setUp()
                mapping = self.cases[0]["target_evidence"][0]
                if mode == "span":
                    mapping["output"]["start"] += 1
                elif mode == "fact":
                    mapping["fact_ids"] = ["unknown-fact"]
                else:
                    second = copy.deepcopy(mapping)
                    second["relation"] = "paraphrase"
                    self.cases[0]["target_evidence"].append(second)
                self.assertIn("evidence", {r["code"] for r in self.reports()[0]["reasons"]})

    def test_claim_cannot_borrow_different_fact_as_evidence(self):
        self.cases[0]["target_evidence"][0]["fact_ids"] = [self.requests[0]["ledger"]["facts"][-1]["id"]]
        self.assertIn("unsupported_claim", {r["code"] for r in self.reports()[0]["reasons"]})

    def test_numeric_paraphrase_is_checked_against_cited_facts(self):
        case = self.cases[0]
        mapping = case["target_evidence"][-1]
        old = mapping["output"]["text"]
        new = old + " Improved efficiency by 97%."
        start, end = mapping["output"]["start"], mapping["output"]["end"]
        case["target"]["resume_md"] = case["target"]["resume_md"][:start] + new + case["target"]["resume_md"][end:]
        mapping["output"] = {"start": start, "end": start + len(new), "text": new}
        mapping["relation"] = "paraphrase"
        self.rehash(case)
        self.assertIn("unsupported_claim", {r["code"] for r in self.reports()[0]["reasons"]})

    def test_target_hash_and_prompt_provenance_drift_rejected(self):
        self.cases[0]["hashes"]["target_sha256"] = "0" * 64
        self.cases[0]["generation"]["prompt_sha256"] = "0" * 64
        codes = {r["code"] for r in self.reports()[0]["reasons"]}
        self.assertTrue({"hash_mismatch", "provenance"} <= codes)

    def test_case_split_and_family_tampering_rejected(self):
        self.cases[0]["split"] = "validation"
        self.cases[0]["candidate_family_id"] = "invented-family"
        self.assertIn("split_leakage", {r["code"] for r in self.reports()[0]["reasons"]})

    def test_retry_requires_original_attempt_and_stops_at_two(self):
        retry = copy.deepcopy(self.cases[0])
        retry["generation"]["attempt"] = 2
        retry["generation"]["parent_case_ids"] = [self.cases[0]["id"]]
        self.rehash(retry)
        self.cases.append(retry)
        self.assertEqual(self.reports()[-1]["status"], "needs_review")
        self.cases = [retry]
        self.assertIn("provenance", {r["code"] for r in self.reports()[0]["reasons"]})
        retry["generation"]["attempt"] = 3
        with self.assertRaises(DataError):
            validate(retry, "case")

    def test_bad_format_rejected(self):
        self.cases[0]["target"]["resume_md"] = self.cases[0]["target"]["resume_md"].replace("## Skills", "## Tools")
        self.rehash(self.cases[0])
        self.assertIn("format", {r["code"] for r in self.reports()[0]["reasons"]})

    def test_non_fixture_preparation_requires_holdout(self):
        self.sources[0]["source"]["use_decision"] = "local-approved"
        with self.assertRaisesRegex(DataError, "frozen holdout"):
            prepare_requests(self.sources, self.ledgers, self.assignments, self.pairs, self.policy)
        with self.assertRaises(DataError):
            render_fixture(self.requests[0], self.sources, "test-run", STAMP, "test-revision")

    def test_final_text_never_enters_request_preparation(self):
        self.assignments = assign_splits(self.sources, self.policy, STAMP,
                                         reservations={s["id"]: "final_benchmark" for s in self.sources})
        with self.assertRaisesRegex(DataError, "final benchmark"):
            prepare_requests(self.sources, self.ledgers, self.assignments, self.pairs, self.policy)

    def test_contact_flags_stop_request_preparation(self):
        source = self.sources[2]
        source["text"] += " Contact fictional@example.com."
        source["text_sha256"] = digest(source["text"])
        source["normalized_sha256"] = digest(normalize(source["text"]))
        source["source"]["raw_sha256"] = source["text_sha256"]
        self.assignments = assign_splits(self.sources, self.policy, STAMP,
                                         reservations={s["id"]: "train" for s in self.sources})
        with self.assertRaisesRegex(DataError, "Contact"):
            prepare_requests(self.sources, self.ledgers, self.assignments, self.pairs, self.policy)


class HoldoutTests(unittest.TestCase):
    def setUp(self):
        self.sources = read_jsonl(ROOT / "data/fixtures/v4/sources.jsonl", "source")
        self.policy = json.loads((ROOT / "configs/v4/split-policy.json").read_text())
        # Only an in-memory software test; this does not freeze a real benchmark.
        self.sources[0]["source"]["use_decision"] = "local-approved"
        self.assignments = assign_splits(self.sources, self.policy, STAMP,
                                         reservations={s["id"]: "final_benchmark" if i == 0 else "train"
                                                       for i, s in enumerate(self.sources)})
        self.holdout = freeze_holdout(self.sources, self.assignments, self.policy, STAMP, digest("test-protocol"))

    def test_no_text_is_in_fingerprint_manifest(self):
        self.assertNotIn(self.sources[0]["text"], json.dumps(self.holdout))
        check_holdout(self.sources[1:], self.holdout, self.policy)

    def test_exact_near_and_related_source_excluded_without_benchmark_text(self):
        for mode in ("exact", "near", "family"):
            with self.subTest(mode=mode):
                source = copy.deepcopy(self.sources[0])
                source["id"] = "renamed-source"
                if mode in {"exact", "near"}:
                    source["family_id"] = "unrelated-family"
                    source["template_family_id"] = "unrelated-template"
                if mode == "near":
                    source["text"] += " Available immediately."
                elif mode == "family":
                    source["text"] = "Completely rewritten unrelated-looking document."
                source["source"]["raw_sha256"] = digest(source["text"])
                source["text_sha256"] = digest(source["text"])
                source["normalized_sha256"] = digest(normalize(source["text"]))
                with self.assertRaisesRegex(DataError, "collides"):
                    check_holdout([source], self.holdout, self.policy)

    def test_fixtures_cannot_be_frozen_as_benchmark(self):
        self.sources[0]["source"]["use_decision"] = "fixture-only"
        with self.assertRaises(DataError):
            freeze_holdout(self.sources, self.assignments, self.policy, STAMP, digest("test-protocol"))

    def test_manifest_tampering_detected(self):
        self.holdout["entries"][0]["text_sha256"] = "0" * 64
        with self.assertRaises(DataError):
            check_holdout(self.sources[1:], self.holdout, self.policy)


class StrictJSONTests(unittest.TestCase):
    def test_duplicate_keys_nonfinite_and_invalid_dates_rejected(self):
        for raw in ('{"id": "first", "id": "second"}', '{"score": NaN}', '{"score": Infinity}'):
            with self.assertRaises(DataError):
                parse_json(raw)
        with self.assertRaises(DataError):
            validate("2026-02-30T00:00:00Z", "timestamp")


if __name__ == "__main__":
    unittest.main()
