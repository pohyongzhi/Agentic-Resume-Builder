import copy
import tempfile
import unittest
from pathlib import Path

from v4_data.contracts import ROOT, DataError, read_jsonl, validate, write_jsonl


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.sources = read_jsonl(ROOT / "data/fixtures/v4/sources.jsonl", "source")

    def test_fictional_sources_and_ledgers(self):
        self.assertEqual(len(self.sources), 4)
        self.assertEqual(len(read_jsonl(ROOT / "data/fixtures/v4/ledgers.jsonl", "ledger")), 2)

    def test_missing_unknown_version_and_hash_rejected(self):
        for field, value in (("schema_version", "2.0"), ("text_sha256", "bad"), ("unexpected", True)):
            with self.subTest(field=field):
                row = copy.deepcopy(self.sources[0])
                row[field] = value
                with self.assertRaises(DataError):
                    validate(row, "source")
        row = copy.deepcopy(self.sources[0])
        del row["source"]["license"]
        with self.assertRaises(DataError):
            validate(row, "source")

    def test_jsonl_round_trip_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            path = Path(directory) / "sources.jsonl"
            write_jsonl(path, self.sources, "source")
            self.assertEqual(read_jsonl(path, "source"), self.sources)
            with self.assertRaises(FileExistsError):
                write_jsonl(path, self.sources, "source")


if __name__ == "__main__":
    unittest.main()
