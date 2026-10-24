import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from v4_data.contracts import ROOT, read_jsonl, validate, write_jsonl


class CLITests(unittest.TestCase):
    def setUp(self):
        (ROOT / "runs").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="v4-cli-test-", dir=ROOT / "runs")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def command(self, *args):
        return subprocess.run([sys.executable, "-m", "v4_data", *map(str, args)], cwd=ROOT,
                              capture_output=True, text=True, timeout=30)

    def demo(self, name="demo"):
        output = self.directory / name
        result = self.command("demo", "--out", output, "--at", "2026-10-24T00:00:00Z")
        self.assertEqual(result.returncode, 0, result.stderr)
        return output

    def validate_command(self, demo, cases, output):
        return self.command("validate", "--sources", "data/fixtures/v4/sources.jsonl",
                            "--ledgers", "data/fixtures/v4/ledgers.jsonl",
                            "--assignments", demo / "assignments.jsonl", "--pairs", demo / "pairs.jsonl",
                            "--cases", cases, "--out", output)

    def test_demo_manifest_and_independent_revalidation(self):
        demo = self.demo()
        manifest = validate(json.loads((demo / "manifest.json").read_text()), "manifest")
        self.assertEqual(manifest["actual_cases"], 2)
        self.assertEqual(manifest["cost_usd"], 0)
        self.assertEqual(manifest["training_base"]["model_id"], "Qwen/Qwen3.5-4B")
        for entry in manifest["input_files"] + manifest["output_files"]:
            self.assertEqual(hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest(), entry["sha256"])
        output = self.directory / "validated"
        result = self.validate_command(demo, demo / "cases.jsonl", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({r["status"] for r in read_jsonl(output / "validation.jsonl", "validation")}, {"needs_review"})

    def test_repeated_requests_are_identical_and_runs_are_not_overwritten(self):
        first, second = self.demo("first"), self.demo("second")
        self.assertEqual((first / "requests.jsonl").read_bytes(), (second / "requests.jsonl").read_bytes())
        before = (first / "manifest.json").read_bytes()
        result = self.command("demo", "--out", first)
        self.assertEqual(result.returncode, 1)
        self.assertIn("already exists", result.stderr)
        self.assertEqual((first / "manifest.json").read_bytes(), before)

    def test_prepare_writes_requests_without_generating_cases(self):
        demo = self.demo()
        output = self.directory / "prepared"
        result = self.command("prepare", "--sources", "data/fixtures/v4/sources.jsonl",
                              "--ledgers", "data/fixtures/v4/ledgers.jsonl",
                              "--assignments", demo / "assignments.jsonl", "--pairs", demo / "pairs.jsonl",
                              "--out", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(read_jsonl(output / "requests.jsonl", "request")), 2)
        self.assertFalse((output / "cases.jsonl").exists())
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual((manifest["expected_cases"], manifest["actual_cases"]), (2, 0))

    def test_rejections_return_two_and_have_structured_reasons(self):
        demo = self.demo()
        cases = read_jsonl(demo / "cases.jsonl", "case")
        cases[0]["target"]["resume_md"] += "Unmapped skill\n"
        edited = self.directory / "edited-cases.jsonl"
        write_jsonl(edited, cases, "case")
        output = self.directory / "rejected"
        result = self.validate_command(demo, edited, output)
        self.assertEqual(result.returncode, 2, result.stderr)
        reports = read_jsonl(output / "validation.jsonl", "validation")
        self.assertEqual(reports[0]["status"], "rejected")
        self.assertIn("evidence", {r["code"] for r in reports[0]["reasons"]})

    def test_legacy_fixture_input_fails_before_writing_output(self):
        output = self.directory / "legacy"
        result = self.command("split", "--sources", "data/pairs.csv", "--out", output)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(output.exists())

    def test_output_must_be_in_ignored_runs_directory(self):
        result = self.command("demo", "--out", ROOT / "data" / "unexpected-output")
        self.assertEqual(result.returncode, 1)
        self.assertIn("under runs/", result.stderr)
        self.assertFalse((ROOT / "data" / "unexpected-output").exists())


if __name__ == "__main__":
    unittest.main()
