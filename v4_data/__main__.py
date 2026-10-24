"""Run with python3 -m v4_data; all commands are offline."""

import argparse
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .contracts import ROOT, SCHEMA_PATH, DataError, canonical, digest, parse_json, read_jsonl, validate, write_jsonl
from .holdout import freeze_holdout
from .pipeline import prepare_requests, render_fixture
from .splitting import assign_splits, pair_sources
from .validation import validate_cases


def local_path(value):
    path = Path(value).resolve()
    if not path.is_relative_to(ROOT):
        raise DataError("All inputs and outputs must stay inside the repository")
    return path


def load_json(path):
    return parse_json(local_path(path).read_text(encoding="utf-8"))


def code_revision():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    hashes = {p.name: digest(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "v4_data").glob("*.py"))}
    return result.stdout.strip() + "+code:" + digest(hashes)


def describe_file(path, record_type, records):
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "records": records, "record_type": record_type}


def run(args):
    stamp = args.at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    validate(stamp, "timestamp")
    # Also reject impossible calendar dates rather than only validating shape.
    datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")
    policy_path = local_path(args.policy)
    policy = validate(load_json(policy_path), "policy")
    revision = code_revision()
    inputs = [describe_file(policy_path, "policy", 1)]
    for path, kind in ((SCHEMA_PATH, "schema"), (ROOT / "prompts/v4/teacher-v1.txt", "prompt"),
                       (ROOT / "configs/v4/legacy-exclusions.json", "exclusions")):
        inputs.append(describe_file(path, kind, 1))
    def read(path, kind):
        path = local_path(path)
        rows = read_jsonl(path, kind)
        inputs.append(describe_file(path, kind, len(rows)))
        return rows
    def json_input(path, kind):
        path = local_path(path)
        result = load_json(path)
        inputs.append(describe_file(path, kind, 1))
        return result

    output = local_path(args.out)
    if not output.is_relative_to(ROOT / "runs") or output == ROOT / "runs":
        raise DataError("Run outputs must use a fresh subdirectory under runs/")
    if output.exists():
        raise DataError("Output directory already exists; choose a new run directory")
    run_id = "run-" + digest([args.command, stamp, str(output.relative_to(ROOT))])[:24]
    products, expected, actual = [], 0, 0
    purpose = "offline-preparation"
    if args.command == "demo":
        purpose = "fixture-smoke"
        sources = read(ROOT / "data/fixtures/v4/sources.jsonl", "source")
        ledgers = read(ROOT / "data/fixtures/v4/ledgers.jsonl", "ledger")
        assignments = assign_splits(sources, policy, stamp, reservations={s["id"]: "train" for s in sources})
        pairs = [pair_sources(c, j, sources, assignments) for c, j in (
            ("candidate-harbour-v1", "jd-inventory-v1"), ("candidate-orbit-v1", "jd-support-v1"))]
        requests = prepare_requests(sources, ledgers, assignments, pairs, policy)
        print("Fixture plan: 2 extractive test cases, 0 model calls, $0 API cost.", flush=True)
        cases = [render_fixture(request, sources, run_id, stamp, revision) for request in requests]
        reports = validate_cases(cases, sources, ledgers, assignments, pairs, policy, stamp)
        products = [("assignments.jsonl", assignments, "assignment"), ("pairs.jsonl", pairs, "pair"),
                    ("requests.jsonl", requests, "request"), ("cases.jsonl", cases, "case"),
                    ("validation.jsonl", reports, "validation")]
        expected = actual = len(cases)
    else:
        sources = read(args.sources, "source")
        existing_pairs = read(args.existing_pairs, "pair") if args.existing_pairs else []
        if args.command == "split":
            previous = read(args.previous, "assignment") if args.previous else []
            reservations = json_input(args.reservations, "reservations") if args.reservations else {}
            if not isinstance(reservations, dict):
                raise DataError("Reservations must map source IDs to split names")
            assignments = assign_splits(sources, policy, stamp, existing_pairs, previous, reservations)
            products = [("assignments.jsonl", assignments, "assignment")]
        else:
            assignments = read(args.assignments, "assignment")
            if args.command == "freeze":
                protocol_path = local_path(args.protocol)
                inputs.append(describe_file(protocol_path, "evaluation-protocol", 1))
                holdout = freeze_holdout(sources, assignments, policy, stamp,
                                         hashlib.sha256(protocol_path.read_bytes()).hexdigest(), existing_pairs)
                products = [("holdout.jsonl", [holdout], "holdout")]
            else:
                ledgers = read(args.ledgers, "ledger")
                pairs = read(args.pairs, "pair")
                holdout = None
                if args.holdout:
                    snapshots = read(args.holdout, "holdout")
                    if len(snapshots) != 1:
                        raise DataError("Expected one frozen holdout snapshot")
                    holdout = snapshots[0]
                expected = len(pairs)
                if args.command == "prepare":
                    requests = prepare_requests(sources, ledgers, assignments, pairs, policy, existing_pairs, holdout)
                    products = [("requests.jsonl", requests, "request")]
                    # Requests are preparation artifacts, not generated cases.
                else:
                    cases = read(args.cases, "case")
                    reports = validate_cases(cases, sources, ledgers, assignments, pairs, policy, stamp, existing_pairs, holdout)
                    actual = len(cases)
                    products = [("validation.jsonl", reports, "validation")]
    if not sources:
        raise DataError("Source inventory is empty")
    # All validation occurs before any output is written. A failed filesystem
    # write leaves an incomplete directory without a manifest; it is never reused.
    output.mkdir(parents=True, exist_ok=False)
    outputs = []
    for name, rows, kind in products:
        path = output / name
        write_jsonl(path, rows, kind)
        outputs.append(describe_file(path, kind, len(rows)))
    manifest = validate({"schema_version": "1.0", "run_id": run_id, "created_at": stamp,
                         "purpose": purpose, "code_revision": revision, "config_sha256": digest(policy),
                         "schema_sha256": digest(SCHEMA_PATH.read_text(encoding="utf-8")),
                         "input_files": inputs, "output_files": outputs, "source_ids": sorted(s["id"] for s in sources),
                         "expected_cases": expected, "actual_cases": actual, "cost_usd": 0,
                         "training_base": {"model_id": "Qwen/Qwen3.5-4B", "revision": "UNPINNED-NOT-READY"}}, "manifest")
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        handle.write(canonical(manifest) + "\n")
    rejected = sum(row["status"] == "rejected" for _, rows, kind in products if kind == "validation" for row in rows)
    print(canonical({"run": str(output.relative_to(ROOT)), "expected_cases": expected,
                     "actual_cases": actual, "rejected": rejected, "accepted_for_training": 0, "cost_usd": 0}))
    return 2 if rejected else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "split", "freeze", "prepare", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--out", required=True, help="Fresh directory under runs/")
        command.add_argument("--at", help="UTC timestamp for reproducibility, e.g. 2026-10-24T00:00:00Z")
        command.add_argument("--policy", default=str(ROOT / "configs/v4/split-policy.json"))
        if name != "demo":
            command.add_argument("--sources", required=True)
            command.add_argument("--existing-pairs", help="Original graph edges used during source splitting")
        if name == "split":
            command.add_argument("--previous", help="Full frozen assignments from the previous corpus version")
            command.add_argument("--reservations", help="JSON object mapping source IDs to explicit splits")
        if name in {"freeze", "prepare", "validate"}:
            command.add_argument("--assignments", required=True)
        if name == "freeze":
            command.add_argument("--protocol", required=True, help="Evaluator-owned frozen evaluation protocol")
        if name in {"prepare", "validate"}:
            command.add_argument("--ledgers", required=True)
            command.add_argument("--pairs", required=True, help="New within-split pairs created after source assignment")
            command.add_argument("--holdout", help="Evaluator-owned fingerprint JSONL, required for non-fixtures")
        if name == "validate":
            command.add_argument("--cases", required=True)
    try:
        return run(parser.parse_args())
    except (DataError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"v4_data: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
