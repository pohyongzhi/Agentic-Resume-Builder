"""Validate the frozen evaluation dataset before running workflows.

Usage:
    python scripts/validate_dataset.py
"""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REQUIRED_COLUMNS = {
    "pair_id", "resume_id", "jd_id", "category", "role_family",
    "seniority", "fit_level", "split", "source", "jd_url",
}
VALID_FITS = {"strong", "moderate", "weak"}
VALID_SPLITS = {"dev", "validation", "test"}
VALID_SENIORITY = {"entry", "mid", "senior"}
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class DatasetValidationError(ValueError):
    pass


def _dataset_hash(paths, data_dir):
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p.relative_to(data_dir))):
        digest.update(str(path.relative_to(data_dir)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_dataset(data_dir=DATA):
    data_dir = Path(data_dir)
    errors = []
    pairs_path = data_dir / "pairs.csv"
    manifest_path = data_dir / "dataset.json"
    if not pairs_path.exists() or not manifest_path.exists():
        missing = [str(p) for p in (pairs_path, manifest_path) if not p.exists()]
        raise DatasetValidationError("Missing required dataset files: " + ", ".join(missing))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with pairs_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        if not REQUIRED_COLUMNS.issubset(columns):
            errors.append(f"pairs.csv missing columns: {sorted(REQUIRED_COLUMNS - columns)}")
        rows = list(reader)

    pair_ids, pair_keys = set(), set()
    resume_splits, jd_splits = defaultdict(set), defaultdict(set)
    category_counts, fit_by_jd = Counter(), defaultdict(Counter)
    resume_ids, jd_ids, hash_paths = set(), set(), [pairs_path]

    for line, row in enumerate(rows, 2):
        pair_id = row.get("pair_id", "")
        key = (row.get("resume_id", ""), row.get("jd_id", ""))
        for label, value in (
            ("pair_id", pair_id),
            ("resume_id", row.get("resume_id", "")),
            ("jd_id", row.get("jd_id", "")),
        ):
            if not SAFE_ID.fullmatch(value):
                errors.append(f"line {line}: unsafe {label} '{value}'")
        if not pair_id or pair_id in pair_ids:
            errors.append(f"line {line}: duplicate or empty pair_id '{pair_id}'")
        if key in pair_keys:
            errors.append(f"line {line}: duplicate resume/JD pairing {key}")
        pair_ids.add(pair_id)
        pair_keys.add(key)

        if row.get("fit_level") not in VALID_FITS:
            errors.append(f"line {line}: invalid fit_level '{row.get('fit_level')}'")
        if row.get("split") not in VALID_SPLITS:
            errors.append(f"line {line}: invalid split '{row.get('split')}'")
        if row.get("seniority") not in VALID_SENIORITY:
            errors.append(f"line {line}: invalid seniority '{row.get('seniority')}'")

        resume_id, jd_id = row.get("resume_id", ""), row.get("jd_id", "")
        resume_splits[resume_id].add(row.get("split"))
        jd_splits[jd_id].add(row.get("split"))
        category_counts[row.get("category", "")] += 1
        fit_by_jd[jd_id][row.get("fit_level", "")] += 1
        resume_ids.add(resume_id)
        jd_ids.add(jd_id)

    for item_id, splits in {**resume_splits, **jd_splits}.items():
        if len(splits) > 1:
            errors.append(f"data leakage: '{item_id}' appears in splits {sorted(splits)}")

    for resume_id in sorted(resume_ids):
        path = data_dir / "resumes" / f"{resume_id}.md"
        if not path.exists():
            errors.append(f"missing resume: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        if len(text.strip()) < 500:
            errors.append(f"resume too short: {resume_id}")
        if not re.search(r"[\w.+-]+@[\w.-]+\.test\b", text):
            errors.append(f"resume lacks a fictional .test email: {resume_id}")
        hash_paths.append(path)

    for jd_id in sorted(jd_ids):
        jd_path = data_dir / "jds" / f"{jd_id}.md"
        kw_path = data_dir / "jds" / f"{jd_id}.keywords.json"
        if not jd_path.exists():
            errors.append(f"missing JD: {jd_path}")
        else:
            if len(jd_path.read_text(encoding="utf-8").strip()) < 800:
                errors.append(f"JD too short: {jd_id}")
            hash_paths.append(jd_path)
        if not kw_path.exists():
            errors.append(f"missing keyword labels: {kw_path}")
        else:
            try:
                keywords = json.loads(kw_path.read_text(encoding="utf-8"))
                if not (8 <= len(keywords) <= 12):
                    errors.append(f"{jd_id}: expected 8-12 keywords, got {len(keywords)}")
                if any(not k.get("keyword") or not isinstance(k.get("synonyms", []), list)
                       for k in keywords):
                    errors.append(f"{jd_id}: invalid keyword schema")
            except (json.JSONDecodeError, AttributeError) as exc:
                errors.append(f"{jd_id}: invalid keyword JSON ({exc})")
            hash_paths.append(kw_path)

    expected_fit = Counter({"strong": 2, "moderate": 2, "weak": 1})
    for jd_id, counts in fit_by_jd.items():
        if counts != expected_fit:
            errors.append(f"{jd_id}: fit distribution is {dict(counts)}, expected {dict(expected_fit)}")

    counts = manifest.get("counts", {})
    actual = {
        "categories": len(category_counts),
        "job_descriptions": len(jd_ids),
        "resumes": len(resume_ids),
        "pairs": len(rows),
    }
    for key, value in actual.items():
        if counts.get(key) != value:
            errors.append(f"manifest count {key}={counts.get(key)}, actual={value}")
    if manifest.get("category_pairs") != dict(category_counts):
        errors.append("manifest category distribution does not match pairs.csv")

    calculated_hash = _dataset_hash(hash_paths, data_dir)
    if manifest.get("dataset_hash") != calculated_hash:
        errors.append(
            f"dataset hash mismatch: manifest={manifest.get('dataset_hash')} actual={calculated_hash}"
        )

    if errors:
        raise DatasetValidationError("\n".join(f"- {error}" for error in errors))
    return {
        "dataset_hash": calculated_hash,
        "counts": actual,
        "category_pairs": dict(category_counts),
    }


if __name__ == "__main__":
    try:
        result = validate_dataset()
    except DatasetValidationError as exc:
        raise SystemExit(f"Dataset validation failed:\n{exc}") from exc
    print(json.dumps(result, indent=2))
