"""Load the frozen evaluation dataset. A pair is one resume matched to one JD."""
import csv
import json
from pathlib import Path
import re

import config

SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _read(path):
    return Path(path).read_text(encoding="utf-8")


def load_manifest(data_dir=None):
    data_dir = Path(data_dir or config.DATA_DIR)
    path = data_dir / "dataset.json"
    if not path.exists():
        return {"name": "unversioned", "version": "0", "dataset_hash": "unversioned"}
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_hash(data_dir=None):
    """The frozen content hash used to separate cached experiment results."""
    return load_manifest(data_dir).get("dataset_hash", "unversioned")


def load_pairs(split=None, categories=None, data_dir=None):
    """Load pairs with optional split/category filters while preserving workflow input keys."""
    data_dir = Path(data_dir or config.DATA_DIR)
    categories = set(categories or [])
    pairs = []
    seen = set()
    with (data_dir / "pairs.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if split and row.get("split") != split:
                continue
            if categories and row.get("category") not in categories:
                continue
            pair_id = row.get("pair_id") or f'{row["resume_id"]}__{row["jd_id"]}'
            for label, value in (
                ("pair_id", pair_id),
                ("resume_id", row["resume_id"]),
                ("jd_id", row["jd_id"]),
            ):
                if not SAFE_ID.fullmatch(value):
                    raise ValueError(f"unsafe {label} in pairs.csv: {value!r}")
            if pair_id in seen:
                raise ValueError(f"duplicate pair_id in pairs.csv: {pair_id}")
            seen.add(pair_id)
            jd_base = data_dir / "jds" / row["jd_id"]
            resume_path = data_dir / "resumes" / f'{row["resume_id"]}.md'
            keywords_path = jd_base.with_suffix(".keywords.json")
            missing = [str(path) for path in (resume_path, jd_base.with_suffix(".md"), keywords_path)
                       if not path.exists()]
            if missing:
                raise FileNotFoundError(f"pair '{pair_id}' references missing files: {missing}")
            pairs.append({
                "pair_id": pair_id,
                "resume_id": row["resume_id"],
                "jd_id": row["jd_id"],
                "resume_path": str(resume_path),
                "resume": _read(resume_path),
                "jd": _read(jd_base.with_suffix(".md")),
                "jd_url": row.get("jd_url", ""),
                # Hand-written must-have keywords for this JD (with synonyms). Used for keyword coverage.
                "keywords": json.loads(keywords_path.read_text(encoding="utf-8")),
                "category": row.get("category", ""),
                "role_family": row.get("role_family", ""),
                "seniority": row.get("seniority", ""),
                "fit_level": row.get("fit_level", ""),
                "split": row.get("split", ""),
                "source": row.get("source", ""),
            })
    return pairs
