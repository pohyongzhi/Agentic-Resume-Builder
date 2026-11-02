import csv
import shutil

import pytest

from common.data import dataset_hash, load_manifest, load_pairs
from scripts.validate_dataset import DatasetValidationError, validate_dataset


def test_frozen_dataset_has_expected_balanced_pairs():
    pairs = load_pairs(split="test")
    assert len(pairs) == 100
    assert len({p["resume_id"] for p in pairs}) == 100
    assert len({p["jd_id"] for p in pairs}) == 20
    assert {p["category"] for p in pairs} == {
        "software_engineering", "data_ai", "cybersecurity",
        "cloud_devops", "business_technology",
    }
    assert all(sum(p["jd_id"] == jd for p in pairs) == 5 for jd in {p["jd_id"] for p in pairs})


def test_category_filter_and_manifest_hash():
    pairs = load_pairs(split="test", categories=["cybersecurity"])
    manifest = load_manifest()
    assert len(pairs) == 20
    assert all(p["category"] == "cybersecurity" for p in pairs)
    assert dataset_hash() == manifest["dataset_hash"]


def test_checked_in_dataset_passes_validation():
    result = validate_dataset()
    assert result["counts"] == {
        "categories": 5,
        "job_descriptions": 20,
        "resumes": 100,
        "pairs": 100,
    }


def test_validation_rejects_missing_keyword_file(tmp_path):
    copied = tmp_path / "data"
    shutil.copytree("data", copied)
    (copied / "jds" / "se_jd_01.keywords.json").unlink()
    with pytest.raises(DatasetValidationError, match="missing keyword labels"):
        validate_dataset(copied)


def test_validation_detects_split_leakage(tmp_path):
    copied = tmp_path / "data"
    shutil.copytree("data", copied)
    pairs_path = copied / "pairs.csv"
    with pairs_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    rows[0]["split"] = "dev"  # its JD is still used by four test rows
    with pairs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(DatasetValidationError, match="data leakage"):
        validate_dataset(copied)
