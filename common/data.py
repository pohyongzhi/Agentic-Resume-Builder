"""Loads the test cases. A 'pair' = one candidate resume + one job description."""
import csv
import json
import os

import config


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_pairs():
    pairs = []
    with open(os.path.join(config.DATA_DIR, "pairs.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            jd_base = os.path.join(config.DATA_DIR, "jds", row["jd_id"])
            resume_path = os.path.join(config.DATA_DIR, "resumes", row["resume_id"] + ".md")
            pairs.append({
                "pair_id": f'{row["resume_id"]}__{row["jd_id"]}',
                "resume_path": resume_path,
                "resume": _read(resume_path),
                "jd": _read(jd_base + ".md"),
                "jd_url": row.get("jd_url", ""),
                # Hand-written must-have keywords for this JD (with synonyms). Used for keyword coverage.
                "keywords": json.load(open(jd_base + ".keywords.json", encoding="utf-8")),
            })
    return pairs
