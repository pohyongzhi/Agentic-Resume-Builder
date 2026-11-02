"""Meaning-preserving input perturbations and robustness aggregation."""
import csv
import hashlib
import json
import random
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

VARIANTS = ("original", "formatting", "reordered_sections")
ROBUSTNESS_METRICS = (
    "faithfulness", "judge_overall_mean", "kw_after", "sim_after",
)


def _formatting(text):
    lines = []
    for line in text.splitlines():
        if line.startswith("- "):
            line = "* " + line[2:]
        elif line.startswith("## "):
            line = "## " + line[3:].upper()
        lines.append(line)
    return "\n\n".join(lines) + "\n"


def _reorder_sections(text):
    matches = list(re.finditer(r"(?m)^##\s+.+$", text))
    if len(matches) < 2:
        return text
    preamble = text[:matches[0].start()].rstrip()
    sections = []
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append(text[match.start():stop].strip())
    return preamble + "\n\n" + "\n\n".join(reversed(sections)) + "\n"


def perturb_pair(pair, variant):
    if variant not in VARIANTS:
        raise ValueError(f"unknown robustness variant: {variant}")
    if variant == "original":
        return pair["resume"], pair["jd"]
    transform = _formatting if variant == "formatting" else _reorder_sections
    resume, jd = transform(pair["resume"]), transform(pair["jd"])
    if not transformation_preserves_content(pair["resume"], resume):
        raise ValueError(f"{variant} changed resume factual content")
    if not transformation_preserves_content(pair["jd"], jd):
        raise ValueError(f"{variant} changed JD factual content")
    return resume, jd


def _content_tokens(text):
    return Counter(re.findall(r"[a-z0-9]+", text.lower()))


def transformation_preserves_content(before, after):
    """Formatting/order variants must preserve the same case-insensitive lexical content."""
    return _content_tokens(before) == _content_tokens(after)


def select_stratified_pairs(pairs, seed=42):
    """Select four per category and globally 8 strong, 8 moderate and 4 weak pairs."""
    by_category = defaultdict(lambda: defaultdict(list))
    for pair in pairs:
        by_category[pair["category"]][pair["fit_level"]].append(pair)
    categories = sorted(by_category)
    if len(categories) != 5:
        raise ValueError(f"expected five categories, got {categories}")
    allocations = (
        {"strong": 2, "moderate": 1, "weak": 1},
        {"strong": 2, "moderate": 1, "weak": 1},
        {"strong": 1, "moderate": 2, "weak": 1},
        {"strong": 1, "moderate": 2, "weak": 1},
        {"strong": 2, "moderate": 2, "weak": 0},
    )
    rng = random.Random(seed)
    selected = []
    for category, allocation in zip(categories, allocations):
        for fit_level, count in allocation.items():
            candidates = sorted(
                by_category[category][fit_level],
                key=lambda pair: (pair.get("role_family", ""), pair["pair_id"]),
            )
            if len(candidates) < count:
                raise ValueError(f"not enough {category}/{fit_level} pairs")
            chosen = rng.sample(candidates, count)
            selected.extend(sorted(chosen, key=lambda pair: pair["pair_id"]))
    return selected


def robustness_manifest(pairs, dataset_hash, seed=42):
    selected = select_stratified_pairs(pairs, seed)
    payload = {
        "version": "robustness-sample-v1",
        "dataset_hash": dataset_hash,
        "seed": seed,
        "pair_ids": [pair["pair_id"] for pair in selected],
        "strata": [
            {
                "pair_id": pair["pair_id"],
                "category": pair["category"],
                "fit_level": pair["fit_level"],
                "role_family": pair.get("role_family"),
                "seniority": pair.get("seniority"),
            }
            for pair in selected
        ],
    }
    payload["manifest_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    return payload


def summarise_robustness(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["workflow"], row["base_pair_id"])].append(row)

    by_workflow = defaultdict(list)
    for (workflow, _), group in grouped.items():
        by_workflow[workflow].append(group)

    summary = []
    for workflow, pair_groups in sorted(by_workflow.items()):
        success_values = []
        metric_deltas = defaultdict(list)
        for group in pair_groups:
            original = next((row for row in group if row["variant"] == "original"), None)
            success_values.extend(1 if not row.get("error") else 0 for row in group)
            if original is None:
                continue
            for row in group:
                if row["variant"] == "original":
                    continue
                for metric in ROBUSTNESS_METRICS:
                    if original.get(metric) is not None and row.get(metric) is not None:
                        metric_deltas[metric].append(row[metric] - original[metric])

        record = {
            "workflow": workflow,
            "pairs": len(pair_groups),
            "variant_runs": sum(len(group) for group in pair_groups),
            "robust_success_rate": round(sum(success_values) / len(success_values), 4)
            if success_values else None,
        }
        for metric in ROBUSTNESS_METRICS:
            deltas = metric_deltas[metric]
            record[f"{metric}_paired_n"] = len(deltas)
            record[f"{metric}_mean_delta"] = round(st.mean(deltas), 4) if deltas else None
            record[f"{metric}_mean_abs_change"] = (
                round(st.mean(abs(value) for value in deltas), 4) if deltas else None
            )
            record[f"{metric}_worst_degradation"] = (
                round(min(deltas), 4) if deltas else None
            )
        summary.append(record)
    return summary


def write_robustness_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
