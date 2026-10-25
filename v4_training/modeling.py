"""Strict Qwen3.5 language-backbone LoRA target discovery."""

import json
import re
from collections import Counter

from .config import TrainingConfigError


WEIGHT_PATTERN = re.compile(
    r"^model\.language_model\.layers\.(\d+)\."
    r"(self_attn|linear_attn|mlp)\.([^.]+)\.weight$"
)


def expected_target_names(weight_index, lora_config):
    names = []
    counts = Counter()
    suffixes = {family: set(values) for family, values in lora_config["target_suffixes"].items()}
    for name in weight_index.get("weight_map", {}):
        match = WEIGHT_PATTERN.fullmatch(name)
        if not match:
            continue
        family, suffix = match.group(2), match.group(3)
        if suffix in suffixes.get(family, set()):
            names.append(name.removesuffix(".weight"))
            counts[family] += 1
    _check_counts(counts, len(names), lora_config)
    return sorted(names), dict(counts)


def _check_counts(counts, total, lora_config):
    expected = lora_config["expected_counts"]
    actual = {family: counts.get(family, 0) for family in ("self_attn", "linear_attn", "mlp")}
    if actual != {family: expected[family] for family in actual} or total != expected["total"]:
        raise TrainingConfigError(
            f"LoRA target count mismatch: expected {expected}, resolved {actual | {'total': total}}")


def read_weight_index(path):
    return json.loads(open(path, encoding="utf-8").read())


def resolve_model_targets(model, lora_config):
    import torch

    prefix = "model.language_model.layers."
    suffixes = {family: set(values) for family, values in lora_config["target_suffixes"].items()}
    forbidden = tuple(lora_config["forbidden_name_fragments"])
    selected = []
    counts = Counter()
    for name, module in model.named_modules():
        if not name.startswith(prefix) or not isinstance(module, torch.nn.Linear):
            continue
        parts = name.split(".")
        if len(parts) < 6:
            continue
        family, suffix = parts[-2], parts[-1]
        if suffix in suffixes.get(family, set()):
            if any(fragment in name for fragment in forbidden):
                raise TrainingConfigError(f"Forbidden LoRA target resolved: {name}")
            selected.append(name)
            counts[family] += 1
    _check_counts(counts, len(selected), lora_config)
    return sorted(selected), dict(counts)
