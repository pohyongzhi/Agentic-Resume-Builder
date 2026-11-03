"""Chat serialization and completion-only labels."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


IGNORE_INDEX = -100


class SerializationError(ValueError):
    """Raised when a record cannot be serialized without weakening masking rules."""


def read_records(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SerializationError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if sorted(row) != ["completion", "prompt"]:
                raise SerializationError(f"{path}:{number}: expected only prompt and completion")
            if len(row["prompt"]) != 2 or [m.get("role") for m in row["prompt"]] != ["system", "user"]:
                raise SerializationError(f"{path}:{number}: expected system and user prompt messages")
            if len(row["completion"]) != 1 or row["completion"][0].get("role") != "assistant":
                raise SerializationError(f"{path}:{number}: expected one assistant completion")
            if any(not m.get("content") for m in row["prompt"] + row["completion"]):
                raise SerializationError(f"{path}:{number}: empty message content")
            rows.append(row)
    return rows


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_training_inputs(config, root):
    """Verify the frozen data, prompt, counts, and hashes before serialization."""
    manifest_path = root / config["data"]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_records = config["data"].get(
        "expected_records", {"train": 450, "validation": 50})
    if (set(expected_records) != {"train", "validation"}
            or any(not isinstance(value, int) or value <= 0
                   for value in expected_records.values())):
        raise SerializationError(
            "data.expected_records must declare positive train and validation counts")
    checks = {
        "train": (root / config["data"]["train"], expected_records["train"]),
        "validation": (
            root / config["data"]["validation"], expected_records["validation"]),
        "system_prompt": (root / config["data"]["system_prompt"], None),
    }
    for key, (path, required_count) in checks.items():
        expected = manifest.get(key, {})
        if expected.get("sha256") != sha256_file(path):
            raise SerializationError(f"{key} SHA-256 does not match the training manifest")
        if required_count is not None:
            rows = read_records(path)
            if expected.get("records") != required_count or len(rows) != required_count:
                raise SerializationError(
                    f"{key} must contain exactly {required_count} records")
    system_prompt = (root / config["data"]["system_prompt"]).read_text(encoding="utf-8").strip()
    for split in ("train", "validation"):
        for index, row in enumerate(read_records(root / config["data"][split]), 1):
            if row["prompt"][0]["content"].strip() != system_prompt:
                raise SerializationError(
                    f"{split} record {index} does not contain the frozen system prompt")
    return manifest


def _as_ids(value):
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise SerializationError("Tokenized chat template did not return input_ids")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise SerializationError("Expected a single serialized conversation")
        value = value[0]
    return list(value)


def apply_chat_template(tokenizer, messages, *, add_generation_prompt):
    """Apply the pinned tokenizer template with thinking disabled when supported."""
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
    }
    try:
        return _as_ids(tokenizer.apply_chat_template(messages, **kwargs))
    except TypeError as exc:
        if "enable_thinking" not in str(exc):
            raise
        kwargs.pop("enable_thinking")
        return _as_ids(tokenizer.apply_chat_template(messages, **kwargs))


def _finish_at_eos(tokenizer, input_ids):
    eos = tokenizer.eos_token_id
    if eos is None:
        raise SerializationError("Tokenizer has no EOS token ID")
    positions = [index for index, token_id in enumerate(input_ids) if token_id == eos]
    if positions:
        last = positions[-1]
        trailing = input_ids[last + 1:]
        if trailing:
            decoded = tokenizer.decode(trailing, skip_special_tokens=False)
            if decoded.strip():
                input_ids.append(eos)
            else:
                del input_ids[last + 1:]
    elif not input_ids or input_ids[-1] != eos:
        input_ids.append(eos)
    if input_ids[-1] != eos:
        raise SerializationError("Serialized completion does not end in EOS")


def serialize_record(tokenizer, record, max_length=4096):
    prompt_ids = apply_chat_template(tokenizer, record["prompt"], add_generation_prompt=True)
    messages = record["prompt"] + record["completion"]
    input_ids = apply_chat_template(tokenizer, messages, add_generation_prompt=False)
    _finish_at_eos(tokenizer, input_ids)
    if input_ids[:len(prompt_ids)] != prompt_ids:
        raise SerializationError(
            "Official full-conversation template does not share the generation-prompt prefix")
    if len(input_ids) > max_length:
        raise SerializationError(
            f"Serialized record has {len(input_ids)} tokens, exceeding {max_length}; truncation is forbidden")
    labels = [IGNORE_INDEX] * len(prompt_ids) + input_ids[len(prompt_ids):]
    if len(labels) != len(input_ids) or not any(label != IGNORE_INDEX for label in labels):
        raise SerializationError("Completion labels are empty or misaligned")
    if any(label != IGNORE_INDEX for label in labels[:len(prompt_ids)]):
        raise SerializationError("Prompt token became supervised")
    if labels[-1] != tokenizer.eos_token_id:
        raise SerializationError("Ending EOS must be supervised")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "prompt_length": len(prompt_ids),
        "supervised_tokens": sum(label != IGNORE_INDEX for label in labels),
    }


def pad_features(features, pad_token_id, pad_to_multiple_of=None):
    if pad_token_id is None:
        raise SerializationError("Tokenizer has no padding token ID")
    width = max(len(row["input_ids"]) for row in features)
    if pad_to_multiple_of:
        width = ((width + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
    output = {"input_ids": [], "attention_mask": [], "labels": []}
    for row in features:
        padding = width - len(row["input_ids"])
        output["input_ids"].append(row["input_ids"] + [pad_token_id] * padding)
        output["attention_mask"].append(row["attention_mask"] + [0] * padding)
        output["labels"].append(row["labels"] + [IGNORE_INDEX] * padding)
    return output


@dataclass
class CompletionOnlyCollator:
    tokenizer: object
    pad_to_multiple_of: int = 8
    last_supervised_tokens: int = 0

    def __call__(self, features):
        import torch

        padded = pad_features(features, self.tokenizer.pad_token_id, self.pad_to_multiple_of)
        self.last_supervised_tokens = sum(
            label != IGNORE_INDEX for row in padded["labels"] for label in row)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in padded.items()}


class TokenizedDataset:
    def __init__(self, records, tokenizer, max_length):
        self.features = [serialize_record(tokenizer, row, max_length) for row in records]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return {key: value for key, value in self.features[index].items()
                if key in {"input_ids", "attention_mask", "labels"}}

    @property
    def supervised_tokens(self):
        return sum(row["supervised_tokens"] for row in self.features)
