"""Tokenizer, template, label-mask, and model-index audits."""

import json
import statistics
from pathlib import Path

from .config import load_config, resolve_repo_path
from .data import IGNORE_INDEX, read_records, serialize_record, verify_training_inputs
from .modeling import expected_target_names


def load_tokenizer(config, cache_dir=None, local_files_only=False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    if tokenizer.eos_token != "<|im_end|>":
        raise RuntimeError(f"Expected Qwen EOS <|im_end|>, found {tokenizer.eos_token!r}")
    if tokenizer.pad_token != "<|endoftext|>":
        raise RuntimeError(f"Expected Qwen padding token <|endoftext|>, found {tokenizer.pad_token!r}")
    if not tokenizer.chat_template:
        raise RuntimeError("Pinned tokenizer has no chat template")
    tokenizer.padding_side = config["sequence"]["padding_side"]
    return tokenizer


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def audit_dataset(config, tokenizer, examples=3, print_examples=True):
    reports = {}
    rendered = []
    for split in ("train", "validation"):
        path = resolve_repo_path(config["data"][split])
        records = read_records(path)
        features = [serialize_record(tokenizer, row, config["sequence"]["max_length"])
                    for row in records]
        lengths = [len(row["input_ids"]) for row in features]
        supervised = [row["supervised_tokens"] for row in features]
        reports[split] = {
            "records": len(records),
            "minimum_tokens": min(lengths),
            "q1_tokens": _percentile(lengths, 0.25),
            "median_tokens": _percentile(lengths, 0.5),
            "q3_tokens": _percentile(lengths, 0.75),
            "maximum_tokens": max(lengths),
            "mean_tokens": round(statistics.mean(lengths), 2),
            "supervised_tokens": sum(supervised),
            "maximum_supervised_tokens": max(supervised),
            "over_budget": sum(length > config["sequence"]["max_length"] for length in lengths),
        }
        for index in range(min(examples, len(records))):
            rendered.append((split, index, features[index]))

    if print_examples:
        for split, index, feature in rendered:
            input_text = tokenizer.decode(feature["input_ids"], skip_special_tokens=False)
            supervised_ids = [token_id for token_id, label in zip(feature["input_ids"], feature["labels"])
                              if label != IGNORE_INDEX]
            supervised_text = tokenizer.decode(supervised_ids, skip_special_tokens=False)
            print(f"\n=== {split} example {index + 1}: serialized ===\n{input_text}")
            print(f"\n=== {split} example {index + 1}: supervised only ===\n{supervised_text}")
            print(json.dumps({
                "total_tokens": len(feature["input_ids"]),
                "prompt_tokens_masked": feature["prompt_length"],
                "assistant_and_eos_tokens_supervised": feature["supervised_tokens"],
                "final_token_id": feature["input_ids"][-1],
                "eos_token_id": tokenizer.eos_token_id,
            }, sort_keys=True))
    return reports


def audit_tokenizer(config_path, cache_dir=None, local_files_only=False, examples=3, report_path=None):
    config = load_config(config_path)
    manifest = verify_training_inputs(config, resolve_repo_path("."))
    tokenizer = load_tokenizer(config, cache_dir, local_files_only)
    report = {
        "model_id": config["model"]["id"],
        "revision": config["model"]["revision"],
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "enable_thinking": False,
        "max_length": config["sequence"]["max_length"],
        "training_manifest_created_at": manifest["created_at"],
        "splits": audit_dataset(config, tokenizer, examples, True),
    }
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n=== tokenizer audit summary ===")
    print(json.dumps(report, indent=2))
    return report


def audit_labels(config_path, cache_dir=None, local_files_only=False, examples=3):
    config = load_config(config_path)
    verify_training_inputs(config, resolve_repo_path("."))
    tokenizer = load_tokenizer(config, cache_dir, local_files_only)
    report = audit_dataset(config, tokenizer, examples, True)
    print("\nAll prompt tokens and future padding positions are masked with -100; "
          "assistant content and terminal EOS are supervised.")
    return report


def audit_weight_index(config_path, cache_dir=None, local_files_only=False, report_path=None):
    from huggingface_hub import hf_hub_download

    config = load_config(config_path)
    model = config["model"]
    index_path = hf_hub_download(
        repo_id=model["id"], filename=model["weight_index_file"], revision=model["revision"],
        cache_dir=cache_dir, local_files_only=local_files_only)
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    names, counts = expected_target_names(index, config["lora"])
    report = {
        "model_id": model["id"], "revision": model["revision"],
        "counts": counts | {"total": len(names)}, "target_modules": names,
    }
    if report_path:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report
