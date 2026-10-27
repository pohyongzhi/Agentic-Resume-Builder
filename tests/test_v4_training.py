import copy
import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from v4_training.config import ROOT, TrainingConfigError, load_config, require_safe_output_dir
from v4_training.data import (
    IGNORE_INDEX,
    SerializationError,
    _as_ids,
    pad_features,
    serialize_record,
    verify_training_inputs,
)
from v4_training.modeling import expected_target_names
from v4_training.trainer import (
    _comparison_rows,
    _fixed_smoke_rows,
    _prepare_output,
    _training_arguments,
)


class FakeTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    @staticmethod
    def _content_ids(content):
        return [1000 + ord(character) for character in content]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        self.last_enable_thinking = enable_thinking
        ids = [1]
        role_tokens = {"system": 10, "user": 20, "assistant": 30}
        for message in messages:
            ids.append(role_tokens[message["role"]])
            if message["role"] == "assistant":
                ids.extend([40, 41])
            ids.extend(self._content_ids(message["content"]))
            ids.append(self.eos_token_id)
        if add_generation_prompt:
            ids.extend([30, 40, 41])
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(value - 1000) for value in ids if value >= 1000)


def sample_record():
    return {
        "prompt": [
            {"role": "system", "content": "Follow the source."},
            {"role": "user", "content": "Tailor this resume."},
        ],
        "completion": [{"role": "assistant", "content": "## Summary\nSupported."}],
    }


class CompletionMaskTests(unittest.TestCase):
    def test_chat_template_batch_encoding_is_unwrapped(self):
        self.assertEqual(_as_ids({"input_ids": [1, 2, 3]}), [1, 2, 3])

    def test_only_assistant_content_and_ending_eos_are_supervised(self):
        tokenizer = FakeTokenizer()
        feature = serialize_record(tokenizer, sample_record(), max_length=4096)
        prompt_length = feature["prompt_length"]
        self.assertTrue(all(label == IGNORE_INDEX for label in feature["labels"][:prompt_length]))
        self.assertEqual(feature["labels"][prompt_length:], feature["input_ids"][prompt_length:])
        self.assertEqual(feature["labels"][-1], tokenizer.eos_token_id)
        self.assertFalse(tokenizer.last_enable_thinking)

    def test_padding_is_never_supervised(self):
        tokenizer = FakeTokenizer()
        first = serialize_record(tokenizer, sample_record())
        second_record = sample_record()
        second_record["completion"][0]["content"] += " More supported detail."
        second = serialize_record(tokenizer, second_record)
        padded = pad_features([first, second], tokenizer.pad_token_id, pad_to_multiple_of=8)
        for attention, labels in zip(padded["attention_mask"], padded["labels"]):
            for mask, label in zip(attention, labels):
                if mask == 0:
                    self.assertEqual(label, IGNORE_INDEX)

    def test_overlength_is_rejected_instead_of_truncated(self):
        tokenizer = FakeTokenizer()
        with self.assertRaisesRegex(SerializationError, "truncation is forbidden"):
            serialize_record(tokenizer, sample_record(), max_length=8)


class TrainingInputTests(unittest.TestCase):
    def test_manifest_hashes_counts_and_system_prompt_are_verified(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            root = Path(directory)
            system = root / "system.txt"
            system.write_text("Frozen prompt\n", encoding="utf-8")
            row = sample_record()
            row["prompt"][0]["content"] = "Frozen prompt"
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            train.write_text("".join(json.dumps(row) + "\n" for _ in range(450)), encoding="utf-8")
            validation.write_text("".join(json.dumps(row) + "\n" for _ in range(50)), encoding="utf-8")
            manifest = {
                "train": {"records": 450, "sha256": hashlib.sha256(train.read_bytes()).hexdigest()},
                "validation": {"records": 50, "sha256": hashlib.sha256(validation.read_bytes()).hexdigest()},
                "system_prompt": {"sha256": hashlib.sha256(system.read_bytes()).hexdigest()},
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            config = {"data": {
                "train": "train.jsonl", "validation": "validation.jsonl",
                "system_prompt": "system.txt", "manifest": "manifest.json",
            }}
            verify_training_inputs(config, root)
            train.write_text(train.read_text(encoding="utf-8") + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(SerializationError, "SHA-256"):
                verify_training_inputs(config, root)

    def test_model_revision_and_output_location_are_guarded(self):
        config = load_config()
        self.assertEqual(config["model"]["revision"], "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
        self.assertEqual(config["sequence"], {
            "max_length": 4096,
            "padding_side": "right",
            "packing": False,
            "enable_thinking": False,
            "completion_only_loss": True,
        })
        self.assertEqual(config["lora"]["rank"], 16)
        self.assertEqual(config["lora"]["alpha"], 32)
        self.assertEqual(config["lora"]["dropout"], 0.05)
        self.assertEqual(config["training"], {
            "learning_rate": 0.0001,
            "optimizer": "adamw_torch",
            "weight_decay": 0.01,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "gradient_checkpointing": True,
            "bf16": True,
            "seed": 42,
            "packing": False,
            "use_cache": False,
        })
        altered = copy.deepcopy(config)
        altered["model"]["revision"] = "main"
        with tempfile.NamedTemporaryFile("w", suffix=".json", dir=ROOT / "runs") as handle:
            json.dump(altered, handle)
            handle.flush()
            with self.assertRaises(TrainingConfigError):
                load_config(handle.name)
        with self.assertRaises(TrainingConfigError):
            require_safe_output_dir(ROOT / "unexpected-training-output")

    def test_smoke_selection_is_fixed(self):
        rows = [{"value": index} for index in range(30)]
        first = _fixed_smoke_rows(rows, 12, "fixed-seed")
        self.assertEqual(first, _fixed_smoke_rows(list(reversed(rows)), 12, "fixed-seed"))

    def test_adapter_comparison_selection_is_fixed_and_bounded(self):
        rows = [{"value": index} for index in range(30)]
        first = _comparison_rows(rows, 8)
        self.assertEqual(first, _comparison_rows(list(reversed(rows)), 8))
        with self.assertRaisesRegex(RuntimeError, "2 to 20"):
            _comparison_rows(rows, 1)
        with self.assertRaisesRegex(RuntimeError, "from 5"):
            _comparison_rows(rows[:5], 8)

    def test_training_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            output = Path(directory) / "new-run"
            self.assertEqual(_prepare_output(output), output)
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                _prepare_output(output)

    def test_transformers_five_receives_fractional_warmup_steps(self):
        class FakeTrainingArguments:
            def __init__(self, **kwargs):
                self.values = kwargs

        fake_transformers = types.SimpleNamespace(TrainingArguments=FakeTrainingArguments)
        with patch.dict("sys.modules", {"transformers": fake_transformers}):
            arguments = _training_arguments(load_config(), ROOT / "runs" / "fake", smoke=True)
        self.assertEqual(arguments.values["warmup_steps"], 0.05)
        self.assertNotIn("warmup_ratio", arguments.values)


class TargetDiscoveryTests(unittest.TestCase):
    def test_exact_language_backbone_targets_resolve_from_index(self):
        config = load_config()
        weight_map = {}
        for layer in range(32):
            family = "self_attn" if layer % 4 == 3 else "linear_attn"
            for suffix in config["lora"]["target_suffixes"][family]:
                weight_map[f"model.language_model.layers.{layer}.{family}.{suffix}.weight"] = "part"
            for suffix in config["lora"]["target_suffixes"]["mlp"]:
                weight_map[f"model.language_model.layers.{layer}.mlp.{suffix}.weight"] = "part"
        weight_map["model.visual.blocks.0.attn.qkv.weight"] = "part"
        weight_map["lm_head.weight"] = "part"
        names, counts = expected_target_names({"weight_map": weight_map}, config["lora"])
        self.assertEqual(len(names), 248)
        self.assertEqual(counts, {"linear_attn": 120, "mlp": 96, "self_attn": 32})
        self.assertFalse(any("visual" in name or "lm_head" in name for name in names))

    def test_missing_projection_fails_closed(self):
        config = load_config()
        with self.assertRaisesRegex(TrainingConfigError, "target count mismatch"):
            expected_target_names({"weight_map": {}}, config["lora"])


if __name__ == "__main__":
    unittest.main()
