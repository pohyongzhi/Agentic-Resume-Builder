"""Single-GPU BF16 LoRA training and smoke verification."""

import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

from .audit import load_tokenizer
from .config import load_config, require_safe_output_dir, resolve_repo_path
from .data import (
    CompletionOnlyCollator,
    TokenizedDataset,
    apply_chat_template,
    read_records,
    verify_training_inputs,
)
from .modeling import resolve_model_targets


TRACKED_PACKAGES = (
    "torch", "transformers", "peft", "trl", "accelerate", "datasets",
    "safetensors", "sentencepiece", "tokenizers", "huggingface-hub", "jinja2",
)


def _versions():
    return {name: version(name) for name in TRACKED_PACKAGES}


def _prepare_output(value):
    path = require_safe_output_dir(value)
    if path.exists():
        raise RuntimeError(f"Output directory already exists: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _disable_cache(model):
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "use_cache"):
        model.config.text_config.use_cache = False


def _load_base(config, cache_dir):
    import torch
    from transformers import Qwen3_5ForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3.5-4B training and adapter evaluation require one CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible CUDA GPU, found {torch.cuda.device_count()}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The allocated GPU does not report BF16 support")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    _disable_cache(model)
    return model


def _prepare_lora(config, cache_dir):
    from peft import LoraConfig, TaskType, get_peft_model

    model = _load_base(config, cache_dir)
    targets, counts = resolve_model_targets(model, config["lora"])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = config["lora"]
    peft_config = LoraConfig(
        r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        bias=lora["bias"], task_type=TaskType.CAUSAL_LM, target_modules=targets,
    )
    model = get_peft_model(model, peft_config)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any(any(bad in name for bad in lora["forbidden_name_fragments"])
                            for name in trainable):
        raise RuntimeError("Trainable parameters are empty or escaped the language-backbone LoRA scope")
    return model, targets, counts, trainable


def _telemetry_callback(collator, output_dir):
    from transformers import TrainerCallback

    class StepTelemetryCallback(TrainerCallback):
        def __init__(self):
            self.started = None
            self.path = output_dir / "telemetry.jsonl"

        def on_step_begin(self, args, state, control, **kwargs):
            import torch
            self.started = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        def on_log(self, args, state, control, logs=None, **kwargs):
            import torch
            if logs is None:
                return
            event = dict(logs)
            event["step"] = state.global_step
            if self.started is not None:
                event["step_time_seconds"] = round(time.perf_counter() - self.started, 4)
            event["supervised_tokens_last_microbatch"] = collator.last_supervised_tokens
            if torch.cuda.is_available():
                event["gpu_peak_memory_gib"] = round(
                    torch.cuda.max_memory_allocated() / 2**30, 3)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")

    return StepTelemetryCallback()


def _training_arguments(config, output_dir, *, smoke=False, max_steps=None):
    from transformers import TrainingArguments

    train = config["training"]
    kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=train["per_device_train_batch_size"],
        per_device_eval_batch_size=train["per_device_eval_batch_size"],
        gradient_accumulation_steps=(config["smoke"]["gradient_accumulation_steps"]
                                     if smoke else train["gradient_accumulation_steps"]),
        learning_rate=train["learning_rate"],
        weight_decay=train["weight_decay"],
        optim=train["optimizer"],
        lr_scheduler_type=train["lr_scheduler_type"],
        warmup_ratio=train["warmup_ratio"],
        max_grad_norm=1.0,
        bf16=train["bf16"],
        fp16=False,
        gradient_checkpointing=train["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_strategy="steps",
        logging_steps=1,
        eval_strategy="steps" if smoke else "epoch",
        save_strategy="steps" if smoke else "epoch",
        save_total_limit=3,
        seed=train["seed"],
        data_seed=train["seed"],
        report_to="none",
        remove_unused_columns=False,
        prediction_loss_only=True,
        dataloader_num_workers=0,
    )
    if smoke:
        kwargs["max_steps"] = max_steps or config["smoke"]["max_steps"]
        kwargs["num_train_epochs"] = 1
        kwargs["eval_steps"] = 20
        kwargs["save_steps"] = 50
    else:
        kwargs["num_train_epochs"] = train["num_train_epochs"]
    return TrainingArguments(**kwargs)


def _trainer(config, cache_dir, output_dir, train_records, validation_records, *, smoke=False,
             max_steps=None):
    from transformers import Trainer

    tokenizer = load_tokenizer(config, cache_dir)
    maximum = config["sequence"]["max_length"]
    train_data = TokenizedDataset(train_records, tokenizer, maximum)
    validation_data = TokenizedDataset(validation_records, tokenizer, maximum)
    collator = CompletionOnlyCollator(tokenizer)
    model, targets, counts, trainable = _prepare_lora(config, cache_dir)
    args = _training_arguments(config, output_dir, smoke=smoke, max_steps=max_steps)
    trainer = Trainer(
        model=model, args=args, train_dataset=train_data, eval_dataset=validation_data,
        data_collator=collator, callbacks=[_telemetry_callback(collator, output_dir)],
    )
    metadata = {
        "model_id": config["model"]["id"], "revision": config["model"]["revision"],
        "packages": _versions(), "lora_target_counts": counts,
        "lora_target_modules": targets, "trainable_parameter_names": trainable,
        "train_records": len(train_data), "validation_records": len(validation_data),
        "train_supervised_tokens": train_data.supervised_tokens,
        "validation_supervised_tokens": validation_data.supervised_tokens,
    }
    (output_dir / "run-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return trainer, tokenizer


def _load_rows(config):
    verify_training_inputs(config, resolve_repo_path("."))
    return (read_records(resolve_repo_path(config["data"]["train"])),
            read_records(resolve_repo_path(config["data"]["validation"])))


def train_full(config_path, cache_dir, output_dir):
    config = load_config(config_path)
    train_rows, validation_rows = _load_rows(config)
    output = _prepare_output(output_dir)
    trainer, _ = _trainer(config, cache_dir, output, train_rows, validation_rows)
    trainer.train()
    trainer.save_model(str(output / "adapter-final"))
    metrics = trainer.evaluate()
    (output / "validation-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def _fixed_smoke_rows(rows, count, seed):
    ranked = sorted(rows, key=lambda row: hashlib.sha256(
        (seed + json.dumps(row, sort_keys=True, separators=(",", ":"))).encode()).hexdigest())
    return ranked[:count]


def _generate(model, tokenizer, record, maximum):
    import torch

    ids = apply_chat_template(tokenizer, record["prompt"], add_generation_prompt=True)
    inputs = torch.tensor([ids], dtype=torch.long, device=model.device)
    attention = torch.ones_like(inputs)
    model.eval()
    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs, attention_mask=attention, do_sample=False,
            max_new_tokens=maximum, use_cache=True, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0, len(ids):].tolist()
    return output, tokenizer.decode(output, skip_special_tokens=False)


def _reload_adapter(config, cache_dir, adapter):
    from peft import PeftModel
    base = _load_base(config, cache_dir)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    return model


def overfit_smoke(config_path, cache_dir, output_dir, examples=None, max_steps=None,
                  required_loss_ratio=None):
    import torch

    config = load_config(config_path)
    train_rows, _ = _load_rows(config)
    count = examples or config["smoke"]["examples"]
    if count < 8 or count > 16:
        raise RuntimeError("Overfit smoke set must contain 8 to 16 examples")
    output = _prepare_output(output_dir)
    smoke_rows = _fixed_smoke_rows(train_rows, count, config["smoke"]["selection_seed"])
    trainer, tokenizer = _trainer(
        config, cache_dir, output, smoke_rows, smoke_rows, smoke=True, max_steps=max_steps)
    before_ids, before_text = _generate(
        trainer.model, tokenizer, smoke_rows[0], config["smoke"]["generation_max_new_tokens"])
    initial = trainer.evaluate(metric_key_prefix="initial")
    trainer.train()
    final = trainer.evaluate(metric_key_prefix="final")
    adapter = output / "adapter-final"
    trainer.save_model(str(adapter))
    trained_ids, trained_text = _generate(
        trainer.model, tokenizer, smoke_rows[0], config["smoke"]["generation_max_new_tokens"])
    del trainer
    torch.cuda.empty_cache()
    reloaded = _reload_adapter(config, cache_dir, adapter)
    reloaded.to("cuda")
    reloaded_ids, reloaded_text = _generate(
        reloaded, tokenizer, smoke_rows[0], config["smoke"]["generation_max_new_tokens"])
    if trained_ids != reloaded_ids:
        raise RuntimeError("Deterministic output changed after adapter reload")
    ratio = final["final_loss"] / initial["initial_loss"]
    required = (required_loss_ratio if required_loss_ratio is not None
                else config["smoke"]["required_final_to_initial_loss_ratio"])
    result = {
        "examples": count, "initial_loss": initial["initial_loss"],
        "final_loss": final["final_loss"], "final_to_initial_loss_ratio": ratio,
        "required_ratio": required, "memorization_gate_passed": ratio <= required,
        "base_before_training": before_text, "trained_before_reload": trained_text,
        "reloaded": reloaded_text, "reload_token_ids_identical": True,
    }
    (output / "smoke-result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if ratio > required:
        raise RuntimeError(f"Smoke loss ratio {ratio:.4f} did not meet required {required:.4f}")


def evaluate(config_path, cache_dir, output_dir, adapter):
    from transformers import Trainer, TrainingArguments

    config = load_config(config_path)
    verify_training_inputs(config, resolve_repo_path("."))
    output = _prepare_output(output_dir)
    tokenizer = load_tokenizer(config, cache_dir)
    validation = TokenizedDataset(
        read_records(resolve_repo_path(config["data"]["validation"])), tokenizer,
        config["sequence"]["max_length"])
    model = _reload_adapter(config, cache_dir, adapter)
    model.to("cuda")
    collator = CompletionOnlyCollator(tokenizer)
    args = TrainingArguments(
        output_dir=str(output), per_device_eval_batch_size=1, bf16=True,
        report_to="none", remove_unused_columns=False, prediction_loss_only=True)
    trainer = Trainer(model=model, args=args, eval_dataset=validation, data_collator=collator)
    metrics = trainer.evaluate()
    (output / "validation-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def reload_test(config_path, cache_dir, output_dir, adapter):
    import torch

    config = load_config(config_path)
    verify_training_inputs(config, resolve_repo_path("."))
    output = _prepare_output(output_dir)
    tokenizer = load_tokenizer(config, cache_dir)
    record = read_records(resolve_repo_path(config["data"]["validation"]))[0]
    first = _reload_adapter(config, cache_dir, adapter)
    first.to("cuda")
    first_ids, first_text = _generate(first, tokenizer, record, 384)
    del first
    torch.cuda.empty_cache()
    second = _reload_adapter(config, cache_dir, adapter)
    second.to("cuda")
    second_ids, second_text = _generate(second, tokenizer, record, 384)
    result = {"identical": first_ids == second_ids, "first": first_text, "second": second_text}
    (output / "reload-result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if first_ids != second_ids:
        raise RuntimeError("Deterministic adapter generation differs across reloads")
