# V4 Qwen3.5-4B LoRA training package

This package trains completion-only LoRA adapters from the accepted 450/50 V4 dataset. It does not use the local Ollama quantization. Model, cache, log, checkpoint, and adapter paths must remain outside ordinary Git.

## Frozen inputs and base

- Train: `runs/v4-dataset-500-v2/sft-train-450.jsonl` (450 records)
- Validation: `runs/v4-dataset-500-v2/sft-validation-50.jsonl` (50 records)
- System prompt: `runs/v4-dataset-500-v2/student-v1.txt`
- Data manifest: `runs/v4-dataset-500-v2/training-manifest.json`
- Base: `Qwen/Qwen3.5-4B`
- Hugging Face commit: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`

The loader checks the manifest hashes, record counts, and frozen system prompt before training. The model ID and 40-character revision are guarded in configuration.

## Environment

Use Python 3.11.11 and install `requirements-v4-training.txt` in a new environment. The pinned packages are PyTorch 2.14.1, Transformers 5.20.2, PEFT 0.21.2, TRL 1.16.0, Accelerate 1.15.0, datasets 5.1.0, safetensors 0.8.0, sentencepiece 0.2.2, tokenizers 0.23.3, huggingface-hub 2.3.0, and Jinja2 3.1.6. Select the PyTorch wheel source only after checking the cluster driver and CUDA module. Do not silently change a package version.

Example environment creation after the CUDA choice is known:

```bash
python3.11 -m venv "$V4_STORAGE_ROOT/venvs/v4-qwen35"
"$V4_STORAGE_ROOT/venvs/v4-qwen35/bin/python" -m pip install -r requirements-v4-training.txt
```

## Audits completed locally

The official pinned tokenizer uses `<|im_end|>` as EOS (ID 248046), `<|endoftext|>` as padding (ID 248044), and accepts `enable_thinking=false`. Its non-thinking generation prefix contains an empty `<think>...</think>` block; that entire prefix is masked. Only the Markdown resume and final EOS are supervised.

All 500 examples fit the 4,096-token budget without truncation:

| Split | Records | Minimum | Median | Maximum | Mean | Supervised tokens |
|---|---:|---:|---:|---:|---:|---:|
| Train | 450 | 698 | 903 | 1,322 | 916.62 | 92,280 |
| Validation | 50 | 671 | 984 | 1,336 | 976.92 | 11,546 |

The pinned safetensors index resolves 248 language-backbone projections: 32 full-attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`), 120 DeltaNet projections (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`), and 96 MLP projections (`gate_proj`, `up_proj`, `down_proj`). Vision modules, embeddings, `lm_head`, and `mtp` are excluded. The trainer repeats this check against actual `torch.nn.Linear` modules after loading weights and stops on any mismatch.

## Local and node commands

Tokenizer/template audit:

```bash
python -m v4_training audit-tokenizer --cache-dir "$HF_HUB_CACHE" --examples 3 --report runs/v4-training-audits/tokenizer-audit.json
```

Label-mask audit:

```bash
python -m v4_training audit-labels --cache-dir "$HF_HUB_CACHE" --local-files-only --examples 3
```

Pinned weight-index target audit; this downloads only the index if it is not cached:

```bash
python -m v4_training audit-modules --cache-dir "$HF_HUB_CACHE" --report runs/v4-training-audits/module-audit.json
```

One-GPU 12-example overfit smoke test:

```bash
python -m v4_training overfit-smoke --cache-dir "$HF_HUB_CACHE" --output-dir "$V4_STORAGE_ROOT/runs/v4-overfit-smoke-001" --examples 12
```

Full 450/50 pilot training, only after the smoke test passes:

```bash
python -m v4_training train --cache-dir "$HF_HUB_CACHE" --output-dir "$V4_STORAGE_ROOT/runs/v4-pilot-training-001"
```

Validation-only evaluation:

```bash
python -m v4_training evaluate --cache-dir "$HF_HUB_CACHE" --adapter "$V4_STORAGE_ROOT/runs/v4-pilot-training-001/adapter-final" --output-dir "$V4_STORAGE_ROOT/runs/v4-validation-001"
```

Adapter reload test:

```bash
python -m v4_training reload-test --cache-dir "$HF_HUB_CACHE" --adapter "$V4_STORAGE_ROOT/runs/v4-pilot-training-001/adapter-final" --output-dir "$V4_STORAGE_ROOT/runs/v4-reload-001"
```

Every output directory must be new. Training records package versions and resolved targets. Trainer logs include loss, gradient norm, learning rate, step time, supervised tokens in the last microbatch, and peak allocated GPU memory. The smoke test saves and reloads its adapter, checks deterministic output token identity, and requires final validation loss to be no more than half the initial loss. It never launches a repair or full run automatically.

## Training recipe

The initial recipe is BF16 on one GPU, LoRA rank 16, alpha 32, dropout 0.05, no bias, learning rate `1e-4`, AdamW, weight decay 0.01, cosine scheduling, warmup ratio 0.05, at most three epochs, microbatch 1, gradient accumulation 16, length 4,096, gradient checkpointing on, cache off during training, no packing, and seed 42.

## Slurm smoke template

No SMU-specific value is embedded. Substitute the eight angle-bracket placeholders in this command:

```bash
sbatch --account=<SMU_ACCOUNT> --partition=<SMU_PARTITION> --qos=<SMU_QOS> --time=<SMU_TIME_LIMIT> --gres=<SMU_GPU_GRES> --output=<SMU_STORAGE_ROOT>/logs/v4-qwen35-smoke-%j.out --export=ALL,V4_STORAGE_ROOT=<SMU_STORAGE_ROOT>,V4_PROJECT_ROOT=<SMU_PROJECT_ROOT>,V4_PYTHON=<SMU_PYTHON> slurm/v4-qwen35-overfit-smoke.sbatch
```

Before submission, confirm the account, partition, QoS, L40S GRES syntax, allowed time, private storage root, repository path, environment Python path, CUDA driver/module combination, and whether the compute node can populate the Hugging Face cache. If compute nodes lack network access, pre-stage the pinned snapshot in the same private cache without committing it.
