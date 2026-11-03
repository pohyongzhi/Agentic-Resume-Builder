# V4.2 accuracy and quality improvement

## Outcome

The local V4.2 package is ready for tokenizer audit and experimental GPU training at `runs/v4-dataset-600-v4-2/`. It preserves the V4.1 source inputs and adds a controlled accuracy-focused revision rather than indiscriminate volume.

| Item | Result |
| --- | ---: |
| Total records | 600 |
| Training records | 530 |
| Validation records | 70 |
| V4.1 source records | 550 |
| Versioned target-summary repairs | 29 |
| Unchanged V4.1 records | 521 |
| New records | 50 |
| New candidate families | 25 |
| New split | 40 train / 10 validation |

The 29 repairs change only target summaries and evidence offsets. Original resumes, JDs, family assignments and splits remain unchanged. The repair set removes negative authority framing, moves approval and ownership limits back to the relevant experience bullet, and corrects an older unsupported QA-title summary.

The 50 new examples cover finance, payroll, procurement, supply planning, customer success, HR, recruitment, marketing, content, sales operations, compliance, legal operations, research, laboratory work, facilities, events, hospitality, retail, community programmes, reporting, UX research, IT support, security monitoring, software testing and field operations. Each of the 25 new candidate families has two distinct JDs.

## Review and validation

All 79 changed or new targets received direct editorial review. This was not an independent external review.

| Check | Result |
| --- | ---: |
| Factual support | 79/79 passed |
| Actual-title fidelity | 79/79 passed |
| Constraint preservation | 79/79 passed |
| Positive-summary usefulness | 79/79 passed |
| Application usefulness | 79/79 passed |
| Rejected records | 0 |
| Additional repairs after build | 0 |
| Unsupported numeric strings | 0 |
| Unsupported experience headings | 0 |
| Invalid evidence spans | 0 |
| Direct-identifier patterns | 0 |
| Candidate-family split leaks | 0 |
| Frozen-holdout overlaps | 0 |
| Maximum holdout five-word-shingle Jaccard | 0.009288 |

Validate the ignored local package with:

```bash
python3 runs/v4-dataset-600-v4-2/validate.py
```

The training manifest, acceptance decision, review decisions, hashes and audit results are in the same ignored run directory. No paid generation call was made.

## Prompt change

The V1.2 student prompt retains all previous factual constraints and adds one style rule: summaries present supported strengths positively, while approval, ownership, supervision and temporary-duty limits remain in the supported experience bullet where they occurred. This directly addresses the negative-summary behavior observed in V4.1 without removing factual boundaries from the resume.

## Controlled training experiments

Two configurations use the same dataset, optimizer settings and random seed:

| Setting | Rank-16 primary | Rank-32 capacity ablation |
| --- | ---: | ---: |
| Config | `qwen35-4b-lora-v4-2-r16.json` | `qwen35-4b-lora-v4-2-r32.json` |
| LoRA rank / alpha | 16 / 32 | 32 / 64 |
| Learning rate | `5e-5` | `5e-5` |
| Epochs | 5 | 5 |
| Gradient accumulation | 8 | 8 |
| Effective examples per optimizer step | 8 | 8 |
| Expected optimizer steps | approximately 335 | approximately 335 |
| Checkpoint selection | lowest validation loss | lowest validation loss |

All other established settings remain unchanged: official pinned `Qwen/Qwen3.5-4B`, BF16, microbatch one, 4,096-token budget, cosine schedule, 0.05 warmup, weight decay 0.01, gradient checkpointing, cache disabled, packing disabled, completion-only loss and seed 42.

Rank 16 is the primary run. Rank 32 should run only if rank 16 completes successfully; it is a controlled capacity comparison, not an automatic replacement.

## Local and GPU commands

Set the existing cache and output root on the compute node:

```bash
export V4_CACHE="<HUGGING_FACE_CACHE>"
export V4_OUTPUT_ROOT="<PRIVATE_OUTPUT_ROOT>"
```

Tokenizer and label audits:

```bash
python -m v4_training audit-tokenizer \
  --config configs/v4/qwen35-4b-lora-v4-2-r16.json \
  --cache-dir "$V4_CACHE" --local-files-only --examples 3

python -m v4_training audit-labels \
  --config configs/v4/qwen35-4b-lora-v4-2-r16.json \
  --cache-dir "$V4_CACHE" --local-files-only --examples 3
```

Rank-16 smoke test and training:

```bash
python -m v4_training overfit-smoke \
  --config configs/v4/qwen35-4b-lora-v4-2-r16.json \
  --cache-dir "$V4_CACHE" \
  --output-dir "$V4_OUTPUT_ROOT/v4-2-r16-smoke-001"

python -m v4_training train \
  --config configs/v4/qwen35-4b-lora-v4-2-r16.json \
  --cache-dir "$V4_CACHE" \
  --output-dir "$V4_OUTPUT_ROOT/v4-2-r16-train-001"
```

Validation, per-case loss and deterministic reload:

```bash
python -m v4_training evaluate \
  --config configs/v4/qwen35-4b-lora-v4-2-r16.json \
  --cache-dir "$V4_CACHE" \
  --adapter "$V4_OUTPUT_ROOT/v4-2-r16-train-001/adapter-final" \
  --output-dir "$V4_OUTPUT_ROOT/v4-2-r16-eval-001"

python -m v4_training evaluate-cases \
  --config configs/v4/qwen35-4b-lora-v4-2-r16.json \
  --cache-dir "$V4_CACHE" \
  --adapter "$V4_OUTPUT_ROOT/v4-2-r16-train-001/adapter-final" \
  --output-dir "$V4_OUTPUT_ROOT/v4-2-r16-case-loss-001"

python -m v4_training reload-test \
  --config configs/v4/qwen35-4b-lora-v4-2-r16.json \
  --cache-dir "$V4_CACHE" \
  --adapter "$V4_OUTPUT_ROOT/v4-2-r16-train-001/adapter-final" \
  --output-dir "$V4_OUTPUT_ROOT/v4-2-r16-reload-001"
```

If rank 16 passes, repeat the smoke and training commands with `qwen35-4b-lora-v4-2-r32.json` and new `v4-2-r32-*` output directories. Never reuse an output directory.

## Model-selection gates

Validation loss is a diagnostic, not the sole acceptance criterion. The selected checkpoint must also meet these gates:

- zero fabricated employers, qualifications, dates and actual job titles in the reviewed comparison;
- exact preservation of explicit dates and numeric scope;
- at least 95% preservation of tested proficiency and authority constraints;
- required headings in every output;
- no raw control tokens;
- no reappearance of negative missing-authority summaries;
- better or equal blinded usefulness than V4.1 on the fixed comparison set.

The new `evaluate-cases` command records loss for every archival validation case and reports inherited, repaired and new-example cohorts plus the ten hardest cases. This identifies whether aggregate validation loss is driven by new constraints, target repairs or inherited records.

## Post-training quantization check

After selecting the better adapter, compare the merged BF16 model with Q6_K, Q5_K_M and Q4_K_M on the same fixed development cases. Use identical prompts and deterministic decoding. If BF16 preserves a constraint that Q4_K_M loses, treat the problem as quantization loss rather than a training defect. Select the smallest quantization that passes the factual and usefulness gates.

Do not open the frozen final benchmark until the training recipe, adapter rank and deployment quantization have been selected.
