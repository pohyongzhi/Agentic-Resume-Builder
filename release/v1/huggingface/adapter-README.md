---
base_model: Qwen/Qwen3.5-4B
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
- qwen3.5
- peft
- lora
- resume-tailoring
- completion-only
---

# Agentic Resume Fine-tuned V1 — LoRA Adapter

This private model repository contains the selected V4.1 LoRA adapter for the Agentic Resume Builder experiment. It tailors an original resume to a target job description while being instructed to preserve candidate facts and omit unsupported requirements.

This adapter is **not** the product baseline. The comparison baseline remains the original `qwen3.5:4b-q4_K_M`; this artifact is used only by the fine-tuned workflow variant.

## Base model

- Model: `Qwen/Qwen3.5-4B`
- Pinned revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Architecture: `Qwen3_5ForConditionalGeneration`
- Base-model license: Apache-2.0

The base weights are not included. Loading this adapter requires the pinned official base model.

## Training

- Method: LoRA, completion-only supervised fine-tuning
- Development records: 550
- Training records: 490
- Validation records: 60
- LoRA rank: 16
- Alpha: 32
- Dropout: 0.05
- Sequence length: 4096
- Epochs: 3
- BF16 training
- Thinking mode disabled
- Assistant resume tokens and terminal EOS supervised; prompt and padding tokens masked

The dataset is not distributed with this model. It combines fictional and screened, sanitized source material. Some original free-text provenance, consent and redistribution questions remain unresolved, so the adapter and source derivatives should remain private pending renewed review.

## Evaluation

- Validation loss: `0.5822656750679016`
- Adapter reload generation was deterministic in the recorded test.
- Adapter and merged-model verification outputs were token-identical.
- A seven-case local Ollama development comparison found V4.1 preferable in two targeted cases, the comparison model preferable in one, and four ties.

This is a small development evaluation, not a final benchmark. The frozen final holdout was not used for model selection.

## Loading

Compatible package versions used during training included Transformers 5.20.2 and PEFT 0.21.2.

```python
import torch
from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

base_id = "Qwen/Qwen3.5-4B"
base_revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
adapter_id = "echokes/agentic-resume-finetuned-v1"

tokenizer = AutoTokenizer.from_pretrained(adapter_id)
base = Qwen3_5ForConditionalGeneration.from_pretrained(
    base_id,
    revision=base_revision,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, adapter_id)
model.eval()
```

Apply the official Qwen3.5 chat template with `enable_thinking=False`. Use the evidence-constrained system prompt recorded in the project release when reproducing application behavior.

## Limitations and intended use

- Intended for local experimental resume tailoring and workflow comparison.
- Generated resumes require human factual and editorial review.
- The model may omit relevant evidence or introduce undesirable negative framing in summaries.
- It must not infer candidate qualifications from a job description.
- It must not be used to make autonomous hiring decisions or rank people.
- The adapter has not been cleared for public redistribution.

## Integrity

`adapter_model.safetensors` SHA-256:

```text
1d059d631ff308c3077cc25c7c433389178b47ae45d448a06ad2749c518ad808
```

Project and reproducibility records: [Agentic Resume Builder, `v1-local`](https://github.com/pohyongzhi/Agentic-Resume-Builder/tree/v1-local)
