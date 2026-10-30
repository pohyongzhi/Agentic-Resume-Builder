---
base_model:
- Qwen/Qwen3.5-4B
- echokes/agentic-resume-finetuned-v1
library_name: gguf
pipeline_tag: text-generation
license: apache-2.0
tags:
- qwen3.5
- gguf
- ollama
- q4_k_m
- resume-tailoring
---

# Agentic Resume Fine-tuned V1 — GGUF

This public repository contains the deployable GGUF produced from the selected V4.1 Agentic Resume Builder LoRA experiment. It is published for reproducible school-project use and can be downloaded without a Hugging Face account.

The artifact was created by merging the V4.1 adapter into the pinned official `Qwen/Qwen3.5-4B` base, converting the merged model to GGUF, excluding the auxiliary next-token-prediction tensors used by the source architecture, and quantizing to `Q4_K_M` for local inference.

This model is **not** the project baseline. It is the fine-tuned model used by the designated fine-tuned workflow. The baseline remains the original `qwen3.5:4b-q4_K_M`.

## Artifact

- File: `agentic-resume-v4-1-no-nextn-q4_K_M.gguf`
- Quantization: `Q4_K_M`
- Size: approximately 2.7 GB
- Historical Ollama tag: `agentic-resume-v4-1:q4_K_M`

SHA-256:

```text
30b3ac98715ad1e0aee49477606840e00d0e4339725f8c31e6f99ac56596f804
```

## Ollama usage

Download both files into the same directory. No Hugging Face login is required:

```bash
curl -L -C - -o agentic-resume-v4-1-no-nextn-q4_K_M.gguf \
  https://huggingface.co/echokes/agentic-resume-finetuned-v1-GGUF/resolve/main/agentic-resume-v4-1-no-nextn-q4_K_M.gguf

curl -L -o Modelfile \
  https://huggingface.co/echokes/agentic-resume-finetuned-v1-GGUF/resolve/main/Modelfile
```

Then import the model:

```bash
ollama create agentic-resume-v4-1:q4_K_M -f Modelfile
ollama show agentic-resume-v4-1:q4_K_M
```

The historical tag above is the tag expected by the current application. The supplied Modelfile uses a relative path and configures Qwen terminal stop tokens. Application prompts and generation settings remain in the Agentic Resume Builder code repository.

## Evaluation and limitations

- The GGUF produced the required Markdown headings in all seven recorded development cases.
- No raw terminal control token appeared in those seven outputs.
- V4.1 improved several constraint-retention cases but was not consistently superior to the comparison model.
- It produced undesirable negative framing in one reviewed summary.
- Quantization introduces an additional difference from the BF16 adapter evaluation.
- Generated resumes require human factual and editorial review.
- The model must not be used for autonomous hiring or candidate ranking.
- This artifact is published for this school project, but a formal legal or privacy review of the source-data lineage has not been completed.
- Public availability does not grant permission to redistribute the underlying training dataset, which is not included.

The dataset is intentionally not included.

Project and reproducibility records: [Agentic Resume Builder, `v1-local`](https://github.com/pohyongzhi/Agentic-Resume-Builder/tree/v1-local)
