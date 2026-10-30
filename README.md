# Agentic Resume Builder

Agentic Resume Builder is a local experiment for comparing seven resume-tailoring variants under a shared input, output and evaluation contract.

- Product: Agentic Resume Builder
- Product version: `1.0.0`
- Active experiment: `V5`
- Shared base model: `qwen3.5:4b-q4_K_M`
- Fine-tuned model: `agentic-resume-v4-1:q4_K_M`

Product versioning, workflow-experiment versioning and historical model-experiment names are deliberately separate.

## V5 variant design

| No. | Stable ID | Technique | Model | Status |
| ---: | --- | --- | --- | --- |
| 1 | `one_pass` | Resume and JD in one prompt | Base model | Implemented; default |
| 2 | `agentic_baseline` | Resume parser, JD retriever, creator, validator and renderer | Base model | Planned |
| 3 | `multi_critic_review` | Separate fact, recruiter and formatting critics | Base model | Planned |
| 4 | `evidence_grounded_generation` | Claims linked to source-resume spans and verified | Base model | Planned |
| 5 | `fine_tuned_small_generator` | Selected V4.1 LoRA-derived generator | Fine-tuned model | Implemented |
| 6 | `multi_draft_selector` | Several drafts plus a selector | Base model | Planned |
| 7 | `planner_section_writers` | Planner, parallel section writers and assembler | Base model | Planned |

Only implemented variants are executable or shown in the application selector. The former V1 `baseline` name referred to the one-pass control; V5 names it `one_pass` so the distinct agentic baseline can occupy Variant 2. The legacy `w4_ats_optimiser` example is not registered as a V5 variant.

## Setup

Install Python 3.11 and [Ollama](https://ollama.com/), then create an isolated environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull qwen3.5:4b-q4_K_M
```

Install the fine-tuned model without a Hugging Face account:

```bash
MODEL_DIR="$HOME/Agentic-Resume-Builder-Models/fine-tuned-v1"
mkdir -p "$MODEL_DIR"

curl -L -C - \
  -o "$MODEL_DIR/agentic-resume-v4-1-no-nextn-q4_K_M.gguf" \
  https://huggingface.co/echokes/agentic-resume-finetuned-v1-GGUF/resolve/main/agentic-resume-v4-1-no-nextn-q4_K_M.gguf

curl -L \
  -o "$MODEL_DIR/Modelfile" \
  https://huggingface.co/echokes/agentic-resume-finetuned-v1-GGUF/resolve/main/Modelfile

ollama create agentic-resume-v4-1:q4_K_M -f "$MODEL_DIR/Modelfile"
```

## Run

Start the application:

```bash
python app.py
```

Open <http://127.0.0.1:7860>. Variant 1 One-pass is selected by default. Choose Variant 5 to run the fine-tuned model.

Run both implemented variants on identical repository fixtures:

```bash
python run_all.py \
  --workflows one_pass fine_tuned_small_generator \
  --limit 2 --repeats 1
```

The shared model wrapper uses the base model when no override is supplied. Only Variant 5 explicitly supplies the fine-tuned tag, and every call records the actual model used.

## Repository structure

```text
common/          shared prompts, model wrapper, agent loop and data helpers
workflows/       V5 implementations and planned-variant metadata
eval/            shared comparison metrics
docs/v5/         current seven-variant experiment record
docs/v4/         historical fine-tuning research and training record
v4_data/         reusable dataset contracts and validation
v4_training/     reproducible LoRA training and evaluation tools
release/v1/      frozen V1 release record and artifact checksums
runs/            ignored local datasets and experiment artifacts
```

Model weights remain outside ordinary Git. The runnable GGUF is public on Hugging Face for school-project use; the adapter and training data remain private.

## Limitations

Five V5 variants are not implemented yet and are marked planned. The fine-tuned model works and improved several constraint-retention cases, but it is not consistently superior to the base model and can produce undesirable negative framing. Generated resumes require human factual and editorial review and must not be used for autonomous hiring decisions.

See [docs/v5/README.md](docs/v5/README.md) for V5 scope, [release/v1/README.md](release/v1/README.md) for the preserved V1 release, and [docs/v4/README.md](docs/v4/README.md) for historical fine-tuning research.
