# Agentic Resume Builder

Agentic Resume Builder is a local experimental application for comparing six resume-tailoring workflows under a shared input and output contract. Product version 1.0.0 keeps the original Qwen model as the control and uses the selected V4.1 fine-tune only in Workflow 4.

## Workflow design

| No. | Workflow ID | Technique | Model | Status |
| ---: | --- | --- | --- | --- |
| 1 | `baseline` | Single-pass baseline | `qwen3.5:4b-q4_K_M` | Implemented; default |
| 2 | `multi_critic_review` | Multi-critic review loop | Base model | Planned |
| 3 | `evidence_grounded_generation` | Evidence-grounded generation | Base model | Planned |
| 4 | `fine_tuned_small_generator` | Fine-tuned small generator | `agentic-resume-v4-1:q4_K_M` | Implemented |
| 5 | `multi_draft_selector` | Several drafts plus selector | Base model | Planned |
| 6 | `planner_section_writers` | Planner and section writers | Base model | Planned |

Only implemented workflows are executable. The historical `w4_ats_optimiser` module is an earlier example and is not Workflow 4 in this design.

## Setup

Prerequisites are Python 3.11, [Ollama](https://ollama.com/), and the baseline model. Create an ignored virtual environment rather than installing packages globally:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull qwen3.5:4b-q4_K_M
```

Workflow 4 additionally requires the local tag `agentic-resume-v4-1:q4_K_M`. Its selected GGUF is stored outside Git under `/Users/kester/Desktop/Agentic-Resume-Builder-Models/v4-1-q4_K_M/`; the exact file and hash are recorded in `release/v1/manifest.json`.

Teammates can install the fine-tuned model without a Hugging Face account:

```bash
mkdir -p agentic-resume-finetuned-v1-GGUF
cd agentic-resume-finetuned-v1-GGUF
curl -L -C - -o agentic-resume-v4-1-no-nextn-q4_K_M.gguf \
  https://huggingface.co/echokes/agentic-resume-finetuned-v1-GGUF/resolve/main/agentic-resume-v4-1-no-nextn-q4_K_M.gguf
curl -L -o Modelfile \
  https://huggingface.co/echokes/agentic-resume-finetuned-v1-GGUF/resolve/main/Modelfile
ollama create agentic-resume-v4-1:q4_K_M -f Modelfile
cd ..
```

Confirm both tags before running comparisons:

```bash
ollama list
ollama show qwen3.5:4b-q4_K_M
ollama show agentic-resume-v4-1:q4_K_M
```

## Run

Start the web interface:

```bash
python app.py
```

Open <http://127.0.0.1:7860>. Workflow 1 Baseline is selected by default; the selector exposes only implemented workflows and displays the selected workflow and model.

Run the baseline directly:

```bash
python baseline_tailor.py --resume resume.pdf --jd job.txt --out tailored.md
```

Run a deterministic comparison on the repository fixtures:

```bash
python run_all.py --workflows baseline fine_tuned_small_generator --limit 2 --repeats 1
```

`common.llm.chat()` uses `BASE_MODEL` when no model is provided. Workflow 4 is the only workflow that supplies `FINE_TUNED_MODEL` as an explicit override. Each call records the actual model in its trace.

## Repository structure

```text
common/          shared prompts, model wrapper, data helpers
workflows/       workflow registry and implementations or status metadata
eval/            comparison metrics
v4_data/         reusable dataset contracts and validation
v4_training/     reproducible LoRA training and evaluation tools
configs/v4/      pinned V4 training configurations
docs/v4/         historical research and experiment documentation
release/v1/      V1 manifest, checksums, workflow record and limitations
runs/            ignored local datasets and evaluation artifacts
```

The selected V4.1 dataset, training evidence, local comparison, frozen holdout and source provenance remain under ignored `runs/` paths listed in the release manifest. Model weights remain outside ordinary Git.

## Limitations

The fine-tuned Workflow 4 model works and improved several constraint-retention cases, but it is not consistently superior to the baseline. It can produce undesirable negative framing in some summaries. This repository is a local experimental comparison, not an assurance that generated resumes are accurate, suitable, private or ready to submit without review. Public-source derivatives are restricted to local development pending renewed licensing and privacy review.

See [release/v1/README.md](release/v1/README.md) for the release record and [docs/v4/README.md](docs/v4/README.md) for the historical V4 research workstream.
