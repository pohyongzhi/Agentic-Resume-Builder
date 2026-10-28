# How to run Agentic Resume Builder V1

Create and activate a project-local Python 3.11 environment, install the repository requirements, and confirm Ollama is running:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
ollama pull qwen3.5:4b-q4_K_M
ollama list
```

The original `qwen3.5:4b-q4_K_M` is the baseline and shared default. Workflow 4 alone requests `agentic-resume-v4-1:q4_K_M`; that tag must already have been imported from the selected local GGUF.

Start the interface:

```bash
python app.py
```

Workflow 1 Baseline is selected initially. The interface exposes the two implemented workflows and shows the model associated with the selection.

Run one workflow from the evaluation CLI:

```bash
python run_all.py --workflows baseline --limit 2 --repeats 1
python run_all.py --workflows fine_tuned_small_generator --limit 2 --repeats 1
```

Run both implemented workflows on identical inputs:

```bash
python run_all.py --workflows baseline fine_tuned_small_generator --limit 2 --repeats 1
```

Planned workflow IDs are registered for experiment continuity but cannot be executed until their techniques are implemented. See `README.md` and `release/v1/workflows.md` for their status.

Troubleshooting:

- If Ollama refuses the connection, start the local Ollama application or `ollama serve`.
- If a model is missing, inspect `ollama list`; do not substitute the fine-tuned tag for the baseline.
- PDF and DOCX extraction require `pypdf` and `python-docx` from `requirements.txt`.
- Results are written to the configured local results path and include the actual model tag in every LLM trace.
