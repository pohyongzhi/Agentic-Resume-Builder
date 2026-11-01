# How to Run

## Prerequisites

Install [Ollama](https://ollama.com/download), then:

```bash
ollama pull qwen3.5:4b-q4_K_M
ollama pull nomic-embed-text

python -m venv venv
source venv/bin/activate                 # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you use a different Ollama model, change `MODEL` in `config.py` before running experiments.

## Verify the installation

```bash
python smoke_test.py
pytest -q
```

`smoke_test.py` checks ordinary and structured model output. The pytest suite uses fake model
responses, so it does not require Ollama.

## Web interface

```bash
python app.py
```

Open http://127.0.0.1:7860 and:

1. upload a text-based `.pdf`, `.docx`, `.txt` or `.md` resume;
2. enter a job URL or paste the full job description;
3. click **Generate**;
4. preview and download the validated Markdown result.

Pasted text takes precedence over the URL. If the website blocks retrieval, the app asks for pasted
text. Scanned PDFs are rejected clearly because OCR is outside the baseline scope.

Generated files are placed in `results/outputs/`. The harness refuses paths outside that directory,
existing output files and the original resume.

## Single-pass control

The old one-prompt implementation is kept as V0:

```bash
python baseline_tailor.py --resume my_resume.pdf --jd job.txt
python baseline_tailor.py --resume my_resume.docx --jd job.txt --out tailored.md
```

This does not use the parser agents or harness. Use it only as the control.

## Batch evaluation

The checked-in synthetic evaluation set contains 100 Markdown resumes paired with 20 Markdown JDs.
Validate it before a run:

```bash
python scripts/validate_dataset.py
```

To reproduce the exact files from the versioned specification:

```bash
python scripts/generate_dataset.py --force
python scripts/validate_dataset.py
```

Run all registered workflows on all pairs:

```bash
python run_all.py --split test --repeats 1
```

Useful smaller runs:

```bash
python run_all.py --limit 1 --repeats 1
python run_all.py --workflows w1_single_pass w2_baseline --limit 1 --repeats 1
python run_all.py --categories cybersecurity data_ai --repeats 1
python run_all.py --workflows w2_baseline --skip-faithfulness
```

The evaluator uses the saved JD files in `data/jds/`, not live pages. Results append to
`results/results.jsonl`. A run is skipped only if its workflow, input pair, repeat, model and
configuration/prompt hash and frozen dataset hash already exist. Canonical outputs are stored under
`results/runs/<dataset_version>/<config_hash>/<workflow>/<pair_id>/repeat_<n>/`.

Create the summary:

```bash
python summarise.py
```

The backwards-compatible summary is written to `results/summary.csv`. Detailed summaries are in
`results/summaries/overall.csv`, `by_category.csv` and `by_fit_level.csv`. Failed runs remain in
`success_%`; metric averages use successful outputs only.

## Operational evaluation

Task success, tool accuracy, step efficiency, token cost and latency are recorded during each run.
To apply changed evaluator thresholds to existing outputs without calling the writer model again:

```bash
python evaluate_results.py
python summarise.py --results-file results/evaluated_results.jsonl
```

Task success requires completion, all required resume sections, 200–650 words and faithfulness at or
above `TASK_FAITHFULNESS_THRESHOLD`. If faithfulness was skipped, task success remains unscored
rather than being treated as a pass.

Run the robustness benchmark on a small shared subset:

```bash
python run_robustness.py --limit 20
```

For a faster smoke test, add `--skip-faithfulness`. The runner tests the original pair, formatting
changes and reordered sections, then writes `results/summaries/robustness.csv`.

## Complete reusable evaluation

To evaluate the 100 existing W1 outputs without regenerating them:

```bash
python evaluate.py all \
  --workflows w1_single_pass \
  --judge-model gemma4:e4b \
  --resume
```

Before the full run, use:

```bash
python evaluate.py all --workflows w1_single_pass --judge-model gemma4:e4b --pilot
```

The command validates the frozen dataset and saved-run compatibility, runs the NLI preflight,
scores missing faithfulness, applies the workflow-neutral rubric judge, creates only missing
formatting/section-order variants for the fixed stratified 20-pair sample, and writes reports and
blank human-review forms. Existing `results/results.jsonl` and `results/runs/` files are not changed.

Useful controls are `--limit N`, `--robustness-limit N`, `--force`, and
`--skip-faithfulness`, `--skip-judge`, `--skip-robustness`, `--skip-reports` or
`--skip-human-review`. Compatible stage caches are reused automatically; `--resume` documents that
intent. Results live under:

```text
results/evaluations/<dataset_hash>/<generation_scope>/<evaluator_hash>/
```

The exact installed judge tag must exist. No model is pulled or substituted. `gemma4:e4b` reports
8.0B parameters with Q4_K_M quantization. Judge scores, NLI faithfulness, keyword coverage, JD
similarity and operational metrics remain separate.

## Main configuration

Edit `config.py` before freezing the experiment:

- `MODEL`, `EMBED_MODEL`: Ollama models;
- `TEMPERATURE`, `SEED`, `NUM_CTX`, `THINK`: generation settings;
- `MAX_STEPS`: ReAct model-step budget;
- `MAX_TOOL_CALLS`: creator harness budget;
- `OUTPUT_DIR`: only permitted render directory;
- `PDF_MIN_CHARS_PER_PAGE`: scanned-PDF threshold;
- `JD_MIN_CHARS`, `FETCH_TIMEOUT`: page retrieval thresholds.
- `TASK_FAITHFULNESS_THRESHOLD`, `TASK_MIN_WORDS`, `TASK_MAX_WORDS`: task-success gates.
- `JUDGE_*`: independent rubric judge model and deterministic settings;
- `NLI_*`: faithfulness model, version and chunking settings;
- `ROBUSTNESS_SEED`, `REPORT_BOOTSTRAP_*`: fixed sampling and confidence-interval settings.

Changing the configuration or shared prompts changes the experiment hash, so old rows are not
silently reused.

## Common failures

**Model not found**

```bash
ollama pull qwen3.5:4b-q4_K_M
```

**Ollama connection refused**

```bash
ollama serve
```

**`SCANNED_PDF`**

Upload a resume with a text layer, or export it as DOCX/TXT. OCR is deliberately not part of this
baseline.

**`JD_FETCH_FAILED`**

Paste the JD. LinkedIn and other sites often require login or client-side JavaScript.

**`TOOL_BUDGET_EXCEEDED`**

The creator did not produce a draft that passed validation within `MAX_TOOL_CALLS`. Inspect the
tool trace in `results/results.jsonl`; do not simply remove validation.

**Dataset validation failed**

Do not run the experiment until every missing file, duplicate ID, invalid fit label, split leak or
dataset-hash mismatch reported by `scripts/validate_dataset.py` is resolved.
