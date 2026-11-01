# Resume Agent Lab

**Research question:** which agentic technique makes a small local model write the best tailored
resume, at what cost, and without inventing facts?

The shared baseline is now an end-to-end agentic system. A resume parser and JD retriever run in
parallel, a creator tailors the resume, and a code harness requires validation before rendering.
The original one-prompt implementation is `w1_single_pass`, the lower-bound control. The shared
agentic baseline is `w2_baseline`.

| Workflow | Technique | Status |
|---|---|---|
| `w1_single_pass` | W1: one prompt, no tools or guardrails (lower-bound control) | done |
| `w2_baseline` | W2: parallel extraction agents + creator + validation/render harness | done |
| `w3_critic` | W3: parallel fact, recruiter and formatting critics + rewrites | done |

## Architecture

```text
resume file ──> resume_parser ──┐
                               ├─> check_extractions ─> resume_creator ─> harness
JD URL/text ──> jd_retriever ──┘                              │
                                                              ├─ validate_resume
                                                              └─ render_resume -> results/outputs/
```

LangGraph starts both extraction nodes in the same step and waits for both. The creator only sees
their structured `CandidateData` and `JobData`. If either extraction fails, generation stops.

The baseline creator uses Ollama's constrained `GeneratedResume` schema instead of putting a large,
nested resume inside a tool argument. Python then invokes the harness operations. This avoids
malformed JSON from small models while keeping every validation and file-safety rule.

The harness enforces these rules in Python rather than relying on the prompt:

- only `validate_resume` and `render_resume` are allowed;
- rendering requires the exact latest draft to have passed validation;
- employers, titles, dates, education, skills and certifications are checked against candidate data
  and the original resume text;
- tool calls are capped by `MAX_TOOL_CALLS`;
- output files must stay inside `OUTPUT_DIR`, cannot overwrite the input, and cannot replace an
  existing file.

These checks catch structural and selected factual errors. They do not prove that every rewritten
sentence is accurate; the separate evaluation still measures claim-level faithfulness.

## Setup

1. Install [Ollama](https://ollama.com/download), then pull the configured models:

   ```bash
   ollama pull qwen3.5:4b-q4_K_M
   ollama pull nomic-embed-text
   ```

2. Create the environment:

   ```bash
   python -m venv venv
   source venv/bin/activate          # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Check ordinary and nested-JSON tool calls:

   ```bash
   python smoke_test.py
   ```

4. Run tests (they stub all LLM calls and do not need Ollama):

   ```bash
   pytest -q
   ```

## Use the agentic baseline

```bash
python app.py
```

Open http://127.0.0.1:7860, upload a text-based PDF, DOCX, TXT or Markdown resume, then provide a
job URL or paste the JD. Pasted text takes precedence and is the fallback for pages that require
login or JavaScript. Use the workflow selector to compare the single-pass control, agentic baseline
and multi-critic workflow interactively. Harness-validated Markdown resumes can be
previewed and downloaded; unguarded comparison workflows are clearly marked as experimental.

## W3 multi-critic workflow

`w3_critic` creates a structured draft, then runs three separate reviewers in parallel:

- the fact-checker looks for claims unsupported by the original resume;
- the recruiter checks role relevance, evidence and shortlist quality;
- the formatting reviewer checks completeness, consistency, concision and scanability.

Their structured feedback is combined and sent back to the creator. The workflow stops early when
all three approve with no major or critical issues, or after at most two revisions. The latest draft
must still pass the same deterministic harness used by the baseline before it is rendered. This
workflow is expected to improve quality and faithfulness at the cost of more model calls and time.

`baseline_tailor.py` is intentionally still the non-agentic single-pass CLI, not the new baseline:

```bash
python baseline_tailor.py --resume my_resume.pdf --jd job.txt
```

## Run experiments

```bash
python scripts/validate_dataset.py
python run_all.py --split test --limit 1 --repeats 1
python summarise.py
```

Evaluation uses saved JD snapshots rather than live URLs for reproducibility. Results are appended
to `results/results.jsonl`; canonical per-run resumes, metrics and traces go under `results/runs/`.
Runs are skipped only when workflow, pair, repeat, model, config/prompt hash and frozen dataset hash
all match. Failed runs remain in the summary's success rate.

The first faithfulness run downloads `cross-encoder/nli-deberta-v3-small`.

### Evaluator metrics

Every saved run records resume-quality metrics (JD similarity, keyword coverage and NLI
faithfulness) plus operational metrics:

- task success on the frozen test split, gated by completion, required sections, length and
  faithfulness;
- tool-call and argument accuracy, required sequence validity and recovery after failed calls;
- workflow-aware step efficiency, repeated calls and budget utilisation;
- input/output tokens, total tokens, model time and end-to-end latency;
- robustness on formatting and section-order variants of a small shared test subset.

Operational thresholds can be changed and applied to saved outputs without regenerating resumes:

```bash
python evaluate_results.py
python summarise.py --results-file results/evaluated_results.jsonl
python run_robustness.py --limit 20 --skip-faithfulness
```

Robustness results are written to `results/summaries/robustness.csv`. Tool metrics are `N/A` for
workflows such as W1 that intentionally use no tools.

### Reusable offline evaluation pipeline

The full evaluator preserves generation artifacts and writes versioned records under
`results/evaluations/<dataset_hash>/<generation_scope>/<evaluator_hash>/`. Run W1 faithfulness,
the shared rubric judge, the fixed 20-pair robustness benchmark, reports and human-review packet:

```bash
python evaluate.py all \
  --workflows w1_single_pass \
  --judge-model gemma4:e4b \
  --resume
```

The installed `gemma4:e4b` judge is an 8.0B Q4_K_M model, not a 3B model. The command requires the
exact tag and never pulls or silently substitutes a judge. Use `--pilot` before a new configuration,
`--limit N` for saved-output scoring, `--robustness-limit N` for perturbations, or the
`--skip-faithfulness`, `--skip-judge`, `--skip-robustness`, `--skip-reports` and
`--skip-human-review` flags to omit expensive stages.

The judge sees the original resume, JD and tailored resume and applies the same five-dimension
rubric to every workflow. Its score remains separate from NLI faithfulness, keyword coverage,
embedding similarity and operational metrics. NLI and judge caches are content/configuration
addressed, failed evaluations remain unscored rather than becoming zero, and task success is
recalculated only from completion, structure, length and the predeclared faithfulness threshold.

The evaluation directory contains `evaluations.jsonl`, `robustness/evaluations.jsonl`,
`robustness/sample_manifest.json`, detailed CSV/JSON summaries, NLI preflight evidence, full judge
provenance and a blinded `human_review/` packet. Human forms are blank; automated scores are not
claimed as human-validated until independent ratings are completed.

## Code layout

```text
app.py                    Gradio UI for the agentic baseline
baseline_tailor.py        single-pass V0 CLI
config.py                 shared model, harness and extraction settings
common/
  schemas.py              CandidateData, JobData and GeneratedResume
  state.py                shared LangGraph state and parallel error reducer
  tools.py                resume text extraction and JD page retrieval
  extraction.py           resume-parser and JD-retriever agent nodes
  pipeline.py             parallel fan-out/fan-in graph
  harness.py              validation, rendering, permissions and budgets
  agent.py                shared ReAct tool loop
  llm.py                  only Ollama entry point; logs every call
workflows/
  w1_single_pass.py       W1 one-prompt control
  w2_baseline.py          W2 agentic baseline creator
  w3_critic.py            parallel specialist critics + bounded revision loop
eval/metrics.py           shared evaluator metrics
eval/operational.py       task, tool, step and cost metrics
eval/robustness.py        input perturbations and stability aggregation
run_all.py                batch runner
evaluate_results.py       rescore saved outputs without regeneration
run_robustness.py         shared perturbation benchmark
summarise.py              per-workflow summary
scripts/
  generate_dataset.py     deterministic synthetic dataset materialiser
  validate_dataset.py     dataset integrity and leakage checks
tests/                    harness, extraction-tool and graph tests
```

## Adding a workflow

All variants should reuse `common/pipeline.py` so extraction and guardrails stay constant. Supply
another creator factory in place of `workflows.w2_baseline.make_creator`, call the model only through
`common.llm.chat`, return the standard workflow dictionary, and register it in
`workflows/__init__.py`.

Keep `config.py`, extraction, `BASE_RULES`, the harness and evaluation data frozen across variants.
The test labels must never be shown to a workflow.

## Evaluation data

- `data/resumes/*.md`: 100 fictional candidate resumes.
- `data/jds/*.md`: 20 synthetic JD snapshots across five categories.
- `data/jds/*.keywords.json`: evaluator-only must-have keywords and synonyms.
- `data/pairs.csv`: 100 explicit resume/JD pairs with category, seniority and fit metadata.
- `data/dataset.json`: frozen version, counts, provenance and content hash.
- `data/generation/`: reproducible specification and raw generation records.

Each JD has two strong-fit, two moderate-fit and one weak-fit resume. Regenerate and verify with:

```bash
python scripts/generate_dataset.py --force
python scripts/validate_dataset.py
```

Review every keyword file and a stratified sample of resume/JD pairs before treating the dataset as
final. Workflows receive resume and JD Markdown only; keyword and fit labels stay in the evaluator.
