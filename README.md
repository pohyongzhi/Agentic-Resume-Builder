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
python run_all.py --limit 1 --repeats 1
python summarise.py
```

Evaluation uses saved JD snapshots rather than live URLs for reproducibility. Results are appended
to `results/results.jsonl`; generated resumes go to `results/outputs/`. Runs are skipped only when
workflow, pair, repeat, model and config/prompt hash all match. Failed runs remain in the summary's
success rate.

The first faithfulness run downloads `cross-encoder/nli-deberta-v3-small`.

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
run_all.py                batch runner
summarise.py              per-workflow summary
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

- `data/resumes/*.md`: fictional candidates only.
- `data/jds/*.md`: saved JD snapshots.
- `data/jds/*.keywords.json`: hand-labelled must-have keywords and synonyms.
- `data/pairs.csv`: resume/JD pairings and optional source URLs.

The sample is a placeholder. Expand it before the final experiment and freeze all workflows before
looking at test results.
