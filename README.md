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
| `w4_evidence` | W4: source-span grounding and claim verification | done |
| `w5_fine_tuned` | W5: one-pass generation with the selected V4.1 fine-tuned model | done |
| `w6_best_of_n` | W6: three strategy drafts + two-order selector + validation/render harness | done |
| `w7_plan_execute` | W7: planner + parallel section writers + assembler + validation/render harness | done |

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

   W5 additionally requires the public fine-tuned GGUF. Follow
   [`HOW_TO_RUN.md`](HOW_TO_RUN.md#install-the-w5-fine-tuned-model) to import it as
   `agentic-resume-v4-1:q4_K_M`. The base model remains the default for every other workflow.

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

W3 feedback is limited to five concise issues per critic. Critic responses have a 1,536-token
budget and drafts a 4,096-token budget; invalid JSON/schema responses get one corrective retry.
Failed responses are retained in traces for debugging. Protected-fact validation errors are also
fed into the revision loop, while final rendering still requires successful harness validation.
W3 delegates skill support and paraphrase checks to its LLM fact-checker; its harness does not
require literal skill matches. Other protected fields, certifications, projects and file-safety
checks remain enforced. W2 retains its original skill-matching check.
These settings and `W3_GENERATION_VERSION` are included in the generation configuration hash,
so results produced before this reliability change remain separate.

`baseline_tailor.py` is intentionally still the non-agentic single-pass CLI, not the new baseline:

```bash
python baseline_tailor.py --resume my_resume.pdf --jd job.txt
```

## W5 fine-tuned small generator

`w5_fine_tuned` sends the same resume/JD contract and shared factuality prompt through the selected
V4.1 fine-tuned model. It is an experimental comparison workflow, not the application default:

```bash
python run_all.py --workflows w5_fine_tuned --split test --limit 1 --repeats 1
```

Every call records the actual Ollama model tag. The current fine-tuned model works, but the limited
development comparison did not show a consistent overall win over the base model and some outputs
used undesirable negative framing. Compare it using the shared evaluator rather than assuming that
fine-tuning alone improves every resume.

## W7 plan-and-execute

`w7_plan_execute` plans first, then writes. A planner returns one JSON plan: the candidate's
header, the tailored resume's sections in order, what each section leads with and cuts, and a map
from each JD requirement to a line quoted from the resume (or `null` when the resume has no
evidence, which writers treat as UNSUPPORTED). LangGraph's `Send` then starts one writer per
planned section in the same step; each sees the original resume and only its part of the plan. An
assembler merges the sections in the planned order, removes repeats and contradictions, and drops
cut items and UNSUPPORTED requirements.

```bash
python run_all.py --workflows w7_plan_execute --split test --limit 1 --repeats 1
```

W7 draft generation makes 1 + N + 1 model calls, where N is the number of planned sections.
Finalisation adds two model calls: extraction of candidate facts from the original resume and
schema-constrained transcription of the assembled draft. Actual `validate_resume` and
`render_resume` calls then execute through `ToolHarness`. On success, both tool events are logged
and the app downloads the harness-saved file. All writers share one Ollama server, so "parallel"
may just queue; report `wall_seconds` as latency. The plan is logged as a `w7_plan` event.
See `docs/w7-plan-execute/` for the diagram and design notes.

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

### W6 Best-of-N

W6 generates skills-first, impact-first and JD-wording-alignment drafts by default. It rejects
structurally incomplete drafts, scores the remaining drafts with the same rubric in forward and
reverse order, and returns only a draft that passes both selector orders. The shared NLI evaluator
is intentionally applied afterward, like every other workflow, so it remains an independent
comparison metric.

Run a one-pair smoke test before starting the full dataset:

```bash
python run_all.py --workflows w6_best_of_n --split test --limit 1 --repeats 1
```

Run all 100 frozen test pairs once:

```bash
python run_all.py --workflows w6_best_of_n --split test --repeats 1
```

`W6_NUM_CANDIDATES` in `config.py` accepts 1–6 strategies and is included in the generation
configuration hash. Per-candidate scores and the selected strategy are stored in
`workflow_details`; canonical resumes, metrics and traces are stored below `results/runs/`.
Benchmark one pair on the evaluation machine before estimating the full runtime because W6 makes
at least seven model calls per successful pair, including the two finalisation calls, and may
retry incomplete selector responses. Successful finalisation also records two harness tool calls.

### W6 and W7 harness finalisation

Both workflows retain their existing Markdown drafting techniques. `common/finalisation.py`
parses candidate facts from the original resume in one structured call, then transcribes only the
selected/assembled draft into `GeneratedResume` in a second structured call. The transcriber sees
neither the JD nor the parsed source facts and is instructed to preserve the draft verbatim.
Validation uses the shared harness, including W2's skill checks. Rendering is permitted only for
the exact validated draft and uses the shared fixed template, so final section order follows
that template rather than the strategy or plan. Tool permissions, the tool budget, output-folder
restrictions and original-file/overwrite protections remain enforced by `ToolHarness`.

If a draft starts with a resume section rather than an identity header, transcription constrains
the missing name to an empty string. The harness then fills it from the parsed source, as it does
for other omitted identity fields. An explicit header with a changed name still fails validation.

Preparation failures and final validation rejections return errors with no download. Rejected
Markdown remains in `workflow_details.pre_harness_resume_md` for research, without marking the
run successful. There is no extra repair or critique loop in this finalisation stage. Model calls
use the shared tracker; harness calls record tool names, arguments, outcomes and elapsed time.
Historical traces without a harness marker keep their previous operational interpretation.

These changes alter generation identities and must be benchmarked as new runs. Do not retrofit
tool counts or reuse historical output scores as results for these harness-integrated workflows.
The transcription model can still change wording or omit content despite its instructions;
evaluate the final rendered text independently with NLI and the shared quality judge. Neither
the harness nor a successful transcription proves every rewritten sentence is factually correct.

After generation, create operational summaries with `python summarise.py`. To apply the complete
shared offline evaluator, including the independent rubric judge, run:

```bash
python evaluate.py all --workflows w6_best_of_n --judge-model gemma4:e4b --resume
```

### Evaluator metrics

Every saved run records resume-quality metrics (JD similarity and keyword coverage), three
separate faithfulness components, and operational metrics. Faithfulness comprises claim support,
protected-field violations and copying of evaluator keyword families absent from the source
resume.

- task success on the frozen test split, gated by completion, required sections, length, claim
  faithfulness of at least `0.8`, no protected-field violation and no fabrication-bait violation;
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

### Reusable evaluation pipeline

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
recalculated only from completion, structure, length and the predeclared component gates.

For the shared hosted judge, use Groq's `openai/gpt-oss-120b`. Keep the API key in the environment
only. First inspect the four-request dry run, then execute the fail-closed calibration:

```bash
python -m eval.judge_calibration

export GROQ_API_KEY='replacement-key'
python -m eval.judge_calibration \
  --provider groq \
  --model openai/gpt-oss-120b \
  --reasoning-effort medium \
  --output-dir results/judge-calibration/gpt-oss-120b-v1 \
  --execute
```

Do not start bulk scoring unless `calibration-result.json` reports `"passed": true`. The hosted
judge uses strict JSON Schema output and no model tools. Its API response ID, returned model,
system fingerprint and reported token usage are archived with each judgment. To score saved
outputs after calibration, requests are spaced by 15 seconds for the Groq free tier:

```bash
python evaluate.py score \
  --workflows w5_fine_tuned \
  --judge-provider groq \
  --judge-model openai/gpt-oss-120b \
  --skip-faithfulness \
  --limit 6 \
  --resume
```

Use the identical provider, model, rubric version and reasoning effort for every workflow. Never
compare scores produced by different judge configurations as if they came from one leaderboard.

The evaluation directory contains `evaluations.jsonl`, `robustness/evaluations.jsonl`,
`robustness/sample_manifest.json`, detailed CSV/JSON summaries, NLI preflight evidence, full judge
provenance and a blinded `human_review/` packet. Human forms are blank; automated scores are not
claimed as human-validated until independent ratings are completed.

For a faithfulness-only rescore of existing outputs, Ollama and the rubric judge are not required:

```bash
HF_HUB_OFFLINE=1 python evaluate.py score \
  --workflows w1_single_pass \
  --skip-judge \
  --resume
HF_HUB_OFFLINE=1 python evaluate.py report \
  --workflows w1_single_pass \
  --skip-judge
```

The current W1 result and evaluator limitations are recorded in
[`docs/evaluation/faithfulness-w1-v3-summary.json`](docs/evaluation/faithfulness-w1-v3-summary.json).
Detailed evaluation records are reproducible local artifacts and are ignored by Git.

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
  w4_evidence.py          evidence-grounded generation
  w6_best_of_n.py         W6 strategy drafts + two-order selector
  w7_plan_execute.py      W7 planner + parallel section writers + assembler
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
