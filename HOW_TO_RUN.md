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

`smoke_test.py` checks ordinary tool calls and the nested JSON argument required by
`validate_resume`. The pytest suite uses fake model responses, so it does not require Ollama.

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

## W4: Evidence-grounded generation

W4 ensures every claim in the generated resume cites source spans from the original resume and is
verified for factual accuracy:

```bash
python -c "from workflows.w4_evidence import generate; from common.llm import Tracker; \
    state = generate(Tracker(), 'fixtures/alex-markdown.md', \
    pasted_jd='Data Analyst role. Need SQL, Python and Excel.'); \
    print(state['resume_md']); print('Kept:', state['w4_stats']['kept'], \
    'Dropped:', state['w4_stats']['dropped'])"
```

Each claim must:
- Include project descriptions as cited claims, as well as summary sentences and bullets
- Cite 1-3 source spans from the original resume
- Only reference spans from the same resume section (for experience/education/projects)
- Pass deterministic rule checks (no new numbers, skills, or inflation)
- Pass LLM verification (claim is entailed by cited text)

The workflow attempts up to 2 revision rounds. Unsupported claims are dropped from the final
resume. The evidence report shows which claims were kept/dropped and why.

In the web interface, select "W4 - Evidence-grounded" to see the evidence report with citation
details for every claim.

## Single-pass control

The old one-prompt implementation is kept as V0:

```bash
python baseline_tailor.py --resume my_resume.pdf --jd job.txt
python baseline_tailor.py --resume my_resume.docx --jd job.txt --out tailored.md
```

This does not use the parser agents or harness. Use it only as the control.

## Batch evaluation

Run all registered workflows on all pairs:

```bash
python run_all.py
```

Useful smaller runs:

```bash
python run_all.py --limit 1 --repeats 1
python run_all.py --workflows w1_single_pass w2_baseline --limit 1 --repeats 1
python run_all.py --workflows w2_baseline --skip-faithfulness
```

The evaluator uses the saved JD files in `data/jds/`, not live pages. Results append to
`results/results.jsonl`. A run is skipped only if its workflow, input pair, repeat, model and
configuration/prompt hash already exist.

Create the summary:

```bash
python summarise.py
```

The summary is written to `results/summary.csv`. Failed runs remain in `success_%`; metric averages
use successful outputs only.

## Main configuration

Edit `config.py` before freezing the experiment:

- `MODEL`, `EMBED_MODEL`: Ollama models;
- `TEMPERATURE`, `SEED`, `NUM_CTX`, `THINK`: generation settings;
- `MAX_STEPS`: ReAct model-step budget;
- `MAX_TOOL_CALLS`: creator harness budget;
- `OUTPUT_DIR`: only permitted render directory;
- `PDF_MIN_CHARS_PER_PAGE`: scanned-PDF threshold;
- `JD_MIN_CHARS`, `FETCH_TIMEOUT`: page retrieval thresholds.

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

**Low nested-JSON success in `smoke_test.py`**

Use a model with stronger tool calling. The smoke test exists because a model may support simple
tools but still fail to send a full resume object to `validate_resume`.
