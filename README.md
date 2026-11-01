# Resume Agent Lab (IS469 group project starter)

**Question:** which agentic technique makes a small local model (Qwen 4B) write the best tailored resume,
at what cost, and without making things up?

Same model, same inputs, same base prompt, same scorecard for every workflow. Only the technique changes.

| Workflow | Owner | Technique | Status |
|---|---|---|---|
| baseline | everyone | one prompt, no tools (control) | ✅ done |
| w1_research | | web search + Harvard guide RAG | stub in `workflows/_stubs.py` |
| w2_recruiter | | recruiter agent reviews, writer revises | stub |
| w3_critic | | critic + verify_claims, up to 2 rewrites | stub |
| w4_ats_optimiser | | keyword check → rewrite loop | ✅ worked example |
| w5_plan_first | | plan, then write | stub |
| w6_best_of_n | | three strategy drafts + rubric selector | ✅ implemented |

---

## Setup (everyone, ~20 min)

**1. Install Ollama** from https://ollama.com/download, then in a terminal:
```bash
ollama pull qwen3.5:4b-q4_K_M
ollama pull nomic-embed-text
```

**2. Python environment** (inside this folder):
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Check the model can call tools** (do this before building anything):
```bash
python smoke_test.py
```
Aim for ≥80% valid tool calls. If it's much lower, tell the group before anyone builds a workflow.

**4. Quick end-to-end test** on the sample data:
```bash
python run_all.py --limit 1 --repeats 1
python summarise.py
```
The first run downloads the NLI model for the faithfulness check (a few hundred MB).

---

## Try the baseline on your own resume

Web page (easiest):
```bash
python app.py
```
Open http://127.0.0.1:7860, upload your resume (.pdf/.docx/.txt/.md), paste the job ad, click **Tailor my resume**.

Terminal:
```bash
python baseline_tailor.py --resume my_resume.pdf --jd job.txt
```
Saves `tailored_resume.md`.

---

## How the code is organised

```
baseline_tailor.py      the baseline: one prompt, no tools (also used by workflows/baseline.py)
app.py                  simple web page: upload resume + paste JD -> tailored resume
config.py               every setting (model, temperature, context length, step budget). Same for all workflows.
common/llm.py           the ONLY way to call the model. Logs tokens, time and tool calls into a Tracker.
common/agent.py         shared LangGraph ReAct loop: agent -> tools -> agent ... -> submit_resume
common/prompts.py       BASE_RULES + task prompt shared by all workflows
common/data.py          loads data/pairs.csv (resume + JD + hand-labelled keywords)
workflows/              one file per workflow, registered in workflows/__init__.py
eval/metrics.py         the shared scorecard
run_all.py              runs every workflow on every pair, appends to results/results.jsonl
summarise.py            mean ± SD of each metric per workflow -> results/summary.csv
```

## Building your workflow (owners)

1. Copy `workflows/w4_ats_optimiser.py` to `workflows/w1_research.py` (or yours).
2. Change `NAME`, the `SYSTEM` prompt (keep `BASE_RULES +` at the start) and the tools.
3. A tool = a normal Python function + `Tool(fn, description, params)`. Write the description
   carefully: the model reads it to decide when to call the tool.
4. Always call the model through `common.llm.chat(...)`, never `ollama` directly, or your cost won't be counted.
5. Return `{"resume_md": ..., "hit_budget": ..., "finished": ...}`.
6. Add it to `workflows/__init__.py`, then test: `python run_all.py --workflows w1_research --limit 1 --repeats 1`.

Workflows that aren't a tool loop (e.g. W2 recruiter, W5 plan-first) can be their own small LangGraph
graph like `workflows/baseline.py`, with a fixed number of review rounds.

### W6: Best-of-N

W6 generates strategy-specific drafts, validates their structure and claims with the shared NLI
faithfulness checker, scores valid drafts with the same rubric in both candidate orders, and selects
the highest-scoring valid draft.

Quick test on the first two resume–JD pairs:

```bash
python run_all.py --workflows w6_best_of_n --limit 2 --repeats 1
```

Evaluate all 100 pairs once:

```bash
python run_all.py --workflows w6_best_of_n --repeats 1
```

Evaluate all 100 pairs twice for the final consistency comparison:

```bash
python run_all.py --workflows w6_best_of_n --repeats 2
```

| Command setting | Result rows when starting fresh | When to use it |
|---|---:|---|
| `--limit 2 --repeats 1` | 2 | Quick setup check |
| `--repeats 1` | 100 | One complete W6 pass |
| `--repeats 2` | 200 | Two complete passes for the final experiment |

`--repeats N` is the total desired number of repeats, not the number of additional repeats. The
runner saves after every pair and skips existing `(workflow, pair, repeat)` results, so rerunning the
same command safely continues an interrupted evaluation. Do not use `--skip-faithfulness` for the
final experiment because faithfulness is part of the shared scorecard.

On the development machine, one W6 pair took about two minutes. Allow roughly 3 hours 20 minutes
for 100 pairs once, or 6 hours 40 minutes for two repeats; actual time depends on the hardware and
resume length. Keep Ollama running and prevent the computer from sleeping. When evaluation finishes:

```bash
python summarise.py
```

Set `W6_NUM_CANDIDATES` in `config.py` from 1 to 6. The default of 3 uses skills-first,
impact-first, and JD-wording-alignment strategies. The selected strategy and candidate scores are
stored under `workflow_details` in `results/results.jsonl`.

## The scorecard (eval/metrics.py)

| Metric | How | Why |
|---|---|---|
| Cost | input/output tokens + seconds from Ollama | fancier isn't free |
| JD similarity | cosine of embeddings, JD vs resume, before and after | does it read like a match? |
| Keyword coverage | % of our hand-labelled must-have keywords present | roughly what an ATS filter checks |
| Faithfulness | NLI model checks each claim against the original resume | catches invented skills/numbers; stops "copy the JD" from winning |
| Tool-call success | valid tool calls / all tool calls, repeated calls, budget hits | can a 4B model run an agent reliably? |

Notes:
- Faithfulness uses a separate model (`cross-encoder/nli-deberta-v3-small`) as an independent referee.
  Check the label order on its Hugging Face model card matches the code (contradiction, entailment, neutral).
- If you add an overall quality judge, use a bigger model than the writer, and check it against ~20 human ratings.

## Data (data/)

- `resumes/*.md`: fictional candidates only. Never real people's resumes.
- `jds/<id>.md`: synthetic job descriptions informed by the linked O*NET occupation profiles.
- `jds/<id>.keywords.json`: 10–15 must-have keyword drafts with synonyms. **Two people must review and reconcile these labels before the final run.**
- `pairs.csv`: 100 resume–JD pairs, with exactly four fictional candidates assigned to each JD.
- `SOURCES.md`: occupation provenance, licensing, and dataset limitations.

Current dataset: 100 fictional resumes × one assigned JD each = 100 pairs across 25 JDs. At six
workflows and two repeats, the complete experiment is 1,200 runs. Regenerate and validate the
synthetic files with `python scripts/generate_synthetic_dataset.py`.

## Rules that keep the comparison fair
- Don't change `config.py` or `BASE_RULES` for your own workflow.
- Run experiments one workflow at a time (`run_all.py` already does this) so timings aren't distorted.
- Freeze all workflows before the full run. No tweaking after seeing test results.
