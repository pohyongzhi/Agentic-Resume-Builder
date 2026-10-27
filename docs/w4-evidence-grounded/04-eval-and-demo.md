# W4: evaluation, demo and deep-dive notes

This file is for the team and the presentation, not the coding agent.

## 1. How W4 feeds the shared evaluation

### Workflow metrics (recorded by every workflow)

| Brief item | Where W4 records it |
|---|---|
| Valid rendered resume, or the error that ended it | `run()` → `finished`, `errors` (`W4_DRAFT_FAILED`, `W4_FINAL_VALIDATION_FAILED`, `W4_NO_SPANS`, `W4_FAILED`) |
| Start/end timestamps per node and the whole run | `type="node"` events (`start`, `end`, `seconds`); whole run = `wall_seconds` in `run_all.py` |
| Token counts and number of LLM calls | `Tracker.totals()` (all calls go through `common.llm.chat`) |
| Every tool call and its result | `validate_resume` / `render_resume` `type="tool"` events |
| Variant-specific diagnostics | `type="w4_round"` and `type="w4_final"` events, plus `evidence` |

### W4-specific diagnostics (in the trace)

| Metric | From | What it tells you |
|---|---|---|
| First-draft citation coverage | `w4_final.first_draft_citation_coverage` | Can a 4B model cite its sources at all? |
| First-draft supported rate | `w4_final.first_draft_supported_rate` | How honest is the model before any checking? |
| Rejections by type | `w4_round.rejected_by_code` | *How* it overclaims: invented numbers, JD tools slipped in, inflation, or meaning drift |
| Claims fixed vs dropped | `evidence[].rounds_to_pass`, `w4_final.dropped` | Does feedback fix claims, or do we end up deleting them? |
| Rounds used, extra LLM calls | `w4_final.rounds_used`, tracker totals | The cost of grounding |
| Verifier cache hits | `w4_round.llm_cached` | How much re-checking the cache saved |

### Shared evaluator metrics where W4 should stand out

| Evaluator metric | Expected W4 effect | Why |
|---|---|---|
| Unsupported-claim rate | **Lowest of all variants** | Unsupported claims are dropped before rendering |
| Fabrication-bait rate | **Near zero** | `new_skill` blocks JD tools that aren't in the cited source |
| Protected-field violations | Same as W2 (zero) | Same harness |
| Supportable-requirement coverage | Similar to W2, maybe slightly lower | Dropping claims can remove a truthful-but-badly-cited match |
| Keyword overlap / embedding gain | **Lower than W2/W6** | Honest resumes can't stuff missing JD keywords. This is the trade-off to discuss |
| Rubric quality / pairwise win rate | Possibly lower | Grounded bullets stay closer to the original wording and can read less punchy |
| Tokens / latency | Higher than W2 (≈ 2–4 extra LLM calls) | Verifier call per round + revisions |

**Headline to test:** *"Forcing citations nearly eliminates fabrication, at the cost of some
keyword match and extra tokens."*

**Fairness note for the team:** the evaluator's unsupported-claim judge must not be W4's verifier
(the same Qwen 4B with the same prompt). Otherwise W4 is graded by its own checker. Use a different
or larger judge model, and the hand-rated agreement check, for the evaluator.

## 2. Small experiment worth adding (verifier accuracy)

Measure how good the W4 checks themselves are, which makes a strong deep-dive slide:

1. Take 30 real claims from W2 outputs. Hand-label each as supported or unsupported against the
   original resume (two people, then resolve disagreements).
2. Add 10 planted fakes: an invented %, a JD tool, "led", an invented outcome.
3. Run `rule_check` + `llm_verify` on all 40 and report:
   - **catch rate** (unsupported caught ÷ all unsupported)
   - **false-rejection rate** (supported but rejected ÷ all supported)
   - the split between what code caught and what the LLM caught

## 3. Demo script (2 minutes)

1. Pick W4 in the app. Upload a resume and paste a JD that asks for things the candidate doesn't
   have (e.g. Power BI, A/B testing).
2. Show the tailored resume: clean, no citation markers.
3. Scroll to the **evidence report**: every bullet sits next to the exact line of the original
   resume it came from.
4. Point at a ❌ row, e.g. *"Led a team that cut processing time by 40%"* → `inflated: led`,
   `new_number: 40%` → dropped. *"The model tried to oversell; the harness didn't let it through."*
5. Optional: run W2 on the same input and show the same overclaim surviving there.

## 4. Deep-dive Q&A (be ready to explain)

**Why cite spans instead of asking another LLM "is this true?" (W3)?**
A critic model can be lenient or miss things, especially a 4B model checking its own kind of writing.
A citation turns "trust me" into "here's my evidence", and code can check that the evidence exists,
belongs to the right job, and contains the numbers and tools claimed. The LLM only handles meaning,
which code can't.

**Why split the source into spans in code rather than letting the model quote text?**
Quotes from a small model are often slightly altered or invented. IDs can only point at real text,
and span boundaries stay fixed across rounds, so verdicts can be cached.

**Why cite the raw resume text, not the parser's structured output?**
The parser is also an LLM and can reword things. Citing the raw text grounds every claim in what
the candidate actually wrote.

**Why do rules run before the LLM verifier?**
They're free, deterministic and explainable: invented numbers, JD tools and inflation words are
pattern problems. Running them first means the LLM only sees claims that already passed, which
saves tokens and makes the result less dependent on a small model's judgement.

**Why drop failing claims instead of failing the whole resume?**
One bad bullet shouldn't cost the user a resume. Dropping is fail-safe: the output can only lose
unsupported content, never gain it. `hit_budget` and `dropped` record that it happened.

**Why only 2 revision rounds?**
Topic 7's step budget: each round costs a creator call plus a verifier call. After two rounds,
a claim the model still can't support is usually unsupportable, so more rounds mostly add cost.
(This is worth checking with the data, comparing claims fixed in round 1 vs round 2.)

**Why not use the evaluator's NLI faithfulness model as the verifier?**
That model is the referee for every workflow. If W4 optimised against it directly, W4's
faithfulness score would be inflated by construction rather than earned.

**Why aren't skills and dates cited?**
They're protected fields. The shared harness already checks them exactly against the source, which
is stronger than a citation.

**What are W4's limits?**
- The verifier is the same small model, so subtle meaning drift can get through (the experiment in
  §2 measures this).
- `new_skill` uses capitalisation heuristics and can miss lower-case tools or flag harmless proper nouns.
- Grounding stops fabrication but doesn't make writing better. W4 can be less tailored than W2 or W6.
- A citation proves the claim is supported, not that it's the most relevant evidence for the JD.

**What would be next?**
Combine W4's verification with W7's planner (plan → write with citations → verify), or use the
evidence report to show the user which JD requirements they genuinely can't claim (a gap report).
