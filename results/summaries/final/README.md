# Final evaluation (official)

*Criteria fixed on 2026-11-05. Covers all 7 workflows × 100 resume/JD pairs, using the regenerated outputs (main + fixes #14–16, generated 4 Nov). These files are the single source for the report's results.*

## Criteria: what counts as a good resume

A resume counts as **good** when it passes both checks:

1. **Truthful.** NLI claim faithfulness of the generated resume against the original resume is ≥ **0.80**.
2. **Well written.** The rubric v3 overall score is ≥ **4.0**.

**Workflow result = good resumes / 100 attempts.** A failed generation counts as not good.

All per-resume scores are kept, so both thresholds can be changed later without re-running anything. Use `scripts/quality_rubric.py --summary-only --nli-threshold X --rubric-threshold Y`.

### 1. NLI claim faithfulness (official)

- **How it's measured:**
  - Each claim in the generated resume (bullets, sentences, skills) is checked against evidence retrieved from the original resume.
  - The checker is cross-encoder/nli-deberta-v3-small, using entailment probability ≥ 0.5 and the top 3 evidence units per claim.
  - Faithfulness = supported claims / all claims. The job description is not an input.
- **Source:** evaluator `1cd268431e1c38d1`, run on the team machine on 2026-11-05. It covers all 700 resumes, in `results/evaluations/3bb1861e…/2b526ff0b92d/1cd268431e1c38d1/records/`.
  - These are the normal resumes only. The `robustness/` subfolder holds the separate altered-input test.
- **Reproducibility:** the independent team runs gave scores identical to the last decimal for every resume compared.
  - W1 100/100 and W5 100/100 (Kester).
  - W4 100/100 (Ethan), W6 86/86 (Leonard), W7 100/100 (Jaxs).
  - W3 92/92 (Yong Zhi, partial run).
- **Checker noise:** each original resume was scored against itself; a perfect checker would give 1.0.
  - Mean 0.968; lowest 0.88.
  - Only bare skill words such as "TypeScript" were wrongly flagged.
  - A gate at 0.80 therefore never fails a resume because of checker noise.

| Workflow | Generated | Mean | Median | Unsupported claims per resume | Resumes with bait | Pass ≥ 0.80 | ≥ 0.85 | ≥ 0.88 | ≥ 0.90 | ≥ 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| W1 single pass | 100 | 0.732 | 0.750 | 5.63 | 32 | **34** | 21 | 16 | 9 | 3 |
| W2 baseline | 100 | 0.941 | 0.941 | 1.11 | 1 | **98** | 96 | 90 | 84 | 45 |
| W3 critic | 92 | 0.880 | 0.886 | 2.45 | 10 | **79** | 69 | 50 | 34 | 22 |
| W4 evidence | 100 | 0.963 | 0.955 | 0.69 | 0 | **100** | 100 | 97 | 95 | 65 |
| W5 fine-tuned | 100 | 0.947 | 0.950 | 1.00 | 0 | **99** | 97 | 97 | 79 | 58 |
| W6 best-of-N | 86 | 0.929 | 0.941 | 1.35 | 0 | **85** | 83 | 73 | 59 | 32 |
| W7 plan-execute | 100 | 0.931 | 0.923 | 0.86 | 1 | **94** | 91 | 84 | 79 | 34 |

Pass counts are out of 100 attempts. W3 and W6 failed to generate 8 and 14 resumes respectively.

**Head-to-head vs W2 on the same pairs.** A pair counts as more or less faithful when the difference exceeds 0.05; *p* is from a two-sided sign test.

| Workflow | More / less faithful than W2 | Tie | Mean difference | *p* |
|---|---|---|---|---|
| W4 | 17 / 1 | 82 | +0.021 | <0.001 |
| W5 | 12 / 11 | 77 | +0.006 | 1.0 |
| W7 | 13 / 18 | 69 | −0.010 | 0.47 |
| W6 | 7 / 28 | 51 | −0.015 | <0.001 |
| W3 | 6 / 48 | 38 | −0.061 | <0.001 |
| W1 | 1 / 94 | 5 | −0.209 | <0.001 |

### 2. Rubric v3: writing quality (completed 2026-11-05)

- **Judge:** gpt-oss-120b on DeepInfra, temperature 0, seed 42, medium reasoning.
- **Truth is not judged here,** with one exception: only numbers that appear in the original resume earn credit.
- **Each score must be backed by verbatim quotes,** which are verified. The judge gets up to 3 attempts.
- **Code:** `eval/quality_rubric.py` and `scripts/quality_rubric.py`.

Each dimension is scored as an integer 1–5 by counting problems: 5 = none, 4 = one, 3 = two or three, 2 = four or five, 1 = more than five or pervasive.

| Dimension | What is counted |
|---|---|
| Quantified evidence | Numbered achievements from the original that are dropped or blurred. 1 = none kept. Only numbers present in the original count. |
| Clarity | Vague bullets (no clear action, skill or result) and run-on bullets that merge several achievements |
| Concision | Repeated bullets or claims, a summary that restates the bullets, filler phrases, sentences that add nothing |
| Logical order | Items in the wrong section, related items scattered, the key item buried, a section that breaks the flow. Heading names, bullet style and a sensible alternative section order are not penalised. |

**Overall score** = equal-weight mean of the 4 dimensions.

**Run:** completed 2026-11-05 at 23:39. 64 resumes came back unscored after 3 attempts (the judge failed the quote check) and were re-judged with up to 5 further attempts. Afterwards only 3 of 663 generated resumes remained unscored (W1 2, W6 1). In total the run made 1,149 judge calls (1.9M input and 2.3M output tokens), roughly $0.50–1.15. Identity: `rubric_v3_identity.json` (hash `8337e868a64a4c4d`).

| Workflow | Overall | Quantified evidence | Clarity | Concision | Logical order | Scored |
|---|---|---|---|---|---|---|
| W5 fine-tuned | **4.40** | **4.69** | 3.19 | 4.72 | 5.00 | 100/100 |
| W2 baseline | 4.30 | 4.20 | 3.15 | **4.92** | 4.95 | 100/100 |
| W6 best-of-N | 4.29 | 4.09 | 3.36 | 4.69 | 4.99 | 85/86 |
| W7 plan-execute | 4.19 | **3.09** | **3.90** | 4.78 | 5.00 | 100/100 |
| W4 evidence | 4.14 | 3.84 | 3.31 | 4.47 | 4.93 | 100/100 |
| W1 single pass | 4.06 | 4.08 | 3.03 | 4.15 | 4.98 | 98/100 |
| W3 critic | 4.01 | 4.27 | **2.73** | 4.12 | 4.92 | 92/92 |

The logical order dimension does not discriminate (4.92–5.00 for every workflow) because all workflows share one section template. Clarity is the hardest dimension for every workflow.

## Result: good resumes per workflow (NLI ≥ 0.80 and rubric overall ≥ 4.0)

| Rank | Workflow | NLI pass | Rubric pass | **Good / 100** |
|---|---|---|---|---|
| 1 | **W5 fine-tuned** | 99 | 95 | **94** |
| 2 | **W2 baseline** | 98 | 90 | **88** |
| 3 | W7 plan-execute | 94 | 86 | **83** |
| 4 | W4 evidence | 100 | 77 | **77** |
| 5 | W6 best-of-N | 85 | 69 | **69** |
| 6 | W3 critic | 79 | 60 | **54** |
| 7 | W1 single pass | 34 | 65 | **29** |

**Reading it:**
- **W5 wins.** It is nearly as truthful as W4 and keeps the most real numbers, at the lowest cost (1 call, 3 s per resume).
- **W2 is second.**
- **W4 is the most truthful** (100/100 pass NLI) but loses on writing: blurred numbers and repetition, such as a summary that restates the bullets.
- **W7 writes the clearest bullets** but drops the most numbers.
- **W1 is eliminated by the truth gate:** only 34 of its resumes pass NLI.

**Re-judging the unscored resumes mattered.** Counting them as not good would have understated W5 by 14 and W2 by 10 (first-pass counts: W5 80, W2 78, W7 77, W4 68, W6 63, W3 52, W1 28).

Files:
- `rubric_v3_and_good_per_resume.csv`: per resume, NLI fields, the 4 rubric scores, overall, and the pass and good flags.
- `good_resume_summary.json`: the counts and thresholds used.

## Supplementary (not part of the good-resume count)

- **Operational cost:** time, model calls and tokens per resume, in `results/results.jsonl` and `results/runs/1.0.0/<config>/<workflow>/<pair>/repeat_0/metrics.json`.

## Superseded (kept for history in the team knowledge base; do not use in the report's results)

- **Rubric v1** (Groq, 30 pairs). It judged truth as well as quality and its scores bunched at 4–5. Replaced by rubric v3.
- **Requirement coverage** (gpt-oss-20b). Dropped from the criteria.
- **Pairwise vs W2** (DeepInfra, 6 × 100 pairs). Dropped from the criteria. It can still be cited as the reason quality judges need a truth gate: the judge preferred W1, which fabricates, 13–2.
- **The pairwise screening and the earlier ranking** built on rubric v1 and pairwise.
- **Input robustness** (700 runs on altered inputs, 2026-11-05). It is not part of the final method, and its stability score used the old checks.
