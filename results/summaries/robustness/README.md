# Input robustness results (robustness-variants-v2)

*Run 2026-11-05 on the team machine (RTX 4070, Ollama, local GPU only, no hosted judge). 700 generations: 7 workflows × 10 pairs × 10 inputs (original + 9 variants). Evaluator hash `1cd268431e1c38d1`, robustness identity `3e0667eedc9494da`, dataset hash `3bb1861e…`.*

## What is being evaluated

Does each workflow keep its quality and stay honest when the input is messy or wrong, instead of the clean template data it was built on?

- **Sample:** 10 resume/JD pairs from a stratified, seeded (42) sample covering all 5 categories and strong/moderate/weak fit (`sample_manifest.json`).
- **Variants:** seeded, so every workflow sees identical perturbations. All 630 variant runs were generated with the same workflow code as the saved originals (`generation_identity_matches_original = true`).

| Family | Variant | What changes |
|---|---|---|
| Noisy (in scope) | `typos` | Low-rate typos in resume and JD prose; never in names, dates, keywords or capitalised words |
| | `pdf_extraction` | Resume as PDF-extracted text: no Markdown, hard line wraps, hyphenation splits, page break with page numbers and a running header |
| | `web_clutter` | JD wrapped in careers-page boilerplate: nav, cookie banner, benefits, EEO text, footer |
| Format (in scope) | `formatting` | `*` bullets, upper-case headings |
| | `reordered_sections` | Resume sections in a different order |
| | `plain_text` | Markdown stripped from resume and JD |
| | `heading_aliases` | Headings renamed (e.g. Experience → Work History) |
| Out of scope | `mismatched_jd` | JD from a different job category |
| | `degenerate_jd` | Title only, a one-line posting, or non-job text |

**In-scope metrics** (facts unchanged; outputs scored against the *clean* resume and JD):
- **Kept running** (`execution_retention`): completed variants / completed originals.
- **Unchanged** (`stability_score`): no binary metric (task success, protected-field violation, fabrication bait) flips worse and no [0,1] metric (faithfulness, unsupported-claim rate, keyword coverage, similarity) drops by more than 0.05.
- **Mean paired deltas:** variant minus original.

**Out-of-scope metric** (`expected_behaviour_pass`):
- Pass: stop, or refuse.
- Pass: complete without fabrication (no bait, faithfulness ≥ 0.8, protected fields intact).
- Fail: crash.

## Originals (the 10 sampled pairs, no perturbation)

| | W1 | W2 | W3 | W4 | W5 | W6 | W7 |
|---|---|---|---|---|---|---|---|
| Completed /10 | 10 | 10 | 9 | 10 | 10 | 9 | 10 |
| Mean faithfulness | 0.723 | 0.937 | 0.901 | 0.970 | 0.946 | 0.945 | 0.953 |
| Task success /10 | 2 | 9 | 9 | 8 | 8 | 3 | 1 |

## In scope: noisy (30 runs each) and format (40 runs each)

| Workflow | Kept running (noisy / format) | Unchanged (noisy / format) | Faithfulness Δ (noisy / format) | Task success Δ (noisy / format) | Handled errors |
|---|---|---|---|---|---|
| W1 single pass | 100% / 100% | 0.80 / 0.58 | +0.054 / +0.029 | +0.20 / +0.13 | 0 |
| W2 baseline | 100% / 100% | 0.83 / 0.60 | +0.006 / +0.005 | 0.00 / −0.30 | 0 |
| W3 critic | 96% / 94% | 0.77 / 0.53 | +0.030 / −0.022 | −0.12 / −0.24 | 3 |
| W4 evidence | 100% / **55%** | 0.67 / 0.82 | −0.007 / −0.002 | −0.27 / −0.09 | 18 |
| W5 fine-tuned | 100% / 100% | 0.57 / **0.23** | −0.003 / −0.005 | −0.40 / **−0.68** | 0 |
| W6 best-of-N | 89% / 83% | 0.83 / 0.70 | −0.004 / −0.005 | +0.17 / +0.20 | 9 |
| W7 plan-execute | 100% / 100% | 0.50 / 0.63 | −0.017 / −0.013 | −0.03 / +0.03 | 0 |

Retention below 100% by variant:

| Workflow | Variant | Retention |
|---|---|---|
| W4 | reordered_sections | **0%** (10 handled errors) |
| W4 | heading_aliases | **20%** (8 handled errors) |
| W3 | plain_text | 78% |
| W3 | typos | 89% |
| W6 | formatting, reordered_sections, web_clutter | 78% each |
| W6 | typos, plain_text, heading_aliases | 89% each |

**Crashes:** 0 of 700 runs.

## Out of scope: mismatched and degenerate JDs (10 runs each per workflow)

| Workflow | Completed / stopped (of 20) | Expected behaviour (mismatched / degenerate) | Fabrication rate when completed (mismatched / degenerate) |
|---|---|---|---|
| W1 | 20 / 0 | 0.6 / 1.0 | **40%** / 0% |
| W2 | 20 / 0 | 1.0 / 1.0 | 0% / 0% |
| W3 | 18 / 2 | 0.9 / 1.0 | 11% / 0% |
| W4 | 20 / 0 | 1.0 / 1.0 | 0% / 0% |
| W5 | 20 / 0 | 0.9 / 1.0 | 10% / 0% |
| W6 | 12 / 8 | 0.8 / 1.0 | 33% / 0% |
| W7 | 20 / 0 | 0.9 / 0.8 | 10% / 10% |

- No workflow produced an explicit refusal.
- Pooled expected-behaviour pass rate: W2 and W4 1.00, W3 and W5 0.95, W6 0.90, W7 0.85, W1 0.80.

## Findings

1. **Faithfulness survives messy input.** Across noise and format changes, mean faithfulness moves by at most about 0.05 for every workflow. The grounding guardrails hold.
2. **W4 is brittle to resume layout.**
   - It fails 10/10 when sections are reordered and 8/10 when headings are renamed.
   - It is unaffected by typos, PDF artefacts or web clutter.
   - Its span-splitting step depends on the standard section structure. This is the cost behind its best-in-class faithfulness (0.970 on these originals).
3. **W5's fine-tuning does not generalise across formats.**
   - Under format changes it has the lowest stability (0.23) and the largest task-success drop (−0.68).
   - Faithfulness is unchanged, so the drops come from structure and length checks failing.
   - Training data used one input format; this is a fine-tuning limitation to discuss.
4. **W2 and W4 handle impossible inputs best:** no fabrication on mismatched or degenerate JDs.
   - W1 (no guardrails) fabricates on 40% of mismatched JDs.
   - W6 stops on 8/20 out-of-scope inputs, but fabricates on a third of the mismatched JDs it does complete.
5. **W6 and W3 are the only workflows that ever stop** on altered inputs, through their self-rejection paths. None crash.

## Caveats

- **Small n:** 10 pairs, so each cell covers 10–40 runs. Read the results as patterns; Wilson 95% CIs are in `reports/robustness_families.csv`.
- **Original is the saved generation, not a re-run.** "Unchanged" therefore also includes ordinary run-to-run variation, which inflates instability most for stochastic workflows (W5, W7). A repeat-of-original control would separate the two.
- **Positive task-success deltas** (W1, W6) mean the variant run happened to pass checks the original failed. This is run-to-run noise, not a benefit of noisy input.
- **Not run on variants:** requirement coverage and the rubric judge (no Groq).

## Files in this folder

| File | Contents |
|---|---|
| `robustness_families.csv` | In-scope stats per workflow × family (retention, stability with CI, mean deltas, instability reasons) |
| `robustness_variants.csv` | Same, per variant |
| `robustness_out_of_scope*.csv` | Out-of-scope expected behaviour per workflow (× variant) |
| `robustness_deltas.csv` | Paired deltas per workflow × variant × metric, with eligible/scored/unscored counts |
| `robustness_status.csv`, `robustness.csv` | Per-variant metric summaries |
| `sample_manifest.json`, `identity.json` | Which pairs and variants; perturbation identity |

The full run (700 scored records, every input variant, output resume and trace) is kept locally under the git-ignored `results/evaluations/3bb1861e…/2b526ff0b92d/1cd268431e1c38d1/robustness/3e0667eedc9494da/`. Reproduce it with `evaluate.py robustness` on the seven current config hashes with `--robustness-limit 10 --skip-judge --skip-coverage --skip-pairwise`.
