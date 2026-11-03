# Evaluation metrics integration handover

## Resume here

- Branch: `feature/eval-metrics-integration`
- Worktree: `/tmp/arb-eval-integration`
- Base: `origin/main` at `a866c34`
- Global context: `/mnt/c/Users/keste/Downloads/CLAUDE.md`
- Status: the tier-1 evals are wired, tested and runnable end to end through `evaluate.py`.
  No model-backed pilot or full run has been made. Nothing has been pushed.

```bash
cd /tmp/arb-eval-integration
git status --short --branch
git log --oneline --decorate -10
sed -n '1,400p' docs/evaluation/workstreams/integration/HANDOVER.md
sed -n '1,200p' docs/evaluation/workstreams/integration/CHECKPOINTS.md
PYTHONDONTWRITEBYTECODE=1 /tmp/arb-eval-venv/bin/pytest -q -p no:cacheprovider
```

## Priorities

| Tier | Eval | State on this branch |
| --- | --- | --- |
| 1 | Faithfulness (NLI, protected fields, fabrication bait) | On main; part of `score` |
| 1 | Supportable-requirement coverage | Wired per output in `score` |
| 1 | Pairwise vs `w2_baseline` with position swap | Wired as the `pairwise` stage (in `all`) |
| 1 | Operational metrics | On main; part of `score` |
| 2 | Rubric judge integration checks | Not started; the judge on main is unchanged |
| 2 | Keyword/similarity reporting | Not started; existing rows in `overall.csv` unchanged |
| 3 | ATS proxy | Code merged (`eval/ats.py`), **not wired** into CLI or reports |
| Dropped | Human agreement | Code merged (`eval/human_review.py`), **not wired** |
| Optional | Prompt injection | Wired and tested as the `injection` stage; **not in `all`** |
| Robustness | Input robustness v2 (noisy, format, out-of-scope variants) | Wired as the `robustness` stage (in `all`); not run |

## Merged branches

| Workstream | Branch / head | Merge commit | Note |
| --- | --- | --- | --- |
| Requirement coverage and injection | `feature/eval-coverage-injection` / `42a4668` | `7bbe174` | |
| Pairwise and position swapping | `feature/eval-pairwise-position` / `174463c` | `71d000d` | |
| Human agreement and ATS | `feature/eval-human-ats` / `3c667f8` | `e099d25` | Clean merge, all its tests pass; code kept, not wired |

Integration commits: `84809af`, `f63cba0`, `95f7b4e` (docs), `cea794e` (tier-1 focus), `c8a3cdc`
(docs), `205e3af` (robustness v2), plus the commit containing this file.

## Required behaviour and how it is met

- Evaluation failures stay unscored: each metric writes `None` plus a reason in
  `evaluation_failures[<metric>]`; reports count them as `unscored`/`failed`, never as zero.
- Cache identity: per-output components add `identity()` to
  `evaluator_identity.json["output_components"]`, so they are part of the evaluator hash (the
  results folder). Each component's cache key also hashes its inputs. Pairwise records live under
  `pairwise/<pairwise evaluator_hash>/`; its cache key hashes the four texts, both workflows and
  config hashes, the full judge config, model provider/tag/digest and prompt/schema/rubric hashes.
- Generation artifacts and the frozen dataset are untouched (`git diff origin/main -- data results`
  is empty; `validate_dataset()` still returns `3bb1861e…6f81`).
- Explicit `eligible`/`scored`/`unscored`/`failed` denominators in every new table.
- No network in tests: fake clients and workflows; tier-1 and pairwise tests block sockets.
- Thresholds and historical results unchanged. `EVALUATOR_SCHEMA_VERSION` moved `2.2 -> 2.3`
  because records gained fields; older results folders stay as historical artifacts.

## Tier-1 CLI

Run from the repository root that holds `results/results.jsonl` and the saved run files.

Options that matter for tier 1:

- `--pair-ids P1 P2 ...` / `--pair-ids-file FILE` (one ID per line, `#` comments allowed): score
  and pairwise-judge only those dataset pairs. Results go to the same folder as a full run;
  later subsets add to the indexes and reports instead of replacing them, and completed records
  are never re-judged. Use this for the Groq free tier.
- `--config-hash WORKFLOW=HASH` (repeatable): keep only saved generations with that config hash.
  Required for `w3_critic` (see blockers) and also applied to pairwise pairing.
- `--skip-judge`: skips the tier-2 rubric judge. Coverage and pairwise still use the
  `--judge-provider`/`--judge-model` model (coverage can be overridden with
  `--coverage-provider`/`--coverage-model`).
- `--skip-robustness`: robustness re-generates perturbed inputs with the workflows (Ollama);
  see "Input robustness (v2)" below.
- `--pairwise-baseline` (default `w2_baseline`), `--pairwise-challengers`, `--pairwise-limit N`
  (first N pair IDs in sorted order), `--limit N` (first N saved rows for `score`).

The `--skip-judge`, `--skip-coverage`, judge and coverage model options change the evaluator
hash, so keep them identical across `score`, `pairwise` and `report` calls for one results folder.
The selected `--workflows` (and `--config-hash`) define the generation scope in the folder path,
so always pass `w2_baseline` together with the challengers.

One challenger workflow on a fixed subset (Groq):

```bash
export GROQ_API_KEY=...
python evaluate.py all --workflows w2_baseline w1_single_pass \
  --judge-provider groq --judge-model openai/gpt-oss-120b \
  --skip-judge --skip-robustness \
  --pair-ids bt_resume_001__bt_jd_01 bt_resume_002__bt_jd_01 cd_resume_002__cd_jd_01
```

All workflows:

```bash
python evaluate.py all --workflows w2_baseline w1_single_pass w3_critic w6_best_of_n \
  --config-hash w3_critic=b884fe8fe066 \
  --judge-provider groq --judge-model openai/gpt-oss-120b \
  --skip-judge --skip-robustness \
  [--pair-ids-file pilot_pairs.txt]
```

Stage by stage (same flags): `score` (faithfulness, coverage, operational), `pairwise`, then
`report` to rebuild `summaries/` from saved indexes without any model call.

Without `GROQ_API_KEY`, use the default Ollama judge (`--judge-provider ollama --judge-model
gemma4:e4b`). Faithfulness needs `sentence-transformers` and the NLI model, or
`--skip-faithfulness`.

### Expected hosted calls

`JUDGE_MAX_RETRIES = 2`, so one logical call is at most three requests.

- Coverage, per output: **0 to 2 calls**, usually 0-1. Requirements with an exact keyword or
  synonym match are decided lexically (no call). A supportability call runs only for requirement
  families with no lexical evidence in the original resume, and it is cached per original
  resume and requirement set, so it is shared by every workflow output for that pair. A coverage
  call runs only when a supportable requirement is not lexically present in the output.
  Worst case 6 requests with retries.
- Pairwise, per challenger output: **exactly 2 calls** (challenger first, then baseline first).
  The baseline output itself costs nothing extra. Worst case 6 requests with retries.
  Generation failures and missing outputs cost 0 calls.
- Rubric judge (tier 2, skipped with `--skip-judge`): 1 call per output, worst case 3.
- All hosted calls share one process-wide timer (`eval.judge.wait_for_hosted_request_slot`) with
  `JUDGE_MIN_REQUEST_INTERVAL_SECONDS = 15`. Example: 3 pairs x (w2 + one challenger) is about
  6 outputs x 0-2 coverage calls + 3 x 2 pairwise calls = 6-18 calls, about 1.5-4.5 minutes.
  All 100 pairs x 3 challengers is at least 600 pairwise calls (about 2.5 hours) plus coverage.

## Outputs

Under `results/evaluations/<dataset_hash>/<generation_scope>/<evaluator_hash>/`:

- `records/<workflow>/<pair_id>/repeat_<n>.json`, `evaluations.jsonl`: per-output rows with
  faithfulness, coverage and operational fields.
- `cache/requirement_coverage/`, `cache/nli/`: evaluator caches.
- `pairwise/<pairwise_hash>/records/<challenger>/<pair_id>/repeat_<n>.json`,
  `pairwise/<pairwise_hash>/evaluations.jsonl` (compact: no prompts or raw attempts),
  `pairwise_cache/` (full records).
- `summaries/`: `overall`, `by_category`, `by_fit_level` (existing, now with a
  `requirement_coverage` row), `requirement_coverage`, `requirement_coverage_by_fit_level`,
  `pairwise` (by baseline/challenger) and `pairwise_by_fit_level`, each as CSV and JSON.

## Tier-1 metric semantics

### Requirement coverage

Record fields: `requirement_coverage`, `requirement_coverage_status` (`scored`, `unscored`,
`no_supportable_requirements`, `generation_failed`, `evaluation_error`), supportable/covered
counts, uncovered list, evaluator version/hash, seconds, and the full `requirement_coverage_record`.

- `no_supportable_requirements` is not applicable: excluded from `eligible`, not a failure.
- `evaluation_error` (evaluator unreachable or rate limited) is not cached and is retried on the
  next run. Invalid evaluator output is cached and final for that identity, like the judge.
- The report gives macro mean with bootstrap CI and micro coverage (covered / supportable).
- Limitation: JD keyword files are requirement-family proxies, so this measures the labelled
  must-have families, not every prose requirement.

### Pairwise vs W2

- Each comparison is judged twice with positions swapped; a winner exists only when both orders
  agree. Disagreement is reported as position-swap inconsistency.
- Resume: `scored` and `unscored`+`invalid_output` are final; `judge_failed` is retried;
  `generation_unavailable` (missing or failed output on either side) is recomputed without a call.
- Report columns: `total_pairs`, `generation_unavailable`, `eligible`, `scored`, `unscored`,
  `failed`, `invalid`, strict win/loss rates with Wilson CIs, position-neutral score, swap
  consistency and first-position selection rate.

### Faithfulness and operational

Unchanged from main; computed in `score` for every saved output.

## Input robustness (v2)

Robustness = how reliably a workflow keeps its performance, accuracy and expected behaviour when
its inputs are unexpected, noisy, out of scope or changing. Prompt injection is separate (the
`injection` stage, unchanged).

### Variants

Every variant is a pure, seeded function of `(pair, config.ROBUSTNESS_SEED)` in
`eval/robustness.py` (`build_variant`); `mismatched_jd` also reads the frozen dataset's JD pool
and its choice is recorded in the manifest.

| Family | Variant | What changes |
| --- | --- | --- |
| baseline | `original` | Nothing; reuses the saved repeat-0 generation (no new generation) |
| noisy | `typos` | ~3% (`ROBUSTNESS_TYPO_RATE`) of eligible prose words in resume and JD get one keyboard edit (swap, drop, double, neighbour key). Only lowercase words of 5+ letters on resume claim lines / JD prose lines; never protected-field tokens (name, contact, titles, employers, dates, degrees, institutions), keyword/synonym tokens, the JD title or capitalised words |
| noisy | `pdf_extraction` | Resume as PDF text: Markdown removed, hard wraps at a seeded 56-72 columns, hyphenation splits of long unprotected words, a page break with `Page n of 2` and a running header, irregular spaces |
| noisy | `web_clutter` | JD wrapped in scraped careers-page text: skip link, nav, cookie banner, breadcrumb, apply/share, benefits, EEO, job ID, footer. Blocks that mention a JD keyword family are dropped |
| format | `formatting` | v1: `*` bullets, upper-case headings, blank lines |
| format | `reordered_sections` | v1: sections reversed |
| format | `plain_text` | Markdown removed from resume and JD (`•` bullets, upper-case section labels) |
| format | `heading_aliases` | Section headings renamed (Experience -> Work History, Skills -> Core Competencies, Responsibilities -> Your Impact, ...) |
| out_of_scope | `mismatched_jd` | Resume unchanged; JD from a different category (seeded category, then seeded JD) |
| out_of_scope | `degenerate_jd` | Resume unchanged; JD is one seeded mode per pair: `title_only`, `very_short` ("Hiring now: <title>. Apply today.") or `non_job_text` |

Why this set (9 generated variants): it covers each part of the definition once without
duplication. Typos, PDF extraction and web clutter are the noise the product actually receives
(uploaded PDFs, URL-fetched JDs); plain text and heading aliases are the format changes real
resumes show; the two v1 variants are kept for continuity. Out of scope is one wrong-but-valid
input (mismatched JD) and one invalid input (degenerate JD, three modes rotated by seed rather
than three variants, to keep cost down). Not included: synonym/paraphrase rewrites (they need a
model and cannot be checked for content preservation deterministically), translation, and
multi-perturbation stacks.

Content preservation (`variant_content_problems`, run by `build_variant`, tested over all 100
dataset pairs): identical case-insensitive token multisets for `formatting`,
`reordered_sections`, `plain_text`; only level-2 heading lines change for `heading_aliases`;
for `typos` same lines and word counts, every change is a <=2-edit typo of an unprotected
lowercase word, protected-field values parse identically and keyword families are unchanged;
for `pdf_extraction` the tokens equal the clean resume after removing page lines/header and
joining hyphenation splits, and every protected token survives; for `web_clutter` the clean JD is
contained verbatim and the clutter mentions no keyword family; out-of-scope variants keep the
resume byte-identical.

### Scoring and expected behaviour

- Scoring reference: faithfulness (NLI), protected fields and fabrication bait are always scored
  against the **clean original resume** (`scoring_reference: clean_original_resume` on every
  record), never the perturbed input. In-scope variants also use the clean JD and keywords for
  keyword coverage, similarity and requirement coverage (`scoring_jd: clean_original_jd`).
  Variant rows no longer inherit the original's saved `kw_after`/`sim_after`/`words`; each output
  is scored on its own text. The workflow input file (`resume_path`) is the perturbed resume, so
  W2/W3/W4, which read the file, see the perturbation (v1 passed the clean file).
- In scope (noisy, format): for each workflow, pair and variant, signed deltas (variant minus
  original) for `task_success`, `faithfulness`, `unsupported_claim_rate`,
  `protected_field_violation`, `fabrication_bait_violation`, `requirement_coverage` (when
  enabled), `kw_after`, `sim_after` and `judge_overall_mean` (when the judge runs). A delta exists
  only when both generations completed and both values are scored.
  - Stability (per run): both generations completed, and no binary criterion (`task_success`,
    protected-field violation, bait violation) flipped to worse and no [0, 1] metric
    (`faithfulness`, `unsupported_claim_rate`, `requirement_coverage`, `kw_after`, `sim_after`)
    degraded by more than `ROBUSTNESS_STABILITY_TOLERANCE = 0.05`. Unscorable criteria are
    skipped; a run with none is unscored. Stability score = stable / scored, with Wilson CI, per
    workflow x variant and pooled per workflow x family.
  - Execution retention = completed variant runs / runs whose original completed. Generation
    failures are counted here and in `failed`, never as a zero quality score.
- Out of scope: expected-behaviour checks, not quality deltas. Each run gets
  `generation_outcome`: `crashed` (uncaught exception from the workflow), `handled_error`
  (workflow-reported error or empty output), `refused` (refusal text with fewer than two
  required sections) or `completed`.
  - `crashed` -> `expected_behaviour_pass = False` (a failure, counted in `failed`).
  - `handled_error` / `refused` -> handled gracefully, `expected_behaviour_pass = True`; not a
    crash and not a zero score.
  - `completed` -> pass iff `oos_no_fabrication` (no fabrication bait against the clean resume and
    the union of the original and presented JD keyword families, and faithfulness >=
    `TASK_FAITHFULNESS_THRESHOLD`, i.e. unsupported claims <= 20%) and
    `oos_protected_fields_preserved`. A failed check fails the run; an unscored check (evaluator
    failure, or `--skip-faithfulness`) leaves it unscored unless another check already failed.
  - Out-of-scope runs skip the rubric judge and requirement coverage (no hosted calls).
- Missing original (for example `w3_critic`'s two missing pairs): recorded as
  `generation_outcome: missing_original`; that pair's variants are not generated.
- Generation identity: variants are generated with the current workflow code. Records carry
  `config_hash` (current), `original_config_hash` and `generation_identity_matches_original`,
  and the stage prints a warning when they differ. While blocker 1 stands, in-scope deltas mix
  code drift with perturbation; either regenerate the originals or read deltas with that caveat.

### Reports (`summaries/`)

- `robustness_variants`: workflow x in-scope variant: `runs`, `original_unavailable`,
  `eligible` (original completed), `failed` (`crashed` + `handled_errors`),
  `execution_retention`, `scored`/`unscored` for stability, `stable`, `stability_score` with CI,
  `instability_reasons`, and `<metric>_mean_delta` / `<metric>_paired_n` per metric.
- `robustness_families`: the same pooled per workflow x family (`noisy`, `format`).
- `robustness_deltas`: long format, workflow x variant x metric: `eligible` (original scored on
  the metric), `scored` (paired delta), `unscored = eligible - scored`, `failed` (variant
  generation failed), `evaluation_failures`, mean delta with bootstrap CI, mean absolute change
  and direction-aware `worst_degradation` (positive = worse).
- `robustness_out_of_scope` / `robustness_out_of_scope_families`: `eligible` (attempted),
  `scored` (determinate pass/fail), `unscored`, `failed` (= `crashed`), `completed`, `refused`,
  `handled_errors`, `graceful_handling_rate`, `expected_behaviour_pass_rate` with CI,
  fabrication and protected-field failure counts/rates, mean unsupported-claim rate.
- `robustness` (legacy paired-delta table, in-scope rows only) and `robustness_status`.

### Versioning and layout

`config.py`: `ROBUSTNESS_SAMPLE_VERSION = "robustness-sample-v2"`,
`ROBUSTNESS_VARIANT_VERSION = "robustness-variants-v2"`, `ROBUSTNESS_TYPO_RATE = 0.03`,
`ROBUSTNESS_STABILITY_TOLERANCE = 0.05`, `ROBUSTNESS_SEED = 42` (unchanged). v2 artifacts live
under `<evaluator root>/robustness/<identity_hash>/` (`identity.json`, `sample_manifest.json`
written before generation, `runs/<workflow>/<pair_id>/<variant>/{input_resume.md|.txt,
input_jd.md, resume.md, trace.json, generation.json, evaluation.json}`, `evaluations.jsonl`).
v1 files directly under `robustness/` and the legacy `run_robustness.py` outputs are untouched
history (`run_robustness.py` still runs only the v1 variants). Any change to a perturbation must
bump `ROBUSTNESS_VARIANT_VERSION`; `generation.json` also stores an input hash, so changed inputs
are regenerated rather than reused.

Sample: the same 20 stratified pairs as v1, reordered so every prefix is spread across
categories (round-robin over categories, strong/moderate/weak alternating within each), so
`--robustness-limit 10` gives 2 pairs per category, all strong or moderate.

### Cost and CLI

Generations = pairs x generated variants x workflows x repeats, where generated variants exclude
`original` (it reuses the saved generation) and repeats = 1 (the stage uses repeat 0 at the
fixed seed/temperature). Each generation is one full workflow run (W6 is several model calls).
All defaults: 20 x 9 x workflows. Recommended small config: 10 pairs x 9 variants x 2 workflows
(`w2_baseline` + one challenger) = 180 generations; per family: noisy 10 x 3 x 2 = 60, format
10 x 4 x 2 = 80, out_of_scope 10 x 2 x 2 = 40. Evaluator cost: in-scope outputs add 0-2 coverage
calls each (cached per clean resume/requirements where possible) plus one local embedding call;
out-of-scope outputs make no hosted calls; NLI is local.

```bash
python evaluate.py robustness --workflows w2_baseline w1_single_pass \
  --skip-judge --robustness-limit 10 \
  [--robustness-families noisy format out_of_scope] \
  [--robustness-variants typos degenerate_jd]
```

Keep `--skip-judge`, judge/coverage model and `--config-hash` flags identical to the `score` run
so the robustness tree sits under the same evaluator root. Variant selections add to the same
index; `report` rebuilds every robustness table without generation.

## Extension points

- Per-output metric: implement `OutputComponent` (`name`, `identity()`, `is_complete(record)`,
  `evaluate(record, generation_row, pair, tailored_resume, root, force)`) in `evaluate.py` and
  return it from `build_output_components`. Identity, checkpoint completeness and evaluation
  pick it up automatically. Tier-3 ATS would plug in here.
- Run-level stage: add a `Stage(name, run, index, pilot_limit, help, loader=None, in_all=True)`
  to `STAGES`; the parser adds the command, `--skip-<name>` and `--<name>-limit`.
- Reports: `eval.reporting.STAGE_REPORTERS[name] = writer(output_dir, rows)`, or add a scalar
  field to `METRICS`.
- Constants: all workstream versions, thresholds and weights are in `config.py` (values unchanged),
  including the unwired ATS and agreement ones.

## Optional, deferred and dropped

- Prompt injection (`python evaluate.py injection --workflows ... --injection-limit N`): working
  and tested, writes its manifest before generation, resumable. Not in `all`. It refuses to run
  until blocker 1 is resolved because it generates with the current workflow code.
- ATS proxy (tier 3): `eval/ats.py` and `tests/test_ats.py` are merged; to wire it, add an
  `OutputComponent` calling `evaluate_ats(OfflineATSProvider(), ...)` and an `ats` report from
  `summarise_ats`.
- Human agreement (dropped): `eval/human_review.py` additions and `tests/test_human_agreement.py`
  are merged but not called. The pre-existing blinded review packet from main is still written
  by `all`/`report`; it is now left untouched once any reviewer label exists.

## Validation

```bash
cd /tmp/arb-eval-integration
PYTHONDONTWRITEBYTECODE=1 /tmp/arb-eval-venv/bin/pytest -q -p no:cacheprovider
# 485 passed, 21 subtests passed in 13.60s (full suite, including tests/test_app.py)
PYTHONDONTWRITEBYTECODE=1 /tmp/arb-eval-venv/bin/pytest -q -p no:cacheprovider tests/test_robustness.py
# 39 passed (every variant on all 100 pairs, protected tokens, seeds, OOS logic, denominators, stage)
PYTHONDONTWRITEBYTECODE=1 /tmp/arb-eval-venv/bin/pytest -q -p no:cacheprovider tests/test_eval_integration.py
# 25 passed (includes the tier-1 end-to-end CLI test on a pair subset)
git diff --check                          # clean
git diff --stat origin/main -- data results   # empty
```

`gradio` was installed into `/tmp/arb-eval-venv` so `tests/test_app.py` runs; nothing is excluded.

## Blockers

1. Generation identity drift (also affects robustness v2, which generates variants with current code): current `run_all.config_hash` values (`w1 2f00868a0ee3`,
   `w2 3fdcd18565b5`, `w3 9ab32afb5c88`, `w6 7bb3bef9a553`) differ from the saved rows
   (`w1 84c72812e449`, `w2 c14c2c8de42d`, `w3 b884fe8fe066` plus two partial reruns,
   `w6 26a928dd00bd`). Tier 1 is not affected because it only reads saved outputs. Only the
   optional injection stage is blocked.
2. `w3_critic` has 98 rows under `b884fe8fe066` and 6 rerun rows under two other hashes; two
   `(pair_id, repeat)` keys are duplicated across them. Pass `--config-hash
   w3_critic=b884fe8fe066`; its two missing pairs then appear as `generation_unavailable` in
   pairwise and are excluded from rates.
3. `sentence-transformers` is not in the test venv; NLI runs need the real environment.

## Next steps

1. Run the one-workflow Groq command above on 3 pairs; manually audit coverage semantic
   decisions (`requirement_coverage_record`), pairwise consistency and invalid-output rates.
2. Extend to all workflows on a fixed pilot subset, then the approved full run.
3. Robustness v2 pilot: `--robustness-limit 2 --robustness-families out_of_scope` with one
   workflow to check outcome classification on real outputs, then the 10-pair config above.
   Decide first whether to regenerate originals under the current config hash (blocker 1).
4. Tier 2: rubric judge integration checks and keyword/similarity reporting.
5. Push only with explicit approval.
