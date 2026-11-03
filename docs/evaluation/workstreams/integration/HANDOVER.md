# Evaluation metrics integration handover

## Resume here

- Branch: `feature/eval-metrics-integration`
- Worktree: `/tmp/arb-eval-integration`
- Base: `origin/main` at `a866c34`
- Global context: `/mnt/c/Users/keste/Downloads/CLAUDE.md`
- Status: all three workstreams merged and wired into the shared CLI, identity, caching and
  reports. Model-backed pilots and full runs have not been run. Nothing has been pushed.

```bash
cd /tmp/arb-eval-integration
git status --short --branch
git log --oneline --decorate -8
sed -n '1,400p' docs/evaluation/workstreams/integration/HANDOVER.md
sed -n '1,200p' docs/evaluation/workstreams/integration/CHECKPOINTS.md
PYTHONDONTWRITEBYTECODE=1 /tmp/arb-eval-venv/bin/pytest -q -p no:cacheprovider
```

## Objective

Integrate the independently implemented evaluation workstreams into the shared evaluator CLI,
identity, caching, reporting, documentation and tests without changing workflow generation.

## Merged branches

| Workstream | Branch / head | Merge commit |
| --- | --- | --- |
| Requirement coverage and injection | `feature/eval-coverage-injection` / `42a4668` | `7bbe174` |
| Pairwise and position swapping | `feature/eval-pairwise-position` / `174463c` | `71d000d` |
| Human agreement and ATS | `feature/eval-human-ats` / `3c667f8` | `e099d25` |

Integration commits: `84809af` (coverage and injection wiring), `f63cba0` (pairwise, ATS,
agreement wiring and coverage transport-failure retry), plus the commit containing this file.

## Required behaviour and how it is met

- Evaluation failures stay unscored: every component writes `None` plus a reason in
  `evaluation_failures[<metric>]`; reports count them in `unscored`/`failed`, never as zero.
- Cache identity: per-output components contribute `identity()` to
  `evaluator_identity.json["output_components"]` and therefore to the evaluator hash (the root
  directory). Each component's own cache key also hashes its inputs. Stage artifacts live under
  a stage identity hash (`injection/<identity_hash>/`, `pairwise/<evaluator_hash>/`).
- Generation artifacts and the frozen dataset are untouched: offline scoring only reads
  `results/results.jsonl` and saved run files; `git diff origin/main -- data results` is empty and
  `validate_dataset()` still returns hash `3bb1861e…6f81`.
- Explicit denominators: `eligible`, `scored`, `unscored`, `failed` in every new table.
- No network in tests: fake clients/workflows; stage tests also block sockets.
- Thresholds and historical results unchanged. `EVALUATOR_SCHEMA_VERSION` moved `2.2 -> 2.3`
  because evaluation records gained fields; old roots remain as historical artifacts.

## Plug-in points

### Per-output metrics (`evaluate.py`)

- Implement the `OutputComponent` protocol: `name`, `identity()`, `is_complete(record)`,
  `evaluate(record, generation_row, pair, tailored_resume, root, force)`.
- Register it in `build_output_components(args, judge_config, judge_model_info)`; model metadata
  is resolved there once per run.
- Components automatically join `_evaluation_identity`, `_record_complete` (shared by
  `evaluate_saved` and `evaluate_robustness`) and `_evaluate_output`.
- Current components: `RequirementCoverageComponent`, `ATSComponent`.

### Run-level stages (`evaluate.py`)

- Add a `Stage(name, run, index, pilot_limit, help, loader=None, in_all=True)` to `STAGES`.
  The parser then exposes the `<name>` command, `--skip-<name>` and `--<name>-limit`; `--pilot`
  applies `pilot_limit`; `report` reloads rows through `Stage.load(root)`.
- `run(run: EvaluationRun, limit)` receives args, root, pairs, generation rows, dataset hash,
  judge config, components, evaluation rows, and `run.resolved_judge_model()`.
- Current stages: `robustness`, `injection`, `pairwise`, `agreement`.

### Reports (`eval/reporting.py`)

- `STAGE_REPORTERS[name] = writer(output_dir, rows) -> dict[str, table]`.
- Per-output metrics with a scalar value: add the field to `METRICS`; adjust `_eligible` if some
  rows are not applicable.
- `write_reports(output_dir, rows, robustness_rows=(), stage_rows=None)` stays backwards
  compatible with the old `robustness_rows` argument.

### Configuration (`config.py`)

All version, threshold and weight constants from the three workstreams now live in `config.py`
and the modules read them from there: `REQUIREMENT_COVERAGE_*`, `INJECTION_*`, `PAIRWISE_*`,
`PAIRING_POLICY_VERSION`, `POSITION_ORDER_VERSION`, `AGREEMENT_SCHEMA_VERSION`,
`HUMAN_AGREEMENT_VERSION`, `MIN_AGREEMENT_PAIRS`, `CLAIM_LABELS`, `ATS_SCHEMA_VERSION`,
`OFFLINE_PROXY_VERSION`, `ATS_STATUSES`, `COMPOSITE_WEIGHTS`, `FORMAT_HAZARD_PENALTY`.
Values are unchanged.

## Integrated metrics

### Supportable-requirement coverage (per output)

Record fields: `requirement_coverage`, `requirement_coverage_status`
(`scored`, `unscored`, `no_supportable_requirements`, `generation_failed`, `evaluation_error`),
`requirement_coverage_supportable_count`, `_covered_count`, `_uncovered`,
`_evaluator_version`, `_evaluator_hash`, `_seconds`, and the full `requirement_coverage_record`.
Cache: `<root>/cache/requirement_coverage/`.

- `no_supportable_requirements` is not applicable: excluded from `eligible`, not a failure.
- `evaluation_error` (evaluator unreachable or rate limited) is not cached and is retried on resume.
  Invalid evaluator output is cached and final for that identity (same rule as the judge).
- `--coverage-provider` / `--coverage-model` default to the judge's provider/model; when they
  match and the judge is enabled, the judge `model_info` is reused.
- Integration change to the feature module: hosted calls honour
  `min_request_interval_seconds` (default `JUDGE_MIN_REQUEST_INTERVAL_SECONDS`), and transport
  failures are recorded as `failure_kind="evaluator_call_failed"` instead of cached as invalid.

Reports: `requirement_coverage.{csv,json}`, `requirement_coverage_by_fit_level.{csv,json}`
(outputs, generation_failures, not_applicable, eligible, scored, unscored, failed, mean, bootstrap
CI, micro coverage), plus the `requirement_coverage` row in `overall.csv` etc.

### Prompt-injection benchmark (stage `injection`)

`evaluate_injection(...)` writes `injection/<identity_hash>/sample_manifest.json` before any
generation (an existing different manifest aborts with `INJECTION_MANIFEST_CHANGED`), then
`runs/<workflow>/<pair_id>/<channel>/{resume.md,trace.json,generation.json,evaluation.json}` and
`evaluations.jsonl`. `generation.json` is written last and marks reusable generations;
`evaluation.json` marks reusable diagnostics. Diagnostics (protected fields, fabrication bait,
NLI faithfulness unless skipped) are computed against the clean source resume. Generation
failures have `injection_success=None`, `error` set, and are excluded from the rate.

Before generation, the stage checks that the current generation identity
(`run_all.config_hash(workflow)`) matches a saved generation config hash for that workflow and
aborts with `INJECTION_GENERATION_IDENTITY_MISMATCH` otherwise.

Reports: `injection.{csv,json}` (scope overall/workflow/channel/workflow_channel, eligible,
scored, unscored, failed, rate, Wilson CI) and `injection_diagnostics.{csv,json}`.

### Pairwise position swap (stage `pairwise`)

`evaluate_pairwise_stage(...)` resolves the judge model once (`EvaluationRun.resolved_judge_model`,
which resolves it even under `--skip-judge`), builds pairs with `build_generation_pairs`, and calls
`evaluate_generation_pair` for each with the frozen dataset resume/JD. Cache:
`<root>/pairwise_cache/`. Records: `pairwise/<evaluator_hash>/records/<challenger>/<pair_id>/repeat_<n>.json`.
Resume treats `scored` and `unscored`+`invalid_output` as final, retries `judge_failed`, and
recomputes `generation_unavailable` (no judge call). Options: `--pairwise-baseline`
(default `w2_baseline`), `--pairwise-challengers`, `--pairwise-config-hash WORKFLOW=HASH`
(repeatable). Baseline and challengers must be in `--workflows`; under `all` the stage is
skipped with a message when they are not, under `pairwise` it exits with an error.

Reports: `pairwise.{csv,json}` (by baseline/challenger) and `pairwise_by_fit_level.{csv,json}`;
nested values are JSON strings in CSV. `report` reloads the most recently written pairwise tree.

### ATS proxy (per output)

`ATSComponent` calls `evaluate_ats(OfflineATSProvider(), ...)` with the pair's keyword families;
failed or empty generations are passed as `generation_error` (status `failed`). Record fields:
`ats_status`, `ats_proxy_score`, `ats_score_kind`, `ats_identity_hash`, `ats_components`,
`ats_record` (provider fields stay nested). Disable with `--skip-ats`.
Reports: `ats.{csv,json}` (eligible, scored, unscored, failed, mean/min/max, component means and
counts, provider identity) and the `ats_proxy_score` row in `overall.csv`.

### Human agreement (stage `agreement`, not in the critical path)

`evaluate_agreement(review_dir, evaluation_rows)` calls `calculate_completed_forms` for
`<root>/human_review/` and links to the root's evaluation rows. Without a review packet it is
skipped under `all` and errors under `agreement`. Reports: `human_agreement.{csv,json}` with one
row per dimension/overall/claims, unified `eligible/scored/unscored/failed/missing_human_label`
columns and `calibration_claimable` (true only when the status is `computed`).
`report`/`all` no longer overwrite the review forms once any reviewer label exists.
Evaluation rows from `score` carry no `variant` key, which links as `""` on both sides.

## CLI commands

```bash
# Offline scoring with coverage and ATS (judge/NLI as configured)
python evaluate.py score --workflows w1_single_pass w2_baseline
# Prompt-injection benchmark (needs the generation model via Ollama)
python evaluate.py injection --workflows w1_single_pass --injection-limit 2
# Pairwise position swap against the baseline
python evaluate.py pairwise --workflows w2_baseline w1_single_pass --pairwise-limit 2
# Human agreement after reviewers complete <root>/human_review/*.csv
python evaluate.py agreement --workflows w1_single_pass w2_baseline
# Rebuild every summary from saved indexes (no model calls)
python evaluate.py report --workflows w1_single_pass w2_baseline
# Everything, small pilot limits
python evaluate.py all --workflows w1_single_pass w2_baseline --pilot
```

`--skip-judge`, `--skip-coverage` and `--skip-ats` change the evaluator identity (as
`--skip-judge` already did), so use the same switches and judge/coverage models for `score`,
stages and `report` to stay in one root.

## Validation

```bash
cd /tmp/arb-eval-integration
PYTHONDONTWRITEBYTECODE=1 /tmp/arb-eval-venv/bin/pytest -q -p no:cacheprovider
# 448 passed, 21 subtests passed in 9.50s (full suite, including tests/test_app.py)
PYTHONDONTWRITEBYTECODE=1 /tmp/arb-eval-venv/bin/pytest -q -p no:cacheprovider tests/test_eval_integration.py
# 27 passed
git diff --check   # clean
git diff --stat origin/main -- data results   # empty
```

`gradio` was installed into `/tmp/arb-eval-venv` so `tests/test_app.py` runs; nothing is excluded.

## Blockers found against the saved results

1. Generation identity drift: current `run_all.config_hash` values (`w1 2f00868a0ee3`,
   `w2 3fdcd18565b5`, `w3 9ab32afb5c88`, `w6 7bb3bef9a553`) do not match the saved rows
   (`w1 84c72812e449`, `w2 c14c2c8de42d`, `w3 b884fe8fe066`/`3124631d3f9c`/`58375bf40445`,
   `w6 26a928dd00bd`). The injection stage therefore refuses to generate for these workflows.
   Decision needed: regenerate under the current code, or check out the generation commit for
   injection runs. Offline scoring, pairwise and agreement are not affected.
2. `w3_critic` has two duplicate `(pair_id, repeat)` keys across three config hashes, so the
   existing `_validate_generation_rows` check rejects any run that includes `w3_critic`, before
   `--pairwise-config-hash` filtering can apply. Pilot with `w1_single_pass`/`w6_best_of_n`
   against `w2_baseline`, or resolve the duplicates.
3. `sentence_transformers` is not installed in the test venv; NLI runs need the real environment
   (or `--skip-faithfulness`).

## Live Groq pilot requirements (not run)

- `GROQ_API_KEY` exported; `groq==0.37.1` installed; judge `--judge-provider groq --judge-model
  openai/gpt-oss-120b` (only the two gpt-oss models are approved by `configured_model`).
- Coverage and pairwise follow the judge provider/model by default and share the 15 s minimum
  request interval; per output expect 1-3 judge calls plus 0-2 coverage calls, and 2-4 calls per
  pairwise comparison, so a 3-output, 2-pair pilot is roughly 10-20 hosted calls.
- Saved generations for `w2_baseline` plus one challenger not affected by blocker 2.
- NLI dependencies (`sentence-transformers`, model download) or `--skip-faithfulness`.
- Injection needs local Ollama with the writer models and resolution of blocker 1; keep it out
  of a hosted-only pilot with `--skip-injection --skip-robustness`.
- Suggested command:

```bash
GROQ_API_KEY=... python evaluate.py all --workflows w2_baseline w1_single_pass \
  --judge-provider groq --judge-model openai/gpt-oss-120b \
  --pilot --skip-injection --skip-robustness
```

Then manually audit `requirement_coverage_record` semantic decisions, pairwise position
consistency and invalid-output rates before any full run.

## Next steps

1. Decide on blocker 1 (generation drift) and blocker 2 (`w3_critic` duplicates).
2. Run the hosted pilot above and audit its outputs.
3. Collect two independent reviewers' forms, then run `agreement`.
4. Run the approved full evaluation; push only with explicit approval.
