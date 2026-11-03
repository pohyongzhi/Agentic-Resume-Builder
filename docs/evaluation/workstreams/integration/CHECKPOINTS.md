# Evaluation metrics integration checkpoints

## 2026-11-04 — Workstreams started

Status: complete.

- Read `/mnt/c/Users/keste/Downloads/CLAUDE.md` and adopted it as global context.
- Created isolated feature branches and worktrees for all three metric workstreams plus integration.
- Assigned non-overlapping file ownership to prevent concurrent merge conflicts.
- Created `/tmp/arb-eval-venv` and installed lightweight evaluator test dependencies.
- Verified the existing evaluator baseline:

```text
/tmp/arb-eval-venv/bin/pytest -q \
  tests/test_reporting.py tests/test_judge.py tests/test_evaluator.py
25 passed in 0.34s
```

- Required every branch to maintain its own `HANDOVER.md` and `CHECKPOINTS.md` before commit.

Next checkpoint: review and integrate the first completed workstream commit.


## 2026-11-04 — Coverage and injection integrated

Status: complete.

- Merged `feature/eval-coverage-injection` (`42a4668`) as `7bbe174`.
- Moved coverage and injection versions into `config.py`; bumped `EVALUATOR_SCHEMA_VERSION`
  to `2.3`.
- Added the `OutputComponent` protocol, `build_output_components`, `RequirementCoverageComponent`
  and the shared `_record_complete` checkpoint check in `evaluate.py`.
- Added the `Stage` registry (`STAGES`) with generated `--skip-<stage>`/`--<stage>-limit`
  options, and the `injection` stage with its manifest written before generation, resumable
  `generation.json`/`evaluation.json` artifacts and a generation-identity guard.
- Added coverage and injection tables and `STAGE_REPORTERS` in `eval/reporting.py`.
- Installed `gradio` into `/tmp/arb-eval-venv` so the full suite collects.
- Full suite: `398 passed, 21 subtests passed`. Commit `84809af`.

## 2026-11-04 — Pairwise, ATS and human agreement integrated

Status: complete; pilots not run.

- Merged `feature/eval-pairwise-position` (`174463c`) as `71d000d` and
  `feature/eval-human-ats` (`3c667f8`) as `e099d25`; no conflicts.
- Promoted their identity constants into `config.py` (values unchanged).
- Added `ATSComponent` (per output), and `pairwise` and `agreement` stages.
- Pairwise resumes scored/invalid-output records, retries judge failures, and stores records
  under the pairwise evaluator hash with a dedicated `pairwise_cache/`.
- Agreement reports `calibration_claimable` only when the status is `computed`; review forms
  are no longer overwritten once any reviewer label exists.
- Coverage: transport failures are no longer cached as final unscored results and hosted calls
  respect the judge request interval.
- Added `tests/test_eval_integration.py` (27 tests, no network).
- Full suite: `448 passed, 21 subtests passed in 9.50s`. Commit `f63cba0`.
- Found two data blockers (generation identity drift; duplicate `w3_critic` keys) recorded in
  `HANDOVER.md`.

Next checkpoint: user decision on the blockers, then the hosted Groq pilot.

## 2026-11-04 — Re-scoped to tier-1 evals

Status: complete; pilots not run.

- Priority change: tier 1 is faithfulness, requirement coverage, pairwise vs `w2_baseline` and
  operational metrics. Human agreement dropped; ATS deferred to tier 3; injection optional.
- Removed the ATS component and the agreement stage from the CLI and reports (module code and
  tests from `feature/eval-human-ats` stay merged; the merge was clean).
- Took `injection` out of `all`; it remains its own working, tested stage.
- Added `--pair-ids` / `--pair-ids-file` (fixed pair subsets for score and pairwise; indexes
  and reports merge across subsets) and `--config-hash WORKFLOW=HASH` (filters saved generations
  before validation, which resolves the `w3_critic` duplicates).
- Pairwise index now keeps a compact per-order summary so `report` reproduces
  first-position rates without the full records.
- Judge, pairwise and coverage now share one hosted request timer
  (`eval.judge.wait_for_hosted_request_slot`, 15 s).
- Added a tier-1 end-to-end CLI test (subset, resume, report rebuild).
- Full suite: `446 passed, 21 subtests passed in 11.39s`. Commit `cea794e`.

Next checkpoint: 3-pair Groq pilot for one challenger, then manual audit.
