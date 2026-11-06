# Pairwise and Position-Swap Evaluation Checkpoints

## Completed

- [x] Created isolated branch `feature/eval-pairwise-position` in `/tmp/arb-eval-pairwise` from `a866c34`.
- [x] Committed the initial module and tests as a work-in-progress commit (`073a0c1`).
- [x] Reviewed the initial work: 9 tests passed; module was syntactically complete but lacked
      failure-kind accounting, a generation-pair runner, and a `failed` denominator.
- [x] Separated judge call failures (`judge_failed`) from invalid output (`invalid_output`).
- [x] Kept every failure unscored with no winner; never coerced to a loss or score.
- [x] Stopped caching judge call failures; recompute corrupt or mismatched cache files.
- [x] Retried transport failures with the original prompt instead of a validation correction.
- [x] Treated empty judge output as invalid output rather than a transport failure.
- [x] Added evaluator version and model provider/tag to the cache identity.
- [x] Added optional pre-resolved `model_info`; no client is created on cache hits.
- [x] Added per-dimension normalised preferences and swap consistency.
- [x] Added `evaluate_generation_pair` and `generation_unavailable_record`.
- [x] Reported explicit total/unavailable/eligible/scored/unscored/failed/invalid denominators.
- [x] Counted inconsistent outcomes as ties in `position_neutral_score`.
- [x] Added a network-blocking fixture; all tests use fake clients.
- [x] Passed 18 pairwise tests and 26 existing judge/reporting/evaluator/metrics tests.
- [x] Created completion commit `98d05d9`.
- [x] Preserved shared integration files, frozen dataset, thresholds, and result files.

## Pending integration

- [ ] Promote pairwise version constants and identity to shared configuration and evaluator identity.
- [ ] Add a `pairwise` CLI stage with baseline/challenger/config-hash selection and a skip option.
- [ ] Resolve judge model metadata once per run and pass `model_info` to every comparison.
- [ ] Wire `build_generation_pairs` and `evaluate_generation_pair` into saved-output evaluation with a dedicated cache directory.
- [ ] Re-run `judge_failed` records on resume; treat other statuses as complete.
- [ ] Add pairwise aggregates and per-dimension consistency to report JSON/CSV outputs.
- [ ] Add CLI/integration tests.
- [ ] Run a small model-backed pilot and manually audit position bias and invalid-output rate.
- [ ] Run the approved full evaluation only after pilot acceptance.

## Current status

The isolated implementation is complete and tested. There are no code-level blockers. Integration
is pending because this workstream was scoped away from shared files.

## Resume checklist

1. Enter `/tmp/arb-eval-pairwise` and verify the branch and clean status.
2. Read `HANDOVER.md` fully.
3. Run the pytest command recorded in the handover.
4. Inspect `evaluate_pairwise`, `evaluate_generation_pair`, and `summarise_pairwise` before editing shared integration code.
5. Resolve judge model metadata once per run and pass it to every comparison.
6. Keep `generation_unavailable`, `failed`, and `invalid` counts separate from scored comparisons in every report.
7. Commit with a concise one-line message and do not push unless explicitly requested.
