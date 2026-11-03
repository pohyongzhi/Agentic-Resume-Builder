# Human Agreement and ATS Evaluation Checkpoints

## Completed

- [x] Created isolated branch `feature/eval-human-ats` in `/tmp/arb-eval-human-ats` from `a866c34`.
- [x] Committed the initial, incomplete implementation as `14b5fe0`.
- [x] Reviewed and ran the initial code and tests; identified missing statuses, denominators, identity and aggregation.
- [x] Preserved collision-resistant review IDs (dataset, pair, repeat, workflow, config, output content).
- [x] Added fail-closed linkage of completed forms to saved evaluation rows.
- [x] Added judge-versus-human Spearman, quadratic weighted kappa, exact and within-one agreement, MAE, bias and bootstrap intervals.
- [x] Added claim-support Cohen's kappa, accuracy, precision, recall and F1.
- [x] Reported explicit eligible, scored, unscored, failed and missing-label denominators.
- [x] Reported `insufficient_labels` when no labels exist or fewer than 10 items are paired.
- [x] Kept missing labels and automated failures unscored, never zero.
- [x] Implemented the deterministic offline ATS proxy with `scored`/`unscored`/`failed` states and aggregation.
- [x] Added version and identity hashes for agreement and ATS outputs.
- [x] Added deterministic, network-free tests: 20 focused tests; full suite 385 passed excluding `tests/test_app.py` (needs `gradio`).
- [x] Created completion commit `9790d90`.
- [x] Preserved shared integration files, frozen dataset, thresholds and results.

## Pending integration

- [ ] Promote local identity constants to `config.py` and include agreement and ATS identity hashes in the evaluator identity.
- [ ] Call `evaluate_ats` from `_evaluate_output`, pass generation failures, and add ATS to checkpoint completeness.
- [ ] Add `summarise_ats` tables/files to `eval/reporting.py`.
- [ ] Add a CLI path for `calculate_completed_forms(..., evaluations=...)` and surface `judge_vs_human_status`.
- [ ] Add CLI/integration tests.
- [ ] Regenerate the review packet with the new mapping fields, collect independent ratings from at least two reviewers, then compute agreement.

## Current status

The isolated implementation is complete and tested. There are no code-level blockers. Agreement figures cannot be reported until human ratings are collected; integration is pending because this workstream was scoped away from shared files.

## Resume checklist

1. Enter `/tmp/arb-eval-human-ats` and verify the branch and clean status.
2. Read `HANDOVER.md` fully.
3. Run the focused pytest command recorded in the handover.
4. Inspect `calculate_completed_forms`, `evaluate_ats` and `summarise_ats` before editing shared integration code.
5. Do not create, infer or import human labels that reviewers did not provide.
6. Commit with a concise one-line message and do not push unless explicitly requested.
