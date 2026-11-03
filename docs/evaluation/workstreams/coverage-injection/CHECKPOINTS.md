# Coverage and Injection Evaluation Checkpoints

## Completed

- [x] Created isolated branch `feature/eval-coverage-injection` in `/tmp/arb-eval-coverage` from `a866c34988726e392d81952538048d27025d8e46`.
- [x] Implemented hybrid lexical/semantic supportable-requirement coverage.
- [x] Enforced exact quotation evidence for positive semantic decisions.
- [x] Represented invalid evaluator output as unscored and zero denominator as `None`.
- [x] Separated supportability caching from per-output coverage caching.
- [x] Implemented deterministic resume/JD injection canaries and collision checks.
- [x] Implemented complete-token detection and matched-span audit data.
- [x] Kept generation failures outside the injection success-rate denominator.
- [x] Implemented stratified immutable manifests and Wilson interval aggregation.
- [x] Added fake-client and fake-workflow tests with no network dependency.
- [x] Passed 16 focused tests and 10 existing metric/judge regression tests.
- [x] Created implementation commit `0c4537b88a0dbd966578d4825eeec806af30d39e`.
- [x] Preserved shared integration files and frozen dataset content.

## Pending integration

- [ ] Add shared version/config constants and evaluator identity fields.
- [ ] Integrate coverage into saved-output scoring and checkpoint completeness.
- [ ] Integrate injection command, manifest, resumable artifact paths, and normal scoring diagnostics.
- [ ] Add report tables/files for coverage and injection.
- [ ] Add CLI/integration tests.
- [ ] Run a small model-backed pilot and manually audit semantic decisions.
- [ ] Run the approved full evaluation only after pilot acceptance.

## Current status

The isolated implementation is complete and tested. There are no code-level blockers. Integration is pending because this workstream was scoped away from shared files that other workstreams also modify.

## Resume checklist

1. Enter `/tmp/arb-eval-coverage` and verify the branch and clean status.
2. Read `HANDOVER.md` fully.
3. Run the four-suite pytest command recorded in the handover.
4. Inspect `evaluate_requirement_coverage` and `run_injection_attack` before editing shared integration code.
5. Resolve evaluator model metadata once per run and pass it to coverage calls.
6. Preserve the dataset hash and write the injection manifest before generation.
7. Keep failed generations separate from completed security trials in every report.
8. Commit with a concise one-line message and do not push unless explicitly requested.

