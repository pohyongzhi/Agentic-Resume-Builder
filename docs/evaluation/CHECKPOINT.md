# Evaluation reliability checkpoint

Updated: 2026-11-03 (Asia/Singapore).

## Current state

- Repository: `/Users/ethan/Documents/GitHub/Agentic-Resume-Builder`
- Branch: `fix/w4-operational-step-budget`, based on `main` at `e2ebd15`.
- Implementation and regression tests are complete in the working tree.
- Changes are not yet committed or pushed.
- Full validation: 347 tests passed and 21 subtests passed in 5.30 seconds.
- `git diff --check` passes.

## Completed fixes

1. Exact faithfulness support now requires equality with a complete normalized evidence unit. A positive substring inside a negated or qualified source sentence is sent to NLI instead of being accepted automatically.
2. Protected fields retain their employment or education entry relationship. Reordering and omissions remain valid; swapping employers, dates or institutions between entries is flagged.
3. Keyword bait detection excludes explicit local denials in source and output while retaining positive mentions in contrast clauses. This remains a deliberately narrow lexical rule.
4. W4 has concrete tool expectations and step budgets: minimum 5 when verification has no eligible claims, otherwise actual calls determine efficiency; maximum 13 for the current two grounding rounds.
5. Online, robustness and offline evaluation paths preserve generated resumes and traces when evaluation fails. Component failures are recorded in `evaluation_failures`; generation and evaluation durations are separate online.
6. Generation cache hashes include shared generation code and the selected workflow source. Evaluator-only settings do not invalidate generated outputs.
7. Human-review IDs include workflow, generation configuration and output content, preventing same-pair collisions across outputs.
8. Evaluator/component versions were incremented so stale scores are not presented as current.

## Validation evidence

- Complete command: `PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m pytest -q`.
- Full result on 2026-11-03: 347 tests passed and 21 subtests passed in 5.30 seconds.
- The audit added coverage for online and offline operational-evaluator failures before the final run.
- Deterministic protected-field and bait checks were replayed over all 100 saved W1 outputs. All 100 artifacts loaded. Aggregate historical counts remained 5 outputs with protected-field violations and 60 with bait violations; this was a compatibility check, not a new calibrated evaluation.

## Scientific limits

Passing tests does not make an automated evaluator flawless. Negation handling covers explicit local phrases, NLI can still misclassify support, evidence retrieval can miss the best source, and human calibration is still outstanding. Historical v3 results remain historical and should not be relabelled as v4 results.

Read [HANDOVER.md](HANDOVER.md) for continuation details and [reliability-fixes.md](reliability-fixes.md) for the behavioral contract.
