# Handover: evaluator reliability fixes

Updated: 2026-11-03.

The implementation is on branch `fix/w4-operational-step-budget`, based on the same commit as `main` (`e2ebd15`). Changes are not yet committed or pushed. Preserve the working tree and inspect it before further changes.

## Resume commands

```sh
cd /Users/ethan/Documents/GitHub/Agentic-Resume-Builder
git status --short --branch
git diff --check
git diff --stat
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m pytest -q
```

The full-suite result on 2026-11-03 was 347 tests passed and 21 subtests passed in 5.30 seconds. Expected modified implementation files are `config.py`, `eval/human_review.py`, `eval/metrics.py`, `eval/operational.py`, `evaluate.py`, `run_all.py`, and `run_robustness.py`, plus focused tests and documentation.

## What changed

- `eval/metrics.py`: equality-only deterministic support, explicit local-negation filtering for bait terms, and entry-aware protected-field relationships.
- `eval/operational.py`: W4 tool sequence with 5 minimum steps when verification has no eligible claims and a 13-step maximum for the current workflow.
- `run_all.py`: per-component evaluation failure isolation, artifact preservation, separate timing, and workflow-aware generation source fingerprints.
- `run_robustness.py`: the same artifact-preservation and workflow-hash behavior through shared scoring logic.
- `evaluate.py`: operational scorer failures are captured with the other offline evaluation failures.
- `eval/human_review.py`: collision-resistant IDs derived from dataset, pair, repeat, workflow, config and output content.
- `config.py`: evaluator schema 2.2, NLI v4, protected-fields v2 and keyword-bait v2.

Behavior and limitations are documented in [reliability-fixes.md](reliability-fixes.md). [CHECKPOINT.md](CHECKPOINT.md) contains validation evidence.

## Constraints for continuation

Do not change frozen datasets, thresholds or historical result files to improve reported scores. Do not claim human calibration; no human-labelled validation set has been completed. Do not rewrite the historical W1 v3 summary as v4. New results require an explicit rescore with the new evaluator identity.

The user requested a push and PR. After final review, commit only these changes, push this branch, and open a PR targeting `main`.
