# Handover: evaluator reliability fixes

Updated: 2026-11-01.

The implementation is complete on branch `fix/evaluation-reliability`. It is uncommitted and unpushed. The branch began at `5f86d9a`, the merge of teammate PR #2 into main. Preserve the working tree and inspect it before further changes.

## Resume commands

```sh
cd /Users/ethan/Documents/GitHub/Agentic-Resume-Builder
git status --short --branch
git diff --check
git diff --stat
PYTHONDONTWRITEBYTECODE=1 venv/bin/python -m pytest -q
```

The final full-suite result was 219 passing tests in 3.78 seconds. Expected modified implementation files are `config.py`, `eval/human_review.py`, `eval/metrics.py`, `eval/operational.py`, `evaluate.py`, `run_all.py`, and `run_robustness.py`, plus focused tests and documentation.

## What changed

- `eval/metrics.py`: equality-only deterministic support, explicit local-negation filtering for bait terms, and entry-aware protected-field relationships.
- `eval/operational.py`: W4 tool sequence with 6 minimum and 13 maximum steps for the current workflow.
- `run_all.py`: per-component evaluation failure isolation, artifact preservation, separate timing, and workflow-aware generation source fingerprints.
- `run_robustness.py`: the same artifact-preservation and workflow-hash behavior through shared scoring logic.
- `evaluate.py`: operational scorer failures are captured with the other offline evaluation failures.
- `eval/human_review.py`: collision-resistant IDs derived from dataset, pair, repeat, workflow, config and output content.
- `config.py`: evaluator schema 2.2, NLI v4, protected-fields v2 and keyword-bait v2.

Behavior and limitations are documented in [reliability-fixes.md](reliability-fixes.md). [CHECKPOINT.md](CHECKPOINT.md) contains validation evidence.

## Constraints for continuation

Do not change frozen datasets, thresholds or historical result files to improve reported scores. Do not claim human calibration; no human-labelled validation set has been completed. Do not rewrite the historical W1 v3 summary as v4. New results require an explicit rescore with the new evaluator identity.

No push was requested for this new branch. If the user later requests a push or PR, first inspect the final diff and status, commit only these changes, then push the current branch. Follow environment approval requirements for Git metadata and network operations.
