# Evaluation metrics integration handover

## Resume here

- Branch: `feature/eval-metrics-integration`
- Worktree: `/tmp/arb-eval-integration`
- Base: `origin/main` at `a866c34`
- Global context: `/mnt/c/Users/keste/Downloads/CLAUDE.md`
- Status: integration in progress; do not push without explicit user approval

Start a continuation session with:

```bash
cd /tmp/arb-eval-integration
git status --short --branch
sed -n '1,240p' docs/evaluation/workstreams/integration/HANDOVER.md
sed -n '1,260p' docs/evaluation/workstreams/integration/CHECKPOINTS.md
git worktree list
```

## Objective

Integrate the independently implemented evaluation workstreams into the shared evaluator CLI,
identity, caching, reporting, documentation, and test suite without changing workflow generation.

## Parallel branches

| Workstream | Branch | Worktree | Continuity directory |
| --- | --- | --- | --- |
| Requirement coverage and injection | `feature/eval-coverage-injection` | `/tmp/arb-eval-coverage` | `docs/evaluation/workstreams/coverage-injection/` |
| Pairwise and position swapping | `feature/eval-pairwise-position` | `/tmp/arb-eval-pairwise` | `docs/evaluation/workstreams/pairwise-position/` |
| Human agreement and ATS | `feature/eval-human-ats` | `/tmp/arb-eval-human-ats` | `docs/evaluation/workstreams/human-ats/` |

## Integration ownership

This branch owns shared edits to `config.py`, `evaluate.py`, `eval/reporting.py`, common evaluator
schemas if needed, CLI documentation, and cross-workstream tests. Feature agents must not edit these
shared files.

## Required behavior

- Evaluation failures remain unscored and are never coerced to zero.
- Cache identities include inputs, evaluator versions, model/provider identity, and prompt/schema
  versions.
- Existing generation artifacts and the frozen dataset remain unchanged.
- Metrics report explicit eligible, scored, unscored, and failed denominators.
- No external vendor or hosted judge call occurs in unit tests.
- No commits or branches are pushed without explicit user approval.

## Current state

- Four isolated worktrees were created from `origin/main`.
- A temporary test environment exists at `/tmp/arb-eval-venv`.
- Baseline focused evaluator tests pass: `25 passed` for reporting, judge, and evaluator tests.
- Three agents are implementing isolated modules and tests in parallel.

## Next steps

1. Review each feature branch handover, checkpoints, diff, and tests.
2. Merge or cherry-pick feature commits into this integration branch.
3. Add shared configuration, evaluator identity, CLI stages, resumability checks, and reports.
4. Run focused and full zero-network tests plus dataset validation.
5. Run small local pilots only when required models/providers are available.
6. Record final commands, results, remaining blockers, and commit hashes in both continuity files.

