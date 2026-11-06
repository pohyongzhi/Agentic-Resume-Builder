# Pairwise and Position-Swap Evaluation Handover

## Repository state

- Branch: `feature/eval-pairwise-position`
- Worktree: `/tmp/arb-eval-pairwise`
- Base: `origin/main` at `a866c34`
- Commits: `073a0c1` (initial module and tests, work in progress), `98d05d9` (completion and fixes)
- Status: implementation and isolated tests complete; shared evaluator integration is intentionally pending.

## Scope

This workstream adds an LLM-judge pairwise comparison between a challenger workflow output and a
baseline workflow output for the same dataset pair and repeat. Every comparison is judged twice:
challenger in position A (`forward`) and baseline in position A (`swapped`). Positional verdicts are
mapped to `challenger`/`baseline`/`tie`; a consensus winner exists only when both orders agree.
Order disagreement is reported as position-swap inconsistency, never resolved in favour of either side.

## Non-scope

The following shared integration files were deliberately not changed:

- `config.py`
- `evaluate.py`
- `eval/reporting.py`
- `eval/judge.py`
- `common/schemas.py`
- `README.md`

No generation or model-backed judge runs were performed. No dataset, threshold, or result files
were modified. No branch was pushed.

## Architecture and interfaces

`eval/pairwise.py` reuses `JudgeConfig`, `configured_model`, `DIMENSIONS`, and `RUBRIC` from
`eval/judge.py`, and mirrors its Ollama/Groq call path, retry loop, quotation validation, and
secret redaction. The judge sees anonymous `candidate_a`/`candidate_b` labels; workflow names never
appear in prompts.

Public API:

- `build_generation_pairs(rows, *, baseline_workflow="w2_baseline", challenger_workflows=None, config_hashes=None) -> list[dict]`:
  deterministically matches saved generation rows on `(dataset_hash, pair_id, repeat)`. Rejects
  duplicates, dataset-hash mismatches, and ambiguous config hashes. Each pair has
  `status` `ready` or `generation_unavailable` with a `failure_reason`.
- `candidate_text(row) -> str`: loads `resume_md` or `resume_path`; raises on failed or empty output.
- `evaluate_generation_pair(pair, original_resume, job_description, cache_dir, **kwargs) -> dict`:
  runner for one `build_generation_pairs` entry. Unavailable or unreadable generations return a
  `generation_unavailable` record without any judge call; ready pairs call `evaluate_pairwise`.
- `evaluate_pairwise(original_resume, job_description, challenger_resume, baseline_resume, cache_dir, *, challenger_workflow, baseline_workflow="w2_baseline", pair_id, repeat=0, dataset_hash=None, challenger_config_hash=None, baseline_config_hash=None, judge_config=None, client=None, model_info=None, force=False) -> dict`:
  two-order judgement with caching.
- `generation_unavailable_record(pair, reason=None) -> dict`
- `pairwise_identity(judge_config, model_info) -> dict`: evaluator identity, including `evaluator_hash`.
- `summarise_pairwise(records, group_fields=()) -> list[dict]`: grouped aggregates.

Local constants (candidates for promotion to `config.py`):

- `PAIRWISE_EVALUATOR_VERSION = "pairwise-position-swap-v1"`
- `PAIRWISE_PROMPT_VERSION = "pairwise-v1"`
- `PAIRWISE_SCHEMA_VERSION = "1"`
- `PAIRING_POLICY_VERSION = "same-dataset-pair-repeat-v1"`
- `POSITION_ORDER_VERSION = "challenger-first-then-baseline-first-v1"`
- Failure kinds: `FAILURE_JUDGE = "judge_failed"`, `FAILURE_INVALID = "invalid_output"`,
  `FAILURE_GENERATION = "generation_unavailable"`

### Record semantics

- `status`: `scored` (both orders validated), `unscored` (judge failure or invalid output in either
  order), or `generation_unavailable`.
- `failure_kind`: `None`, `judge_failed` (judge call raised; takes priority), `invalid_output`
  (schema or quotation validation rejected the response after retries), or `generation_unavailable`.
- Unscored records have `consensus_winner=None` and `position_swap_consistent=None`; they are never
  converted into a loss, tie, or score.
- `orders_scored`: 0, 1, or 2.
- `position_swap_consistent`, `consensus_winner`, `dimension_position_swap_consistent`,
  `dimension_consensus`: overall and per-dimension swap agreement.
- `orders.forward` / `orders.swapped`: positional winner, normalised winner and dimensions, full
  attempts (`scored`/`invalid`/`failed`) with raw responses and token metadata, and the user prompt.

### Caching

The cache key hashes the four input texts, the comparison identity (dataset hash, pair id, repeat,
both workflows, both config hashes), and the evaluator identity: full `JudgeConfig` (provider,
model, sampling, retries), model provider/tag/digest, evaluator/prompt/schema/pairing/order
versions, and hashes of the system prompt, rubric, and JSON schema. Records with a judge call
failure are not cached so a resumed run retries them; invalid-output records are cached. Corrupt
or mismatched cache files are recomputed. When `model_info` is supplied, no client is created
on a cache hit.

### Summary denominators

`summarise_pairwise` reports `total_pairs`, `generation_unavailable`, `eligible` (submitted to the
judge), `scored`, `unscored` (= eligible - scored), `failed` (judge failures), and `invalid`. Rates
use `scored` as denominator and are `None` when nothing is scored: `position_swap_consistency` and
`strict_win_rate` (95% Wilson intervals), `strict_loss_rate`, `position_neutral_score` (wins plus
half of ties and inconsistent outcomes), `first_position_selection_rate` over non-tie order
judgements, and per-dimension swap consistency. Failure reasons are counted.

## Files changed

- `eval/pairwise.py`
- `tests/test_pairwise.py`
- `docs/evaluation/workstreams/pairwise-position/HANDOVER.md`
- `docs/evaluation/workstreams/pairwise-position/CHECKPOINTS.md`

## Validation performed

```bash
cd /tmp/arb-eval-pairwise
PYTHONDONTWRITEBYTECODE=1 /tmp/arb-eval-venv/bin/pytest -q tests/test_pairwise.py
# 18 passed in 0.18s

PYTHONDONTWRITEBYTECODE=1 /tmp/arb-eval-venv/bin/pytest -q tests/test_judge.py tests/test_reporting.py tests/test_evaluator.py tests/test_metrics.py
# 26 passed in 0.39s

git diff --check
# clean
```

The pairwise tests use fake clients only. An autouse fixture blocks socket connections and the
default client constructor, so any network attempt fails the test.

## Pending integration

1. Promote the local version constants to `config.py` (or import them from `eval.pairwise`) and add
   the pairwise identity (`pairwise_identity(...)["evaluator_hash"]`) to evaluator identity and
   checkpoint-completeness checks.
2. Add a `pairwise` CLI stage in `evaluate.py` with `--pairwise-baseline`, `--pairwise-challengers`,
   `--pairwise-config-hash WORKFLOW=HASH`, and `--skip-pairwise` options.
3. Resolve judge `model_info` once in preflight via `configured_model` and pass it to every
   `evaluate_generation_pair` call.
4. Load saved generation rows, call `build_generation_pairs`, then `evaluate_generation_pair` for
   each pair with the original resume and JD from the frozen dataset; use a dedicated cache
   directory such as `root/pairwise_cache`.
5. Treat `scored`, `unscored` with `failure_kind="invalid_output"`, and `generation_unavailable` as
   complete for resumability; re-run `judge_failed` records.
6. Add pairwise aggregates (overall and by `challenger_workflow`) to `eval/reporting.py` JSON/CSV
   outputs, keeping all denominators and Wilson intervals.
7. Add CLI/integration tests for resume behaviour and identity changes.

## Blockers and limitations

- No code-level blockers.
- A model-backed pilot is required to check judge position bias and the rate of invalid outputs
  from strict quotation validation before a full run.
- Each comparison costs two to `2 * (max_retries + 1)` judge calls; hosted runs respect
  `min_request_interval_seconds`.
- `JudgeConfig.prompt_version` and `schema_version` refer to the single-output judge; they are
  included in the identity but the pairwise-specific versions are separate fields.

## Exact resume instructions

```bash
cd /tmp/arb-eval-pairwise
git status --short --branch
git log --oneline --decorate -5
sed -n '1,260p' docs/evaluation/workstreams/pairwise-position/HANDOVER.md
sed -n '1,200p' docs/evaluation/workstreams/pairwise-position/CHECKPOINTS.md
PYTHONDONTWRITEBYTECODE=1 /tmp/arb-eval-venv/bin/pytest -q tests/test_pairwise.py tests/test_judge.py tests/test_reporting.py tests/test_evaluator.py tests/test_metrics.py
```

For integration into another branch, cherry-pick the branch commits in order, then perform the
pending integration steps above without altering the frozen dataset.
