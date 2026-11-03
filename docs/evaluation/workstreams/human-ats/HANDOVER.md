# Human Agreement and ATS Evaluation Handover

## Repository state

- Branch: `feature/eval-human-ats`
- Worktree: `/tmp/arb-eval-human-ats`
- Base: `origin/main` at `a866c34`
- Commits: `14b5fe0` (initial implementation, incomplete), `9790d90` (completion and fixes), plus the commit adding these continuity files.
- Status: implementation and isolated tests complete; shared evaluator integration is intentionally pending.

## Scope

This workstream adds two evaluation components:

1. Judge-versus-human agreement: links completed blinded human review forms to saved automated scores and reports agreement per rubric dimension, for the overall rubric mean, and for claim support.
2. A deterministic ATS-style compatibility proxy: section, formatting, parseability and keyword-coverage checks behind a versioned provider boundary.

## Non-scope

The following shared integration files were deliberately not changed:

- `config.py`
- `evaluate.py`
- `eval/reporting.py`
- `common/schemas.py`
- `README.md`

No human labels were created or imported and no agreement figures exist yet. No dataset, threshold, or result files were modified. No hosted or vendor ATS call is made. No branch was pushed.

## Architecture and interfaces

### Human agreement (`eval/human_review.py`)

Blinded review IDs are unchanged: they derive from dataset hash, pair, repeat, workflow, generation config hash and the tailored-resume content hash (see `docs/evaluation/reliability-fixes.md`). Claim IDs derive from the review ID, claim index and claim-text hash.

`write_review_artifacts(output_dir, rows, limit=20, seed=42)` now also writes:

- `review_mapping.private.json` with full linkage identity per review: dataset hash, workflow, pair, repeat, variant, config hash, evaluator hash, SHA-256 of tailored resume / original resume / JD, judge cache key, judge evaluator hash, and per-claim `claim_id` and text hash.
- `review_manifest.json` with schema version, seed, limit, sample size, dataset and evaluator hashes, and review IDs.
- a `claim_id` column in `claim_support_ratings.csv`.

`calculate_completed_forms(output_dir, evaluations=None, bootstrap_samples=1000, seed=42)` is the entry point. `evaluations` is a list of evaluation rows or a JSONL path. It writes `agreement.json` and, when linked, `judge_human_agreement.csv` and `claim_human_agreement.csv`. CLI: `python -m eval.human_review <review_dir> [--evaluations rows.jsonl] [--bootstrap-samples N] [--seed S]`.

Linkage fails closed (`ValueError`) on any identity mismatch: no or several matching rows, stale content hashes, judge cache key or judge evaluator hash mismatch, claim count or claim text changes, unknown review IDs, duplicate ratings, or a `claim_id` that disagrees with the mapping.

Report structure:

- `identity`: from `agreement_identity()`; contains `agreement_version`, `schema_version`, `min_scored_items`, rubric dimensions, claim labels and `agreement_hash`.
- `human_labels`: reviewer, rubric-rating and claim-rating counts.
- `rubric_inter_human`, `claim_support_inter_human`, `inter_human_status` (`insufficient_labels` with fewer than two reviewers).
- `judge_vs_human` (`None` when not linked) and `judge_vs_human_status`: `not_linked`, `insufficient_labels`, `partial`, or `computed`.
- `judge_vs_human.dimensions[<dimension>]` and `judge_vs_human.overall`: `status`, `status_reason`, `eligible_items`, `scored_items`, `unscored_items`, `failed_items` (automated judge score missing), `missing_human_label_items`, `human_labelled_items`, `judge_scored_items`, `ordinal_consensus_items`, `spearman`, `quadratic_weighted_kappa`, `exact_agreement`, `within_one_point`, `mean_absolute_error`, `signed_bias`, a bootstrap `*_ci_95` for each, and `null_reasons`.
- `judge_vs_human.claim_support`: `status`, `eligible_claims`, `scored_claims`, `unscored_claims`, `failed_claims` (automated `supported` not boolean), `missing_human_label_claims`, `consensus_ties`, `cohen_kappa`, `accuracy`, precision/recall/F1 for `supported`, confusion matrix, and the label mapping (`contradicted` and `unsupported` both map to not supported).

Rules:

- A pair needs both a human label and an automated score; anything else is unscored and counted in its denominator. Nothing is imputed or coerced to zero.
- Blank form cells are missing labels. Malformed values fail closed.
- Human consensus for kappa and exact agreement is the unique mode; ties are excluded from ordinal metrics. Non-integer judge scores are excluded from ordinal metrics but kept for Spearman and error metrics.
- The overall comparison uses the unadjusted mean of the five judge dimensions against the mean of the five human dimension means. The judge critical-failure cap is excluded so the scales match. Kappa and exact agreement are undefined on this scale.
- `status` is `insufficient_labels` when there are no human labels or fewer than `MIN_AGREEMENT_PAIRS` (10) scored items. Descriptive statistics are still reported for two or more pairs but must not be presented as calibration.

### ATS proxy (`eval/ats.py`)

- `OfflineATSProvider()`: deterministic local proxy. Components: `parseability`, `section_completeness` (experience, education, skills headings including bold forms), `formatting_compatibility` (HTML/Markdown tables, private-use characters), `keyword_coverage` (diagnostic only), and `supported_requirement_coverage` (only keyword families present in the original resume). The composite `ats_proxy_score` uses `COMPOSITE_WEIGHTS`, renormalised when supported coverage is undefined. Keyword stuffing of unsupported terms cannot raise the score. Visual layout is not assessed.
- `CallableATSProvider(provider, provider_version, scorer, score_kind="ats_vendor_score")`: adapter for an application-owned scorer; used with fakes in tests.
- `ats_identity(provider)`: validated identity with `provider_config_hash` and `ats_identity_hash`.
- `offline_proxy_config_hash()`: hash of every rule that determines the offline score.
- `evaluate_ats(provider, original_resume, job_description, tailored_resume, keywords=None, generation_error=None)`: returns a record with `ats_status` (`scored`, `unscored` when the scorer raises or returns invalid output, `failed` when `generation_error` is set), the score field named by `score_kind` (`None` unless scored), `ats_components`, `details`, identity fields, `ats_cache_key`/`ats_request_hash` (inputs, keywords and provider identity), `ats_response_hash`, a redacted `ats_failure_reason`, and `ats_seconds`.
- `comparable_ats_records(left, right)`: true only for the same provider, version, schema, score kind and config hash.
- `summarise_ats(records, group_by=("workflow",))`: `eligible`, `scored`, `unscored`, `failed`, mean/min/max over scored records only, and per-component means with `n`. Rejects records from more than one scoring contract.

### Local identity constants

These are defined locally and should be promoted to `config.py` or included in the evaluator identity during integration:

| Constant | Module | Value |
| --- | --- | --- |
| `AGREEMENT_SCHEMA_VERSION` | `eval/human_review.py` | `"1.1"` |
| `HUMAN_AGREEMENT_VERSION` | `eval/human_review.py` | `"judge-human-agreement-v1"` |
| `MIN_AGREEMENT_PAIRS` | `eval/human_review.py` | `10` |
| `CLAIM_LABELS` | `eval/human_review.py` | `("supported", "contradicted", "unsupported")` |
| `ATS_SCHEMA_VERSION` | `eval/ats.py` | `"1.1"` |
| `OFFLINE_PROXY_VERSION` | `eval/ats.py` | `"offline-ats-proxy-v1"` |
| `COMPOSITE_WEIGHTS` | `eval/ats.py` | parseability 0.20, sections 0.25, formatting 0.15, supported coverage 0.40 |
| `FORMAT_HAZARD_PENALTY` | `eval/ats.py` | `0.34` |
| `ATS_STATUSES` | `eval/ats.py` | `("scored", "unscored", "failed")` |

## Files changed

- `eval/ats.py` (new)
- `eval/human_review.py`
- `tests/test_ats.py` (new)
- `tests/test_human_agreement.py` (new)
- `docs/evaluation/workstreams/human-ats/HANDOVER.md`
- `docs/evaluation/workstreams/human-ats/CHECKPOINTS.md`

## Fixes applied over the initial implementation

- Agreement summaries had no status: added `insufficient_labels` / `computed` with reasons, a minimum pair count, and top-level `judge_vs_human_status` and `inter_human_status`.
- Denominators did not separate automated failures from missing labels: added `scored`, `failed` and `missing_human_label` counts for dimensions, overall and claims.
- Ordinal metrics truncated judge scores with `int()`; non-integer or out-of-range scores now leave the ordinal metrics instead.
- The overall comparison used the capped judge `overall_mean` against an uncapped human mean; it now uses the unadjusted judge dimension mean.
- Overall `null_reasons` misreported why kappa was null.
- `_weighted_kappa` treated boolean categories as ordinal because `bool` is an `int`; harmless for two categories but now explicit.
- Blank cells in a reviewer's row raised errors; they are now missing labels.
- ATS had no generation-failure state, no aggregation, and no hash of its scoring rules in its identity; added `failed`, `summarise_ats`, `provider_config_hash` and `ats_identity_hash`.
- ATS headings in bold (`**EXPERIENCE**`) were not recognised.

## Validation performed

```bash
cd /tmp/arb-eval-human-ats
/tmp/arb-eval-venv/bin/python -m py_compile eval/ats.py eval/human_review.py
PYTHONDONTWRITEBYTECODE=1 /tmp/arb-eval-venv/bin/pytest -q -p no:cacheprovider tests/test_ats.py tests/test_human_agreement.py tests/test_reporting.py
# 25 passed
PYTHONDONTWRITEBYTECODE=1 /tmp/arb-eval-venv/bin/pytest -q -p no:cacheprovider tests/ --ignore=tests/test_app.py
# 385 passed, 21 subtests passed
git diff --check
# clean
```

`tests/test_app.py` needs `gradio`, which is not installed in the shared venv; it does not touch these modules. `pypdf` and `trafilatura` were installed into `/tmp/arb-eval-venv` so `tests/test_tools.py` can run.

## Pending integration

1. Promote or reference the local identity constants above from `config.py`, and include `agreement_identity()["agreement_hash"]` and `ats_identity(...)["ats_identity_hash"]` in the evaluator identity.
2. Call `evaluate_ats(OfflineATSProvider(), ...)` from `_evaluate_output` with the pair's JD keyword families, passing `generation_error` for failed generations; store the record and add it to checkpoint-completeness checks.
3. Add `summarise_ats` output to `eval/reporting.py` with the eligible/scored/unscored/failed columns.
4. Add an `agreement` (or `report --evaluations`) CLI path that calls `calculate_completed_forms(review_dir, evaluations=<evaluation rows>)` and surfaces `judge_vs_human_status` in reports; never present `insufficient_labels` results as calibration.
5. Confirm evaluation rows carry `variant` consistently so `_link_evaluations` can match robustness outputs.
6. Add CLI/integration tests for identity changes and resume behaviour.

## Blockers and limitations

- No completed human labels exist, so judge-versus-human agreement cannot be reported yet. Collecting independent ratings from at least two reviewers is required.
- Review packets written before this change lack the new mapping fields; linking them fails closed. Regenerate the packet before reviewers start.
- The ATS proxy is a transparent heuristic, not a commercial ATS score, and does not assess visual layout or PDF/DOCX parsing.
- JD keyword files are requirement-family proxies, so supported coverage measures labelled must-have families only.

## Exact resume instructions

```bash
cd /tmp/arb-eval-human-ats
git status --short --branch
git log --oneline --decorate -5
sed -n '1,260p' docs/evaluation/workstreams/human-ats/HANDOVER.md
sed -n '1,240p' docs/evaluation/workstreams/human-ats/CHECKPOINTS.md
PYTHONDONTWRITEBYTECODE=1 /tmp/arb-eval-venv/bin/pytest -q -p no:cacheprovider tests/test_ats.py tests/test_human_agreement.py tests/test_reporting.py
```

For integration, cherry-pick this branch's commits in order, then complete the pending integration steps without altering the frozen dataset.
