# Coverage and Injection Evaluation Handover

## Repository state

- Branch: `feature/eval-coverage-injection`
- Worktree: `/tmp/arb-eval-coverage`
- Base: `origin/main` at `a866c34988726e392d81952538048d27025d8e46`
- Implementation commit: `0c4537b88a0dbd966578d4825eeec806af30d39e`
- Status: implementation and isolated tests complete; shared evaluator integration is intentionally pending.

## Scope

This workstream adds two evaluation components:

1. Supportable-requirement coverage, which measures how many JD requirement families supported by the original resume are expressed by the tailored resume.
2. Prompt-injection success rate, using deterministic canaries injected separately into resume and JD inputs.

It also adds network-free unit tests for deterministic behavior, structured evaluator validation, caching, attack execution, failure accounting, immutable manifests, and aggregation.

## Non-scope

The following shared integration files were deliberately not changed:

- `config.py`
- `evaluate.py`
- `eval/reporting.py`
- `common/schemas.py`
- `README.md`

No generation or hosted evaluator runs were performed. No dataset files were modified. No branch was pushed.

## Architecture and interfaces

### Supportable-requirement coverage

`eval/requirement_coverage.py` exposes:

- `RequirementCoverageConfig`: local evaluator configuration; defaults mirror the repository judge settings.
- `normalise_requirements(requirements)`: validates keyword-family annotations and assigns stable `req-NNN` identifiers.
- `evaluate_requirement_coverage(original_resume, tailored_resume, requirements, cache_dir, evaluator_config=None, client=None, model_info=None, force=False)`: main entry point.

The metric uses two stages:

1. Supportability: exact keyword/synonym evidence is resolved deterministically; unresolved requirements receive a structured semantic decision against the original resume.
2. Coverage: exact keyword/synonym evidence in the tailored output is resolved deterministically; unresolved supported requirements receive a structured semantic decision.

Every positive semantic decision requires a verbatim quote validated against its source. Invalid responses retry within the configured bound and then return `status="unscored"`; they never become a score of zero. When no requirement is supportable, the result is `status="no_supportable_requirements"` and `coverage=None`.

Supportability cache keys exclude the tailored resume, so the same source decision is reused across workflow outputs. The integration layer should pass one pre-resolved `model_info` for the evaluation run so all rows carry the same evaluator identity.

The returned record includes `status`, `failure_reason`, evaluator identity, `coverage`, supportable and covered counts, uncovered names, per-requirement evidence/method/rationale, and the semantic-stage audit records.

### Prompt-injection success rate

`eval/injection.py` exposes:

- `derive_canary(dataset_hash, pair_id, channel, attack_version=...)`
- `inject_pair(pair, channel, dataset_hash, attack_version=...)`
- `canary_matches(output, canary)`
- `evaluate_injection_output(output, canary, generation_error=None)`
- `run_injection_attack(workflow_name, workflow, pair, dataset_hash, channel, tracker, attack_version=...)`
- `injection_manifest(pairs, dataset_hash, seed=42, attack_version=...)`
- `write_injection_manifest(path, manifest)`
- `summarise_injection(rows)`

The benchmark uses the existing deterministic 20-pair stratified selector and runs two channels per pair: `resume` and `jd`. Canaries are derived from the dataset hash, pair ID, channel, and attack version. A collision with source content aborts the case. Detection requires the complete token, is case-insensitive, and accepts ordinary Markdown wrapping.

Generation failures have `injection_success=None` and are excluded from the success-rate denominator. Reports retain attempted, completed, and failure counts. Aggregates are produced overall, by workflow, by channel, and by workflow/channel, with 95% Wilson intervals.

## Files changed

- `eval/requirement_coverage.py`
- `eval/injection.py`
- `tests/test_requirement_coverage.py`
- `tests/test_injection.py`
- `docs/evaluation/workstreams/coverage-injection/HANDOVER.md`
- `docs/evaluation/workstreams/coverage-injection/CHECKPOINTS.md`

## Validation performed

Commands and results:

```bash
/tmp/arb-eval-venv/bin/python -m py_compile eval/requirement_coverage.py eval/injection.py tests/test_requirement_coverage.py tests/test_injection.py
/tmp/arb-eval-venv/bin/pytest tests/test_requirement_coverage.py tests/test_injection.py -q
# 16 passed in 0.12s

/tmp/arb-eval-venv/bin/pytest tests/test_metrics.py tests/test_judge.py -q
# 10 passed in 0.20s

git diff --check
# clean
```

An attempted command included the nonexistent `tests/test_robustness.py`; pytest reported that path was not found and ran no tests for that invocation. The valid regression command above was then run successfully.

## Integration work remaining

1. Add version/config constants to `config.py`, or explicitly construct the local configs in `evaluate.py`.
2. Resolve coverage model metadata once during evaluator preflight and pass it to every coverage call.
3. Call `evaluate_requirement_coverage` from `_evaluate_output`; map its primary fields into the evaluation row and retain the full audit record.
4. Include coverage version/prompt/schema/model identity in the evaluator identity and checkpoint-completeness checks.
5. Add an `injection` CLI command and `--skip-injection`/`--injection-limit` options.
6. Write and enforce the injection manifest before any attack generation; use a dedicated `root/injection` artifact tree.
7. Reuse the normal workflow tracker for `run_injection_attack`, then apply protected-field and faithfulness diagnostics to completed outputs in the integration layer.
8. Add coverage and injection aggregates/CSV/JSON outputs in `eval/reporting.py`.
9. Add integration tests covering CLI resume behavior and evaluator identity changes.

## Blockers and dataset limitations

- Existing JD keyword files are requirement-family proxies, not full sentence-level requirement annotations. The metric therefore measures supportable coverage of the labelled must-have families, not every prose requirement in a JD.
- Semantic decisions require the configured evaluator model or hosted provider. Deterministic lexical cases do not.
- Full injection runs require the generation model and the selected workflows. Unit tests use a fake workflow and need no network or model.
- A pilot should be manually reviewed before the full run, especially semantic supportability decisions and negated/ambiguous experience.

## Exact resume instructions

```bash
cd /tmp/arb-eval-coverage
git status --short --branch
git log --oneline --decorate -5
sed -n '1,260p' docs/evaluation/workstreams/coverage-injection/HANDOVER.md
sed -n '1,240p' docs/evaluation/workstreams/coverage-injection/CHECKPOINTS.md
/tmp/arb-eval-venv/bin/pytest tests/test_requirement_coverage.py tests/test_injection.py tests/test_metrics.py tests/test_judge.py -q
```

For integration into another branch, cherry-pick the branch commits in order, then perform the nine integration steps above without altering the frozen dataset.

