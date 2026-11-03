# Evaluation metrics integration handover

## Resume here

- Branch: `feature/eval-metrics-integration`
- Worktree: `/tmp/arb-eval-integration`
- Base: `origin/main` at `a866c34`
- Global context: `/mnt/c/Users/keste/Downloads/CLAUDE.md`
- Status: the tier-1 evals are wired, tested and runnable end to end through `evaluate.py`.
  No model-backed pilot or full run has been made. Nothing has been pushed.

```bash
cd /tmp/arb-eval-integration
git status --short --branch
git log --oneline --decorate -10
sed -n '1,400p' docs/evaluation/workstreams/integration/HANDOVER.md
sed -n '1,200p' docs/evaluation/workstreams/integration/CHECKPOINTS.md
PYTHONDONTWRITEBYTECODE=1 /tmp/arb-eval-venv/bin/pytest -q -p no:cacheprovider
```

## Priorities

| Tier | Eval | State on this branch |
| --- | --- | --- |
| 1 | Faithfulness (NLI, protected fields, fabrication bait) | On main; part of `score` |
| 1 | Supportable-requirement coverage | Wired per output in `score` |
| 1 | Pairwise vs `w2_baseline` with position swap | Wired as the `pairwise` stage (in `all`) |
| 1 | Operational metrics | On main; part of `score` |
| 2 | Rubric judge integration checks | Not started; the judge on main is unchanged |
| 2 | Keyword/similarity reporting | Not started; existing rows in `overall.csv` unchanged |
| 3 | ATS proxy | Code merged (`eval/ats.py`), **not wired** into CLI or reports |
| Dropped | Human agreement | Code merged (`eval/human_review.py`), **not wired** |
| Optional | Prompt injection | Wired and tested as the `injection` stage; **not in `all`** |

## Merged branches

| Workstream | Branch / head | Merge commit | Note |
| --- | --- | --- | --- |
| Requirement coverage and injection | `feature/eval-coverage-injection` / `42a4668` | `7bbe174` | |
| Pairwise and position swapping | `feature/eval-pairwise-position` / `174463c` | `71d000d` | |
| Human agreement and ATS | `feature/eval-human-ats` / `3c667f8` | `e099d25` | Clean merge, all its tests pass; code kept, not wired |

Integration commits: `84809af`, `f63cba0`, `95f7b4e` (docs), `cea794e` (tier-1 focus), plus the
commit containing this file.

## Required behaviour and how it is met

- Evaluation failures stay unscored: each metric writes `None` plus a reason in
  `evaluation_failures[<metric>]`; reports count them as `unscored`/`failed`, never as zero.
- Cache identity: per-output components add `identity()` to
  `evaluator_identity.json["output_components"]`, so they are part of the evaluator hash (the
  results folder). Each component's cache key also hashes its inputs. Pairwise records live under
  `pairwise/<pairwise evaluator_hash>/`; its cache key hashes the four texts, both workflows and
  config hashes, the full judge config, model provider/tag/digest and prompt/schema/rubric hashes.
- Generation artifacts and the frozen dataset are untouched (`git diff origin/main -- data results`
  is empty; `validate_dataset()` still returns `3bb1861e…6f81`).
- Explicit `eligible`/`scored`/`unscored`/`failed` denominators in every new table.
- No network in tests: fake clients and workflows; tier-1 and pairwise tests block sockets.
- Thresholds and historical results unchanged. `EVALUATOR_SCHEMA_VERSION` moved `2.2 -> 2.3`
  because records gained fields; older results folders stay as historical artifacts.

## Tier-1 CLI

Run from the repository root that holds `results/results.jsonl` and the saved run files.

Options that matter for tier 1:

- `--pair-ids P1 P2 ...` / `--pair-ids-file FILE` (one ID per line, `#` comments allowed): score
  and pairwise-judge only those dataset pairs. Results go to the same folder as a full run;
  later subsets add to the indexes and reports instead of replacing them, and completed records
  are never re-judged. Use this for the Groq free tier.
- `--config-hash WORKFLOW=HASH` (repeatable): keep only saved generations with that config hash.
  Required for `w3_critic` (see blockers) and also applied to pairwise pairing.
- `--skip-judge`: skips the tier-2 rubric judge. Coverage and pairwise still use the
  `--judge-provider`/`--judge-model` model (coverage can be overridden with
  `--coverage-provider`/`--coverage-model`).
- `--skip-robustness`: robustness re-generates perturbed inputs with Ollama; not tier 1.
- `--pairwise-baseline` (default `w2_baseline`), `--pairwise-challengers`, `--pairwise-limit N`
  (first N pair IDs in sorted order), `--limit N` (first N saved rows for `score`).

The `--skip-judge`, `--skip-coverage`, judge and coverage model options change the evaluator
hash, so keep them identical across `score`, `pairwise` and `report` calls for one results folder.
The selected `--workflows` (and `--config-hash`) define the generation scope in the folder path,
so always pass `w2_baseline` together with the challengers.

One challenger workflow on a fixed subset (Groq):

```bash
export GROQ_API_KEY=...
python evaluate.py all --workflows w2_baseline w1_single_pass \
  --judge-provider groq --judge-model openai/gpt-oss-120b \
  --skip-judge --skip-robustness \
  --pair-ids bt_resume_001__bt_jd_01 bt_resume_002__bt_jd_01 cd_resume_002__cd_jd_01
```

All workflows:

```bash
python evaluate.py all --workflows w2_baseline w1_single_pass w3_critic w6_best_of_n \
  --config-hash w3_critic=b884fe8fe066 \
  --judge-provider groq --judge-model openai/gpt-oss-120b \
  --skip-judge --skip-robustness \
  [--pair-ids-file pilot_pairs.txt]
```

Stage by stage (same flags): `score` (faithfulness, coverage, operational), `pairwise`, then
`report` to rebuild `summaries/` from saved indexes without any model call.

Without `GROQ_API_KEY`, use the default Ollama judge (`--judge-provider ollama --judge-model
gemma4:e4b`). Faithfulness needs `sentence-transformers` and the NLI model, or
`--skip-faithfulness`.

### Expected hosted calls

`JUDGE_MAX_RETRIES = 2`, so one logical call is at most three requests.

- Coverage, per output: **0 to 2 calls**, usually 0-1. Requirements with an exact keyword or
  synonym match are decided lexically (no call). A supportability call runs only for requirement
  families with no lexical evidence in the original resume, and it is cached per original
  resume and requirement set, so it is shared by every workflow output for that pair. A coverage
  call runs only when a supportable requirement is not lexically present in the output.
  Worst case 6 requests with retries.
- Pairwise, per challenger output: **exactly 2 calls** (challenger first, then baseline first).
  The baseline output itself costs nothing extra. Worst case 6 requests with retries.
  Generation failures and missing outputs cost 0 calls.
- Rubric judge (tier 2, skipped with `--skip-judge`): 1 call per output, worst case 3.
- All hosted calls share one process-wide timer (`eval.judge.wait_for_hosted_request_slot`) with
  `JUDGE_MIN_REQUEST_INTERVAL_SECONDS = 15`. Example: 3 pairs x (w2 + one challenger) is about
  6 outputs x 0-2 coverage calls + 3 x 2 pairwise calls = 6-18 calls, about 1.5-4.5 minutes.
  All 100 pairs x 3 challengers is at least 600 pairwise calls (about 2.5 hours) plus coverage.

## Outputs

Under `results/evaluations/<dataset_hash>/<generation_scope>/<evaluator_hash>/`:

- `records/<workflow>/<pair_id>/repeat_<n>.json`, `evaluations.jsonl`: per-output rows with
  faithfulness, coverage and operational fields.
- `cache/requirement_coverage/`, `cache/nli/`: evaluator caches.
- `pairwise/<pairwise_hash>/records/<challenger>/<pair_id>/repeat_<n>.json`,
  `pairwise/<pairwise_hash>/evaluations.jsonl` (compact: no prompts or raw attempts),
  `pairwise_cache/` (full records).
- `summaries/`: `overall`, `by_category`, `by_fit_level` (existing, now with a
  `requirement_coverage` row), `requirement_coverage`, `requirement_coverage_by_fit_level`,
  `pairwise` (by baseline/challenger) and `pairwise_by_fit_level`, each as CSV and JSON.

## Tier-1 metric semantics

### Requirement coverage

Record fields: `requirement_coverage`, `requirement_coverage_status` (`scored`, `unscored`,
`no_supportable_requirements`, `generation_failed`, `evaluation_error`), supportable/covered
counts, uncovered list, evaluator version/hash, seconds, and the full `requirement_coverage_record`.

- `no_supportable_requirements` is not applicable: excluded from `eligible`, not a failure.
- `evaluation_error` (evaluator unreachable or rate limited) is not cached and is retried on the
  next run. Invalid evaluator output is cached and final for that identity, like the judge.
- The report gives macro mean with bootstrap CI and micro coverage (covered / supportable).
- Limitation: JD keyword files are requirement-family proxies, so this measures the labelled
  must-have families, not every prose requirement.

### Pairwise vs W2

- Each comparison is judged twice with positions swapped; a winner exists only when both orders
  agree. Disagreement is reported as position-swap inconsistency.
- Resume: `scored` and `unscored`+`invalid_output` are final; `judge_failed` is retried;
  `generation_unavailable` (missing or failed output on either side) is recomputed without a call.
- Report columns: `total_pairs`, `generation_unavailable`, `eligible`, `scored`, `unscored`,
  `failed`, `invalid`, strict win/loss rates with Wilson CIs, position-neutral score, swap
  consistency and first-position selection rate.

### Faithfulness and operational

Unchanged from main; computed in `score` for every saved output.

## Extension points

- Per-output metric: implement `OutputComponent` (`name`, `identity()`, `is_complete(record)`,
  `evaluate(record, generation_row, pair, tailored_resume, root, force)`) in `evaluate.py` and
  return it from `build_output_components`. Identity, checkpoint completeness and evaluation
  pick it up automatically. Tier-3 ATS would plug in here.
- Run-level stage: add a `Stage(name, run, index, pilot_limit, help, loader=None, in_all=True)`
  to `STAGES`; the parser adds the command, `--skip-<name>` and `--<name>-limit`.
- Reports: `eval.reporting.STAGE_REPORTERS[name] = writer(output_dir, rows)`, or add a scalar
  field to `METRICS`.
- Constants: all workstream versions, thresholds and weights are in `config.py` (values unchanged),
  including the unwired ATS and agreement ones.

## Optional, deferred and dropped

- Prompt injection (`python evaluate.py injection --workflows ... --injection-limit N`): working
  and tested, writes its manifest before generation, resumable. Not in `all`. It refuses to run
  until blocker 1 is resolved because it generates with the current workflow code.
- ATS proxy (tier 3): `eval/ats.py` and `tests/test_ats.py` are merged; to wire it, add an
  `OutputComponent` calling `evaluate_ats(OfflineATSProvider(), ...)` and an `ats` report from
  `summarise_ats`.
- Human agreement (dropped): `eval/human_review.py` additions and `tests/test_human_agreement.py`
  are merged but not called. The pre-existing blinded review packet from main is still written
  by `all`/`report`; it is now left untouched once any reviewer label exists.

## Validation

```bash
cd /tmp/arb-eval-integration
PYTHONDONTWRITEBYTECODE=1 /tmp/arb-eval-venv/bin/pytest -q -p no:cacheprovider
# 446 passed, 21 subtests passed in 11.39s (full suite, including tests/test_app.py)
PYTHONDONTWRITEBYTECODE=1 /tmp/arb-eval-venv/bin/pytest -q -p no:cacheprovider tests/test_eval_integration.py
# 25 passed (includes the tier-1 end-to-end CLI test on a pair subset)
git diff --check                          # clean
git diff --stat origin/main -- data results   # empty
```

`gradio` was installed into `/tmp/arb-eval-venv` so `tests/test_app.py` runs; nothing is excluded.

## Blockers

1. Generation identity drift: current `run_all.config_hash` values (`w1 2f00868a0ee3`,
   `w2 3fdcd18565b5`, `w3 9ab32afb5c88`, `w6 7bb3bef9a553`) differ from the saved rows
   (`w1 84c72812e449`, `w2 c14c2c8de42d`, `w3 b884fe8fe066` plus two partial reruns,
   `w6 26a928dd00bd`). Tier 1 is not affected because it only reads saved outputs. Only the
   optional injection stage is blocked.
2. `w3_critic` has 98 rows under `b884fe8fe066` and 6 rerun rows under two other hashes; two
   `(pair_id, repeat)` keys are duplicated across them. Pass `--config-hash
   w3_critic=b884fe8fe066`; its two missing pairs then appear as `generation_unavailable` in
   pairwise and are excluded from rates.
3. `sentence-transformers` is not in the test venv; NLI runs need the real environment.

## Next steps

1. Run the one-workflow Groq command above on 3 pairs; manually audit coverage semantic
   decisions (`requirement_coverage_record`), pairwise consistency and invalid-output rates.
2. Extend to all workflows on a fixed pilot subset, then the approved full run.
3. Tier 2: rubric judge integration checks and keyword/similarity reporting.
4. Push only with explicit approval.
