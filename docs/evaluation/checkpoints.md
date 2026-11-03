# Evaluation checkpoints

This file records resumable checkpoints for the evaluation workstream. Generated evaluations and
credentials remain outside Git.

## 2026-11-02 — Groq judge integration

Status: implementation and zero-network verification complete; live calibration pending.

- Working branch: `feature/faithfulness-evaluation`.
- Starting commit: `10a1bab`.
- Added a selectable `groq` judge provider while preserving the existing `ollama` provider.
- Approved hosted judge IDs are `openai/gpt-oss-120b` and `openai/gpt-oss-20b`.
- The recommended primary judge is `openai/gpt-oss-120b` with medium reasoning.
- Groq requests use strict JSON Schema output, no tools, temperature zero and a hard completion cap.
- Records include the response ID, returned model, system fingerprint and reported prompt, output,
  reasoning and total token counts.
- API credentials are read only from `GROQ_API_KEY`; they are never written to evaluation records.
- Hosted requests are spaced by 15 seconds by default to respect the Groq free-tier token limit.
- Error messages redact both the current environment value and strings matching Groq key syntax.
- Rubric prompt version advanced to `resume-rubric-v2`. It requires evidence before awarding scores,
  treats 5 as rare and penalises unsupported JD copying, vague claims and negative framing.
- Relevance must cite both the JD and tailored resume. Prioritisation must cite both the original and
  tailored resume. Remaining dimensions must cite the tailored resume.
- Focused result: `7 passed` in `tests/test_judge.py`; all Groq tests use a fake client and make zero
  network calls.
- Added `eval/judge_calibration.py` with four controls: strong, generic, unsupported and padded.
- The calibration fails unless the strong example scores at least 4.0, the generic example stays at
  or below 3.4, unsupported qualifications receive a clear penalty, the quality ordering resolves,
  and padding lowers concision without improving the overall score.
- Calibration is dry-run by default, requires `--execute` for hosted requests and refuses to
  overwrite an earlier `calibration-result.json`.
- Focused judge and calibration result: `9 passed`; dry-run reports exactly four requests.
- Complete repository result after integration: `104 passed`.
- Installed `groq==0.37.1` only in the ignored project virtual environment and pinned that exact
  version in `requirements.txt`.
- A credential pasted into chat was treated as compromised and was not used or stored. Revoke it and
  export a replacement only in the terminal before the live calibration.

Next checkpoint: run the complete test suite, install the pinned Groq SDK in the ignored virtual
environment, then run live calibration only with a replacement environment credential.

## 2026-11-02 — First live calibration

Status: failed closed; preserved at `results/judge-calibration/gpt-oss-120b-v1`.

- Groq connectivity and `openai/gpt-oss-120b` inference succeeded.
- Four controls caused ten requests including retries; only the unsupported control was scored.
- The unsupported control scored `2.8/5`, including `2/5` for prioritisation, so the judge did
  penalise fabricated experience as intended.
- The other controls were unscored because the model sometimes omitted a required source quotation,
  changed one quotation's tense, or removed a Markdown line break from an otherwise exact quote.
- This exposed two integration defects: the prompt did not state the per-dimension source
  requirements, and exact quotation validation treated layout whitespace as semantic text.
- Corrected both defects, added deterministic retry feedback, retained token metadata for invalid
  responses and advanced to `resume-rubric-v3` / `resume-judge-calibration-v2`.

Next checkpoint: run zero-network regression tests and rerun live calibration at a new `v2` path.

## 2026-11-02 — Second live calibration

Status: schema integration passed; quality calibration failed and is preserved at
`results/judge-calibration/gpt-oss-120b-v2`.

- All four controls received valid structured scores in exactly four requests.
- Strong scored `3.6`, generic `2.4`, unsupported `3.8` and padded `2.6` before the new critical
  failure rule.
- The judge correctly gave unsupported experience `1/5` for prioritisation, but its simple mean was
  inflated by `5/5` scores for clarity, concision and organisation.
- The judge incorrectly reduced the strong control because it lacked preferred ERP and PMP
  qualifications that were absent from the original resume.
- Clarified that unsupported JD requirements must not reduce relevance for a fact-preserving resume.
- Added a declared critical-failure cap: prioritisation `1/5` caps overall quality at `2.0`, and
  `2/5` caps it at `3.0`. The unadjusted arithmetic mean remains recorded for transparency.
- Advanced to `resume-rubric-v4` / `resume-judge-calibration-v3` rather than weakening the gate.

Next checkpoint: run regression tests and a new live calibration at the `v3` path.

## 2026-11-02 — Third live calibration

Status: all substantive checks passed; preserved at `results/judge-calibration/gpt-oss-120b-v3`.

- Strong scored `5.0`, generic `2.0`, unsupported `2.0` and padded `3.8`.
- Strong-quality, generic-quality, unsupported-claim and padding checks all passed.
- The sole failed assertion required generic to score strictly above unsupported. Both correctly
  entered the same critical-failure band, so their relative order does not measure judge quality.
- Corrected the ordering gate to require strong to beat both failure types. The independent maximum
  scores and unsupported-prioritisation threshold remain unchanged.
- Advanced the calibration implementation to `resume-judge-calibration-v4`; rubric and scoring
  rules were not relaxed.

Next checkpoint: run regression tests and the final live calibration at the `v4` path.

## 2026-11-02 — Accepted live calibration

Status: passed and preserved at `results/judge-calibration/gpt-oss-120b-v4`.

- Calibration implementation: `resume-judge-calibration-v4`.
- Judge: Groq `openai/gpt-oss-120b`, medium reasoning, rubric `resume-rubric-v4`.
- All six gates passed: complete scoring, strong quality, generic penalty, unsupported-claim penalty,
  strong/failure separation and padding penalty.
- Scores: strong `5.0`, generic `2.0`, unsupported `2.0`, padded `3.8`.
- Usage: six requests including two corrected retries, 9,111 prompt tokens, 7,349 output tokens,
  5,302 reported reasoning tokens and 16,460 total tokens.
- Result SHA-256:
  `eca7d654de5c299f48aa68cfc34586015a0a23947f4f63cf0d38ba49955b7296`.
- Generated calibration records are ignored under `results/judge-calibration/`.
- The current branch's `results/results.jsonl` contains only 100 W1 records. The existing W5
  100-output set remains in `/tmp/agentic-w5-smoke`; no bulk Groq evaluation was started.

Next checkpoint: run a six-output W5 pilot using the accepted judge identity, review score spread
and retry rate, then decide whether free-tier capacity is sufficient for the 100-output run.
