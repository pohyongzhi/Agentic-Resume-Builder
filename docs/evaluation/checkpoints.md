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
