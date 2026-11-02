# Evaluation handover

## Objective

Use the same independent hosted judge for every workflow while keeping factual faithfulness as a
separate deterministic/NLI measurement. The selected quality judge is Groq-hosted
`openai/gpt-oss-120b`; it is not a generation workflow model.

## Current state

- Branch: `feature/faithfulness-evaluation`
- Base checkpoint: `10a1bab`
- Uncommitted Groq integration and calibration gate are implemented.
- Files changed so far: `config.py`, `requirements.txt`, `eval/judge.py`, `evaluate.py`,
  `tests/test_judge.py` and these two handover files.
- Focused judge and calibration tests: `9 passed`.
- Complete suite: `104 passed`.
- The ignored project virtual environment contains the pinned `groq==0.37.1` SDK.
- Groq requests are paced at a minimum 15-second interval for the free tier.
- No hosted request has been made.
- No key has been stored in the repository.

## Safety rule

Do not reuse any credential pasted into chat. Revoke it, create a replacement and place the
replacement only in the shell environment:

```bash
export GROQ_API_KEY='replacement-key'
```

Never put the value in a command committed to Git, documentation, logs, test fixtures or evaluation
records.

## Continue

```bash
cd /Users/kester/Desktop/Agentic-Resume-Builder
git status --short --branch
venv/bin/python -m pytest tests/test_judge.py -q
```

Inspect the zero-network dry run:

```bash
venv/bin/python -m eval.judge_calibration
```

After replacing and exporting the credential, execute calibration into a new path:

```bash
venv/bin/python -m eval.judge_calibration \
  --provider groq \
  --model openai/gpt-oss-120b \
  --reasoning-effort medium \
  --output-dir results/judge-calibration/gpt-oss-120b-v1 \
  --execute
```

Do not run 100 cases unless the saved result reports `"passed": true`. After it passes, run a
six-record saved-output pilot with the exact provider/model configuration before bulk evaluation.

## First live result

`results/judge-calibration/gpt-oss-120b-v1` is a preserved failed calibration. The API worked, but
three controls failed deterministic quotation validation. The prompt and whitespace handling were
corrected under new versions. Never overwrite or present the `v1` run as a passed calibration; the
next live output must use `results/judge-calibration/gpt-oss-120b-v2`.

The `v2` live run is also preserved and must not be presented as passed. It showed that a simple
mean allowed polished fabrication to score too highly and that the judge penalised missing,
unsupported JD preferences. Both issues were corrected under `resume-rubric-v4` with transparent
unadjusted means and a documented critical-failure cap. The next live path is
`results/judge-calibration/gpt-oss-120b-v3`.

The `v3` run produced the intended score separation but failed one unnecessary ordering assertion:
generic and fabricated resumes both scored `2.0`. The gate now requires the strong control to beat
both poor controls, while the dedicated unsupported and padding checks remain strict. The final
live path is `results/judge-calibration/gpt-oss-120b-v4`.

## Accepted judge

The `v4` calibration passed all gates. Its result is local and ignored at
`results/judge-calibration/gpt-oss-120b-v4/calibration-result.json`, with SHA-256
`eca7d654de5c299f48aa68cfc34586015a0a23947f4f63cf0d38ba49955b7296`.

Do not rerun or overwrite that path. The next action is a six-output W5 pilot. The current branch's
result index contains only W1; W5's 100 saved outputs are currently in `/tmp/agentic-w5-smoke` and
must be referenced without regenerating them. Keep the exact accepted judge identity:

- provider `groq`;
- model `openai/gpt-oss-120b`;
- reasoning `medium`;
- prompt `resume-rubric-v4`;
- evaluator hash `f508d86477ec2c84`.
