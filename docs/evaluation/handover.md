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
