# W4 – Evidence-grounded generation (owner: Ethan)

> Every bullet must cite a span of the source resume text. A verifier checks each claim against
> its cited span, and the harness rejects claims without a citation.

This folder is the full brief for building W4 with a coding agent (Claude Code, Codex, etc.).

| File | What it is |
|---|---|
| `01-spec.md` | What W4 does, how it plugs into the existing repo, every design rule and why |
| `02-implementation-plan.md` | Ordered phases with exact files, functions, signatures and "done when" checks |
| `03-test-plan.md` | Every test case to write (fake model, no Ollama needed), with fixtures |
| `04-eval-and-demo.md` | What W4 records for evaluation, the demo, and deep-dive explanations for the presentation |

## How to use this with a coding agent

1. Copy this folder into the repo at `docs/w4-evidence-grounded/` on a new branch:
   ```bash
   git checkout main && git pull
   git checkout -b feature/w4-evidence-grounded
   ```
2. Open the repo in Claude Code / Codex and paste the prompt below.
3. Review each phase's diff and run `pytest -q` yourself before moving on.
4. When all phases pass, try it for real with Ollama running:
   ```bash
   python run_all.py --workflows w4_evidence --limit 1 --repeats 1
   python app.py      # pick "W4 - Evidence-grounded" in the dropdown
   ```

## Prompt to paste into the coding agent

```text
You are implementing workflow W4 (evidence-grounded generation) in this repo.

Read these first, fully, before writing code:
- docs/w4-evidence-grounded/01-spec.md
- docs/w4-evidence-grounded/02-implementation-plan.md
- docs/w4-evidence-grounded/03-test-plan.md
- The existing code they reference: common/schemas.py, common/harness.py, common/pipeline.py,
  common/llm.py, common/prompts.py, workflows/w2_baseline.py, workflows/w3_critic.py,
  tests/conftest.py, tests/test_w3_critic.py, run_all.py, app.py.

Rules:
- Implement one phase at a time, in order. After each phase run `pytest -q`; all tests (old and new)
  must pass before starting the next phase. Show me a short summary of each phase's changes.
- Only create or edit the files listed in 02-implementation-plan.md. Do NOT change shared prompts
  (common/prompts.py), config.py, common/harness.py, common/pipeline.py, eval/, or other workflows:
  other teammates' results depend on them staying identical. The one allowed shared edit is adding
  two optional keys (evidence, w4_stats) to ResumeState in common/state.py, as the plan describes.
- All model calls must go through common.llm.chat so tokens and time are tracked.
- Tests must use fake model responses (monkeypatch the module's `chat`), never a real Ollama call.
- Never use eval.metrics.faithfulness (or its NLI model) inside the workflow. That is the
  evaluator's referee; using it inside W4 would optimise for the test.
- Use fictional data in tests and fixtures only. No real names, phone numbers or emails.
- If something in the spec conflicts with the current code, stop and tell me instead of guessing.

Start with Phase 1.
```
