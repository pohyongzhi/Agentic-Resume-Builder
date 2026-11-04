# Team evaluation plan

Each person evaluates their own workflow's 100 resumes with their own Groq key
(free tier: about 200k tokens a day per model, 8k tokens a minute). Everyone
evaluates the same shared outputs with the same settings, so the results can be
merged into one evaluation.

## What each workflow runs

| Eval | Model | Scope per workflow | Calls | Tokens |
|---|---|---|---|---|
| Claim faithfulness, unsupported claims, protected fields, fabrication bait, keyword coverage, similarity gain, required sections, length, task success, operational cost | local (NLI on CPU, embeddings in Ollama) | all 100 | 0 | 0 |
| Requirement coverage | gpt-oss-20b | all 100 | about 50–80 (only when keyword matching is not enough) | about 150–250k of the 20b allowance |
| Rubric judge (calibrated) | gpt-oss-120b | 30 pairs (`eval_plan/rubric30.txt`) | 30 | about 113k |
| Pairwise vs W2, both orders, with position-swap consistency | gpt-oss-120b | 10 pairs (`eval_plan/pairwise10.txt`) | 20 | about 73k |
| **Total on gpt-oss-120b** | | | **50** | **about 186k** |

Token figures are measured from a dry run on 2026-11-04: a rubric call used
3,769 tokens and a pairwise call about 3,650. Retries add a little; every step
is resumable, so anything cut off by the daily limit continues the next day
with the same command. W2 is the baseline and has no pairwise step (about 113k).

`rubric30` holds 6 pairs per job category, 2 per fit level. `pairwise10` is a
subset of it with 2 pairs per category (4 strong, 3 moderate, 3 weak).

## Who runs what

| Day | Person | Workflow |
|---|---|---|
| 1 | Ethan | W4 evidence-grounded |
| 1 | Yong Zhi | W2 agentic baseline |
| 1 | Kester | W5 fine-tuned |
| 1 | Leonard | W6 best-of-N |
| 1 | Jaxs | W7 plan-and-execute |
| 2 | Ethan | W1 single pass |
| 2 | Yong Zhi | W3 multi-critic |

## Setup (once)

```bash
git pull                                   # shared outputs and the evaluation code
pip install -r requirements.txt
echo "GROQ_API_KEY=<your own key>" > .env  # never commit or share it
ollama pull nomic-embed-text               # used for the similarity metric
```

## Commands

Replace `<workflow>` with yours: `w1_single_pass`, `w2_baseline`, `w3_critic`,
`w4_evidence`, `w5_fine_tuned`, `w6_best_of_n` or `w7_plan_execute`.

```bash
scripts/team_eval.sh <workflow> free       # local metrics + coverage (20b)
scripts/team_eval.sh <workflow> rubric     # rubric judge on 30 pairs (120b)
scripts/team_eval.sh <workflow> pairwise   # vs W2 on 10 pairs (120b); W2 skips this
scripts/team_eval.sh <workflow> status     # progress of each step
```

If a step stops on a rate limit, run the same command again later; finished
work is kept and only what is missing is retried.

## Hand-back

Zip `results/evaluations/` and share it. The folders from all seven workflows
are merged into one evaluation for the report and the dashboard.
