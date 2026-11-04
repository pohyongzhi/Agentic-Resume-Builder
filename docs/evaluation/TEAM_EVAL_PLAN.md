# Team evaluation plan

Each person evaluates their own workflow's 100 resumes with their own Groq key
(free tier: about 200k tokens a day per model). Everyone evaluates the same
shared resumes with the same settings, so the results merge into one evaluation.

## One command

```bash
git pull
python scripts/team_eval.py <your_workflow>
```

It runs everything due for your workflow, shows progress, and packs the results
into `results/handback_<workflow>.zip`: send that file back. If the daily limit
runs out, run the same command again tomorrow; finished work is kept.

Setup once: `pip install -r requirements.txt`, put `GROQ_API_KEY=<your key>` in
`.env` (never commit it), and have Ollama running with `nomic-embed-text`.

## What it runs

| Eval | Model | Scope | Calls | Tokens |
|---|---|---|---|---|
| Faithfulness, unsupported claims, protected fields, invented qualifications, keyword coverage, similarity, sections, length, cost | local | all 100 | 0 | 0 |
| Requirement coverage | gpt-oss-20b | all 100 | about 50–80 | about 150–250k (20b) |
| Rubric judge | gpt-oss-120b | 30 pairs | 30 | about 113k |
| Pairwise vs W2 + position swap (top 3 only, day 2) | gpt-oss-120b | 20 pairs | 40 | about 146k |

## Plan

| Day | Who | What |
|---|---|---|
| 1 | Ethan W4, Yong Zhi W2, Kester W5, Leonard W6, Jaxs W7 | the command above |
| 2 | Ethan W1, Yong Zhi W3 | the command above |
| after day 2 | coordinator | merge results, run `python scripts/select_pairwise.py <merged results/evaluations>`, push `eval_plan/pairwise_selected.txt` |
| 3 | owners of the 3 selected workflows | `git pull`, then the same command (it now adds pairwise) |

Selection rule, fixed in advance: among challengers with mean claim
faithfulness of at least 0.9, the three with the highest mean rubric score.
Pairwise uses the same fixed 20 pairs for every selected workflow.
