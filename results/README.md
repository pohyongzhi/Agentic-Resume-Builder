# Results

Start with [latest.json](latest.json). It points to the current saved evaluations for W1, W2 and W3 and the runtime summary. Completion of an evaluation batch does not mean every rubric judgment succeeded; check scored and unscored counts in its summaries.

- `evaluations/` — output scores, claim evidence, judge responses and evaluation summaries.
- `runs/` — generated resumes, generation metrics and execution traces.
- `outputs/` — resumes saved by the application.
- `recovered/` — retained drafts from failed generation runs.
- `judge-calibration/` — judge calibration examples and scores.
- `logs/` — one log per workflow, such as `w1.log` or `w3.log`. Each contains labelled generation, evaluation and NLI sections where available. Original pre-cleanup logs are preserved in `archive.zip`.
- `manifests/` — lists of outputs selected for evaluation and their provenance.
- `summaries/` — generation summaries, including `runtime.json`.
- `results.jsonl` — generation attempt records used by the batch runner.
- `summary.csv` — compatibility copy written by the generation summary script. It is not the latest offline evaluation scorecard.

## Naming

Use `logs/w<number>.log` for each workflow and append new runs under a labelled task section. Do not combine multiple workflows in one log. Use `w<number>-<task>` for manifest files, for example `manifests/w3-outputs.jsonl`. Use `latest.json` for current evaluation pointers instead of separate location files. Workflows without recorded execution logs do not yet have a log file.

The hashed folders within `runs/` and `evaluations/` identify dataset and experiment configurations. Their names and paths are preserved because saved records reference them. Use `latest.json` to find the relevant evaluations without navigating the hashes manually.

Historical scores and failed attempts are retained. Do not combine overlapping evaluation batches as independent samples.
