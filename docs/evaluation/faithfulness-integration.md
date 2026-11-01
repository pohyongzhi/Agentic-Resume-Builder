# Faithfulness evaluator integration

This evaluator is workflow-neutral. Apply it to saved outputs only after the relevant workflow
branch is merged and registered in `workflows.WORKFLOWS`. Do not regenerate a workflow merely to
change evaluator logic.

## Components

Each output receives three independent measurements:

1. `claim_faithfulness` and `unsupported_claim_rate` from substantive output claims checked
   against retrieved source-resume evidence;
2. `protected_field_violation` plus exact findings for altered names, contacts, titles, employers,
   dates, education fields and numbers;
3. `fabrication_bait_violation` plus exact matches for evaluator keyword families absent from the
   source resume.

The compatibility field `faithfulness` equals `claim_faithfulness`. Higher claim faithfulness is
better; lower unsupported-claim and violation rates are better. Do not combine these values into
an undocumented composite.

Task success requires a completed structurally valid resume of the configured length,
`claim_faithfulness >= 0.8`, no protected-field violation and no fabrication-bait violation. A
missing component leaves task success unscored.

## Local setup

Use an ignored virtual environment and the pinned project requirements. The first semantic run
needs the public NLI model; it does not use a paid API.

```bash
python3 -m venv venv
venv/bin/python -m pip install -r requirements.txt
venv/bin/hf download cross-encoder/nli-deberta-v3-small
```

The observed model revision for the W1 rescore was
`fa2804872c3b4bd748f38c0185cc85775361e735`. Every run records the resolved revision and config
hash in its evaluator identity.

## Score saved outputs

Run the pilot first. `--skip-judge` avoids the separate rubric-quality judge and does not disable
faithfulness.

```bash
HF_HUB_OFFLINE=1 venv/bin/python evaluate.py score \
  --workflows w1_single_pass \
  --skip-judge \
  --pilot \
  --limit 6
```

Inspect the versioned output root printed by the command. Every pilot record must have no
`evaluation_failures`, all three current component-version fields and non-empty evidence
candidates for every semantic claim. Then resume the complete saved-output scope:

```bash
HF_HUB_OFFLINE=1 venv/bin/python evaluate.py score \
  --workflows w1_single_pass \
  --skip-judge \
  --resume

HF_HUB_OFFLINE=1 venv/bin/python evaluate.py report \
  --workflows w1_single_pass \
  --skip-judge
```

Replace `w1_single_pass` with any merged workflow ID, or pass several IDs after `--workflows`.
Generation rows must already exist in `results/results.jsonl`, refer to the frozen dataset hash and
have valid saved resume/trace paths. Evaluation records are written under a content-addressed
`results/evaluations/` path and do not overwrite generation artifacts.

## Cross-workflow procedure

For W2-W7:

1. merge the workflow implementation and its saved generation rows without changing the frozen
   test resumes, JDs or evaluator labels;
2. run the complete test suite and dataset validator;
3. run a six-record faithfulness pilot for that workflow;
4. verify zero evaluator failures and inspect automated best/worst evidence diagnostics;
5. score the full saved scope and generate reports;
6. compare workflows using the same dataset hash, evaluator hash, model revision and task gates.

Do not compare records with different evaluator hashes as though they came from one metric. Do not
lower the `0.8` threshold to manufacture passes. Do not convert evaluator failures to zeros.

## Current W1 reference

The `claim-evidence-nli-v3` rescore covered 100 saved outputs and 2,872 claims with zero evaluator
failures. Macro mean claim faithfulness was `0.3355`; five outputs had protected-field violations,
60 had fabrication-bait violations and none passed the strict task gate. See
[`faithfulness-w1-v3-summary.json`](faithfulness-w1-v3-summary.json) for exact hashes and strata.

These results are automated and not human-calibrated. NLI can be conservative when a claim needs
multiple source units, while a mixed claim containing both supported and unsupported content is
intentionally treated as unsupported. Keep that limitation visible in reports.
