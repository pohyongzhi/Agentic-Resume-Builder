# Faithfulness evaluation workstream

Last updated: 2026-11-01 (Asia/Singapore)

## Resume here

- Working branch: `feature/faithfulness-evaluation`
- Base: `origin/main` at `da23338`
- Current phase: claim extraction and evidence segmentation
- Last completed task: `F1` - reproduce the current evaluator and add regression fixtures
- Next task: `F2` - repair claim extraction and source evidence segmentation
- Do not modify or stage the untracked top-level `runs/` directory.
- Do not regenerate workflow outputs. Existing saved outputs must be rescored.
- Do not lower the faithfulness threshold to make results pass.
- Do not use paid APIs or interactive Codex responses as evaluation labels.

The next operator should start with:

```bash
cd /Users/kester/Desktop/Agentic-Resume-Builder
git switch feature/faithfulness-evaluation
git status --short --branch
sed -n '1,260p' docs/evaluation/faithfulness-status.md
```

## Objective

Replace the current misleading claim-NLI score with a reproducible, workflow-neutral
faithfulness evaluation that is useful without manual claim labelling. Preserve the frozen
100-pair dataset and saved workflow outputs.

The project brief requires three distinct faithfulness measurements:

1. protected-field violations, calculated in code;
2. unsupported-claim rate, requiring semantic support assessment;
3. fabrication-bait rate, calculated from qualifications absent from the source resume.

The primary groundedness score remains on a `0.0` to `1.0` scale:

```text
claim_faithfulness = supported_claims / scorable_claims
unsupported_claim_rate = unsupported_or_contradicted_claims / scorable_claims
```

Higher claim faithfulness is better. Lower violation and fabrication-bait rates are better.
The three components must be reported separately rather than hidden inside one composite.

## Current defect

The evaluator in `origin/main` extracts Markdown role and education headers as claims and
compares every claim with arbitrary 60-word source chunks using
`cross-encoder/nli-deberta-v3-small`. There are 100 committed W1 generations, but only one row
in `results/results.jsonl` has a direct faithfulness score (`0.2083`); the other 99 remain
unscored. The largest offline scoring scope contains 23 outputs with mean `0.1340`, range
`0.0435` to `0.2917`, and none reaches the `0.8` task-success gate.

That number is not trustworthy. In the inspected `da_resume_001__da_jd_01` output, supported
employers, dates and paraphrased bullets were marked unsupported, and a preserved role header
was classified as a contradiction after harmless Markdown and `(fictional)` differences.

The current evaluator also lacks independent output metrics for protected-field violations and
fabrication bait. Existing workflow harness validation cannot replace evaluator-side checks
because not every workflow uses the same harness.

## Scope and constraints

- Work from the shared evaluator on `main`, not from an individual workflow branch.
- Keep writer models, prompts, saved outputs and the frozen dataset unchanged.
- Make evaluation logic workflow-neutral.
- Use deterministic rules before semantic judging.
- Keep the existing NLI model only where it adds value; do not treat it as ground truth.
- If semantic adjudication is needed, use the already configured local `gemma4:e4b` judge with
  deterministic parameters and a structured schema. Tests must use a fake client.
- Store evidence, evaluator identity, prompt/schema hashes and cache keys with every score.
- Report the absence of human calibration as a limitation; do not fabricate agreement figures.
- Never treat an evaluation failure as a score of zero.

## Task plan and status

| ID | Task | Status | Completion evidence |
| --- | --- | --- | --- |
| F0 | Create dedicated branch and resumable status document | Complete | Branch created from `origin/main`; this document added |
| F1 | Capture baseline tests and reproduce known false positives | Complete | `74 passed, 1 xfailed`; baseline JSON and four regression fixtures committed |
| F2 | Repair claim extraction and source evidence segmentation | Not started | Pending |
| F3 | Add deterministic protected-field violation metrics | Not started | Pending |
| F4 | Add automatically derived fabrication-bait metrics | Not started | Pending |
| F5 | Add semantic support adjudication and evaluator versioning | Not started | Pending |
| F6 | Integrate component metrics into records, task gates and reports | Not started | Pending |
| F7 | Run a saved-output pilot and full W1 rescore | Not started | Pending |
| F8 | Final verification, documentation and integration handoff | Not started | Pending |

## Detailed tasks

### F1 - Baseline and regression fixtures

1. Prepare the ignored project environment with the test dependencies from `requirements.txt`.
2. Run the complete `main` test suite without Ollama calls.
3. Preserve a compact machine-readable baseline for the committed W1 results.
4. Add regression fixtures covering:
   - Markdown role and education headers;
   - Unicode versus ASCII date dashes;
   - harmless `(fictional)` source markers;
   - supported paraphrases;
   - mixed supported and unsupported clauses;
   - invented numbers, qualifications and employers.
5. Demonstrate the current failures before changing evaluator behavior.

Gate: baseline counts are recorded and regression tests isolate the known failure modes.

### F2 - Claims and evidence units

1. Classify output lines as contact, section heading, protected-field header or substantive claim.
2. Exclude contact and heading lines from unsupported-claim scoring.
3. Split substantive text into stable atomic claims without dropping short skill claims.
4. Extract source evidence units from summary sentences, bullets, skills and education details
   rather than arbitrary word windows.
5. Normalize Markdown, whitespace, dash variants and explicit fictional markers for comparison
   only; retain original text in evidence records.

Gate: supported claims in the known W1 case retrieve the correct source evidence, while added JD
claims remain unsupported.

### F3 - Protected fields

Add evaluator-side checks for altered or invented:

- candidate name and contact details;
- employers and actual job titles;
- employment and education dates;
- degrees and institutions;
- numerical quantities and percentages.

Return exact violation types, output spans and expected source values. Do not require every source
field to appear; omission is different from fabrication.

Gate: deterministic tests catch alterations and do not flag faithful formatting changes.

### F4 - Fabrication bait

For each pair, derive bait terms from evaluator keyword labels that are absent from the original
resume, including declared synonyms. Check whether the tailored output newly asserts those terms.
Record the bait terms, matched output spans and per-output pass/fail result.

Gate: no additional hand labels are needed, and injected absent skills are detected in tests.

### F5 - Semantic support

1. Evaluate claims against the most relevant source evidence units rather than arbitrary chunks.
2. Accept deterministic exact/normalized support before invoking a semantic model.
3. Use NLI for clear entailment, contradiction and neutral cases.
4. Send only ambiguous cases to the local structured judge if the pilot shows it is necessary.
5. Require valid source quotations or evidence-unit identifiers from the judge.
6. Cache by source, output, evaluator model, prompt, schema and settings.
7. Increment the evaluator version so stale scores cannot be reused.

Gate: no paid requests, no interactive labels, deterministic reruns, and complete claim evidence.

### F6 - Evaluation records and reports

Add these fields without overwriting old records:

- `claim_faithfulness` and `unsupported_claim_rate`;
- `protected_field_violations` and a per-output violation indicator;
- `fabrication_bait_terms`, matches and a per-output bait indicator;
- scored, unscored and failure counts;
- evaluator identity and cache provenance.

Update task-success gating only after the new metric passes its regression gates. Keep old scores
distinguishable by evaluator version.

Gate: reports aggregate each component separately by workflow, category and fit level.

### F7 - Automated rescore

1. Run a fixed saved-output pilot first.
2. Inspect automated diagnostics for the best and worst claims; do not manually label a corpus.
3. If the pilot passes regression gates, rescore all 100 saved W1 outputs.
4. Regenerate summaries and the human-review packet, leaving human fields blank.
5. Do not regenerate W1 model outputs.

Gate: every saved W1 output is scored or explicitly unscored with a failure reason.

### F8 - Handoff

1. Run the complete test suite and dataset validator.
2. Run Git diff and whitespace checks.
3. Record exact commands, runtimes, score distributions and remaining limitations here.
4. Commit coherent changes with concise messages.
5. Prepare integration notes for applying the evaluator to W2-W7 after their branches merge.

Gate: clean worktree, reproducible commands, and no missing handoff information.

## Acceptance criteria

- Markdown headings and protected-field headers are not counted as semantic claims.
- Harmless formatting changes do not create contradictions.
- Supported paraphrases retain verifiable source evidence.
- Unsupported JD-derived claims remain unsupported.
- Altered protected facts and invented numerical claims are detected.
- Absent qualification bait is derived automatically and detected in outputs.
- Evaluator failures remain unscored rather than silently becoming failures or zeros.
- Repeated scoring with identical inputs and settings is deterministic and cache-safe.
- The full test suite and dataset validator pass.
- Saved model outputs, frozen dataset files and workflow code remain unchanged.

## Progress log

### 2026-11-01 - F0 complete

- Fetched and inspected all workflow branches and the evaluator on `origin/main`.
- Confirmed that the final seven-workflow comparison is not complete.
- Reviewed the project brief's three faithfulness requirements and IBM's groundedness definition.
- Created `feature/faithfulness-evaluation` from `origin/main` at `da23338`.
- Added this resumable work plan.
- Preserved the untracked top-level `runs/` directory without modification.

Next action: execute F1 and update this file before beginning F2.

### 2026-11-01 - F1 complete

- Installed missing `pytest` and `trafilatura` packages in the ignored `venv/` only.
- Ran the original `main` suite successfully: `73 passed`.
- Recorded the exact partial-evaluation baseline in
  `docs/evaluation/faithfulness-baseline.json`.
- Corrected the earlier interpretation: 100 W1 generations exist, one is directly scored in
  `results/results.jsonl`, and the largest offline scoring attempt covers 23 unique outputs.
- Added four fictional regression fixtures covering formatting/paraphrase, mixed unsupported
  clauses, altered protected fields and automatically derivable fabrication bait.
- Added an expected-failure test proving that the legacy claim extractor treats protected-field
  Markdown headers as semantic claims.
- Ran the expanded suite: `74 passed, 1 xfailed`.
- Did not change evaluator behavior, saved outputs, frozen data or workflow code.

Next action: execute F2, remove the expected-failure marker when the repaired extractor passes,
and update this file before beginning F3.
