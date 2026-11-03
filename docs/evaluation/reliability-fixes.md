# Evaluator reliability fixes

This branch corrects seven confirmed reliability defects without changing frozen evaluation data or acceptance thresholds.

## Behavioral contract

### Faithfulness support

The deterministic shortcut accepts only a complete normalized evidence-unit match. A claim such as `AWS` is not automatically supported by `No experience with AWS.` Subsets and paraphrases proceed through the configured NLI evaluator.

### Protected fields

Names, contacts and individual protected values must exist in the source. Employment titles, employers and dates must also form a compatible source entry; degrees, institutions and dates follow the same rule. Entries may be reordered or omitted. Formatting changes and split Markdown headers remain valid.

### Fabrication bait

A keyword family is eligible as bait when it has no positive source mention. Explicit local denials such as `no AWS experience`, `did not use AWS`, `AWS: no experience`, and `AWS not used` do not count as positive mentions. Contrast clauses are assessed separately, so `no AWS experience, but used Python` retains the Python assertion.

This is a high-precision lexical safeguard, not general semantic negation analysis. Complex scope, irony and indirect claims still require human review or a semantic model.

### W4 operational scoring

W4 expects `validate_resume` followed by `render_resume`. The minimum is five agent steps when verification has no eligible claims: two extraction calls, draft, validation and rendering. When claims need verification, those calls increase the observed step count. With two grounding rounds and one retry for draft/revisions, its maximum is thirteen steps.

### Failure isolation

Generation failures remain in `error`. Evaluation failures are stored by component in `evaluation_failures`. A successful generated resume and its trace are saved even when every evaluator is unavailable. Online runs record `generation_seconds` and `evaluation_seconds` separately.

### Generation cache identity

The generation hash includes generation settings, shared prompt values, shared generation code and the selected workflow module. It excludes evaluator modules and evaluator-only settings, allowing offline rescoring without regenerating outputs.

### Blinded review identity

Review IDs derive from dataset, pair, repeat, workflow, generation config and the tailored-resume content hash. The public packet remains blinded; the workflow stays in the private mapping.

## Versioning and validation

- Evaluator schema: `2.2`
- NLI evaluator: `claim-evidence-nli-v4`
- Protected fields: `protected-fields-v2`
- Fabrication bait: `keyword-bait-v2`
- Full suite on 2026-11-03: 347 tests passed and 21 subtests passed in 5.30 seconds.

The historical v3 evaluation remains a historical artifact. A v4 score must be produced by running the evaluator under the new identity. Automated scores still need calibration against independent human annotations before they can be treated as validated estimates of real-world correctness.
