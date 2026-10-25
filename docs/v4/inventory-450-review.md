# V4 450-pair source inventory review

## Outcome

The additional development inventory is ready for tailored-target generation at `runs/v4-inventory-450-v1/inventory-042`. The directory is ignored by Git and contains no generated tailored resumes. It preserves the frozen V2 pilot and final-evaluation holdout without modification.

The inventory contains exactly 450 source pairs: 410 training and 40 validation. These combine with the accepted V2 pilot's 40 training and 10 validation cases to produce the intended 450/50 final development split. The new inventory uses 225 candidate families with two distinct JDs per candidate. Candidate and JD families are isolated between splits.

## Sources and provenance

| Decision | Source | Pinned version | Card license | Screened | Retained | Reason |
| --- | --- | --- | --- | ---: | ---: | --- |
| Included | [Djinni candidate profiles (English)](https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english) | `86255174c6c378a2b52cd7e81d79002c64d899a6` | MIT | 4,200 | 225 | Anonymous 2020–2023 CV text with enough explicit evidence after screening |
| Included | [Djinni job descriptions (English)](https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english) | `b56a6054c10f1266a141e37c8df66c79ff2863af` | MIT | 7,400 | 264 | Responsibilities, requirements, seniority, and skills remained useful after sanitization |
| Rejected | [datasetmaster/resumes](https://huggingface.co/datasets/datasetmaster/resumes) | Reviewed 2026-10-25 | MIT card declaration | — | 0 | Real and synthetic records cannot be separated reliably; upstream consent and provenance are insufficient |
| Rejected | [candidate-matching-synthetic](https://huggingface.co/datasets/michaelozon/candidate-matching-synthetic) | Reviewed 2026-10-25 | MIT card declaration | — | 0 | Narrow templates and generic responsibility prose do not meet the V2 evidence reference |
| Rejected | [RJDB](https://github.com/megagonlabs/rjdb) | Reviewed 2026-10-25 | CC BY-SA 4.0 | — | 0 | Synthetic matching task plus share-alike/upstream constraints were unnecessary |
| Rejected | [USAJOBS](https://developer.usajobs.gov) | Reviewed 2026-10-25 | Public-data terms with exceptions | — | 0 | No credentials were used; federal-domain skew was unnecessary after Djinni screening |

The included dataset cards report 210,250 candidate records and 141,897 JD records. Selective Hugging Face Dataset Server slices were used instead of full downloads. Raw slices and normalized records remain under ignored local directories.

The cards declare MIT, but the provenance of every third-party free-text fragment, original consent, and downstream redistribution permission cannot be independently established from the cards alone. The records therefore remain approved only for local development. Redistribution or publication requires a renewed source, privacy, attribution, and license review.

## Filtering and sanitization

Exactly 3,975 of 4,200 screened candidate rows and 7,136 of 7,400 screened JD rows were rejected or not selected. Rejection reasons are non-exclusive.

Candidate screening recorded 3,692 thin records, 2,789 with too few fact units, 2,352 with too few explicit skills, 1,584 with too few responsibilities, 552 outside the supported occupations, 262 with contradictory experience totals, 76 with distinctive personal details, nine without two compatible JDs, four with post-publication dates, one inverted timeline, and one frozen-holdout overlap. Another 58 eligible records were not selected.

JD screening recorded 4,261 thin records, 3,323 with too few explicit skills, 2,990 with too few responsibilities, 1,454 outside the supported occupations, 432 leadership roles, 257 with insufficient occupation evidence, 103 near duplicates, 91 with residual PII, 15 exact duplicates, two malformed titles, two unsupported language requirements, and two frozen-holdout overlaps. Another 1,505 eligible records were not selected.

Across all screened rows, deterministic sanitization recorded:

- candidate organization redaction in 759 rows, product-name redaction in 106 rows, contact/profile-link removal in 270, non-English-line removal in 206, and sensitive-personal-line removal in 85;
- structured JD company-name redaction in all 7,400 rows, additional organization redaction in 2,060 rows, and product-name redaction in 74 rows.

Selected records have zero detected contact patterns, non-technology domains, malformed redaction markers, known audited organization residues, non-English remnants, exact duplicates, near duplicates, split-family leaks, Java/JavaScript crossover matches, or holdout overlaps. Employer and product names, recruiter details, links, application instructions, benefits, discriminatory language, prompt injections, company marketing, aspirations, inconsistent experience totals, and unsupported occupation evidence are removed or rejected deterministically.

## Pairing and split audit

The split was reserved before pairing. Stable candidate and JD family assignments prevent cross-split reuse, and pair graph components are validated after allocation.

| Metric | Result |
| --- | ---: |
| Training candidate families | 205 |
| Validation candidate families | 20 |
| Training JD families | 233 |
| Validation JD families | 31 |
| Training pairs | 410 |
| Validation pairs | 40 |
| Frontend / Java / QA pairs | 150 / 150 / 150 |
| Distinct candidate/JD edges | 450 |
| JD reuse | 148 once, 46 twice, 70 three times |
| Pair score, minimum / median / maximum | 0.716667 / 0.906667 / 1.0 |
| Skill coverage, minimum / median | 0.375 / 0.8 |
| Experience shortfalls | 0 |
| Exact or near duplicates selected | 0 |
| Cross-split family leaks | 0 |
| Frozen-holdout overlaps | 0 |

Every pair has the same occupation family, at least three explicitly shared skills, at least 30% JD-skill coverage, non-increasing seniority, and no JD minimum-experience requirement above the candidate's reported experience. Automation and named specialization roles require corresponding candidate evidence. Every candidate has two distinct JDs; one JD may support at most three candidates and never crosses splits.

Candidate source length is 150 / 177 / 204 / 244 / 428 words at minimum, first quartile, median, third quartile, and maximum. JD source length is 94 / 114 / 132 / 167 / 465 words at the same points.

## Manual review

Thirty final pairs were reviewed: ten per occupation family, stratified across low, median, and high pairing scores and including validation examples. All 30 were accepted as coherent, evidence-bearing source combinations suitable for later tailoring. The sample included lower-overlap pairs and a genuinely entry-level candidate. The review checked occupation, responsibility overlap, explicit skill support, seniority, experience requirements, employer and product-name residue, JD marketing, source coherence, and practical application usefulness.

During the audit, rejected, repaired, or replaced source and pair candidates included experience shortfalls, contradictory experience totals, hidden leadership requirements, unsupported Java automation requirements, unsupported language requirements, malformed JD titles, automation or framework specialization without evidence, misleading JD bodies, company marketing, branded product descriptions, employer residues, aspiration text, cross-line redaction damage, and duplicate candidate/JD allocations. The final inventory contains none of those observed defects. The ignored review sample is `runs/v4-inventory-450-v1/inventory-042/manual-review-sample.jsonl`.

## Validation

Run the independent validator with:

```sh
python3 scripts/v4_inventory_450.py validate \
  --inventory runs/v4-inventory-450-v1/inventory-042 \
  --holdout runs/v4-expansion-500/holdout-freeze/holdout.jsonl
```

It reports 489 sources, 225 candidates, 264 JDs, 450 pairs, 410 training pairs, 40 validation pairs, zero targets, and zero holdout overlaps. The complete V4 test suite must also pass before generation begins.

The inventory is safe to use for local tailored-target generation. It is not cleared for redistribution, public release, or transmission to an external model provider without a separate privacy and licensing decision. The existing V2 pilot and frozen final-evaluation artifacts remain the quality and leakage references.
