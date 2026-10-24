# Revised V4 50-example pilot

Completed 24 October 2026 on `v4-finetuning-workstream`.

The separate local pilot is in `runs/v4-pilot-50-v2/`; `review.md` contains all
original resumes, JDs, final targets, claim references and per-target reviews.
The initial `runs/v4-pilot-50/` is unchanged, verified against a file-hash snapshot.

- 50 new fictional cases: 25 candidate families × two JDs; 40 train / 10 validation.
- 24 families have two or three substantial experiences; one is genuinely sparse.
- 50/50 pass performed deterministic factual content checks and direct editorial support review, with 613 evidence mappings.
- No detected cross-split family/component or lexical near-duplicate leakage; poor-fit and injection cases are validation-only.
- Target lengths: 138–293 words, median 261.5, p90 279.1. Counts: 2 below 150; 2 at 150–199; 12 at 200–249; 34 at 250–299.
- Original lengths: 152–299 words, median 277 across the 25 distinct originals.
- 18 edits repaired 16 targets; no examples discarded. Repairs clarify approval responsibilities, preserve manual-versus-automated test scope and remove unnecessary disclaimers.
- 47 useful application bodies, one stretch application and two deliberate poor-fit QA cases. Every target is a usable resume body for its actual background; the two poor-fit targets cannot credibly satisfy their advertised jobs.
- Zero external model calls and $0 incremental external API charges; session accounting unavailable.

`pilot.jsonl` and `ledgers.jsonl` retain exact target/source evidence spans.
`editorial-review.jsonl`, `repairs.json`, `coverage.jsonl`, `validation.jsonl`,
`audit-summary.json` and `manifest.json` preserve findings and artifact hashes.
Generated data and authoring records remain ignored under `runs/`.

These remain development drafts, with zero cases accepted for training. The
unmodified batch gate requires the still-missing frozen benchmark exclusion
manifest. Strict archival case validation rejects unavailable sampling metadata,
which remains null. The user-supplied model label is not a verified model revision.
Every target received an in-session editorial read-through; independent human
sign-off remains pending. Content checks do not waive either archival gate.

No framework, training code or cluster scripts were added. No paid APIs, cluster
access, training, weight download or GitHub push occurred.
