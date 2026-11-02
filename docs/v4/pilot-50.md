# 50-example local development pilot

Completed 24 October 2026 on `v4-finetuning-workstream`.

- 50 fictional resume/JD/target cases across 25 candidate families and 50 JD families.
- Source-family assignments precede pairing and targets: 40 train, 10 development validation.
- Six job areas; 30 Singapore-context cases.
- Task mix: 20 close, 13 partial, 7 poor, 8 sparse and 2 noisy cases.
- 356 evidence mappings; target lengths of 69–109 words reflect short source histories.
- All 50 pass the performed deterministic content checks, with no detected cross-split target near-duplicates.
- Zero external model calls and $0 incremental external API charges.

Local artifacts are in `runs/v4-pilot-50/` (ignored by Git). Read `review.md` for
all input/output cases, `pilot.jsonl` for the draft corpus, and `audit-summary.json`
plus `validation.jsonl` for the findings. The manifest records file hashes and
counts. Generated content is not included in this documentation commit.

This is a development draft, not an accepted training set. Independent semantic
and human review remain pending, and no final benchmark has been frozen. The
session's exact model revision, sampling settings and per-case token usage were
unavailable. These values remain explicit unknowns; the strict archival case
schema therefore rejects the incomplete provenance. Source, ledger and evidence
contracts were checked separately. The standard batch gate also stops at the
missing final-benchmark exclusion manifest. Neither gate was weakened.

No framework or training code was changed for this pilot. The current application
model, baseline fixtures and existing test corpus were left unchanged.
