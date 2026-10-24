# V4 session trial: 20 reviewed drafts

The 20 prepared development requests now have locally archived resume drafts,
complete output-span evidence mappings and per-case editorial reviews. The user
selected Medium reasoning and session drafting instead of API execution. No paid
requests, training, cluster access or remote publication took place.

The local review is
`runs/v4-expansion-500/trial-20/session-medium-001/review.md`.
The reviewed records are `reviewed-drafts.jsonl` in that directory. They are
**content drafts, not formally accepted training records**. All generated text,
source excerpts, review artifacts and batch assembly scripts remain ignored.

## Findings and next gate

The batch retains its existing 16 train / 4 validation assignments. Seven targets
are content-ready for independent review. Thirteen need source, pairing or
privacy review. These decisions do not alter the frozen assignments or the
500-case inventory. Targets range from 52 to 252 words, with a median of 196.5;
the shortest is a quarantined profile, not an accepted example of resume quality.

The prepared inputs do not yet meet the V2 reference quality consistently:

- Several training pairs match unrelated occupations, including retail or design
  experience against software-development vacancies. Other pairs require
  unsupported specialist experience or a substantial seniority stretch.
- One fictional internship attributes operational work to a year before the
  internship. Other timelines contain overlaps or projects before the named
  employment period without explanatory context.
- Several Djinni profiles lack employers and dates. One ledger categorizes
  scrum-master experience as education and lists aspirations under skills.
- One validation source retains distinctive event and award details requiring
  further privacy review. Its target omits those details; that omission does not
  clear the source for use.

All 20 targets received an in-session editorial review against their visible
facts and JDs. This is not independent human sign-off. Actual titles, approval
relationships, supervision and proficiency remain explicit. Unsupported JD
qualifications and injected credential instructions are omitted. Two targets
received one editorial correction pass each, narrowing master-record and
invoice-approval wording. Initial drafts and exact edits are retained.

Not every target is useful as an application to its paired job. Faithful prose
cannot repair an unrelated pairing, missing employment evidence or an ambiguous
timeline. The remaining expansion stays behind the independent trial-review
gate. Source defects should be resolved in a versioned follow-up; do not silently
replace protected artifacts or count quarantined drafts toward the 500 accepted
examples.

## Verification and provenance

`content-audit.jsonl` records evidence-span validity, complete claim coverage,
number support, literal role-header preservation, section order and privacy
pattern checks: 20/20 pass, with 182 evidence mappings and no detected cross-split
target collisions. These deterministic checks are not semantic acceptance tests.
`formal-validation.jsonl` records the existing validator's result, including
full source, split and holdout preflight. Strict provenance requirements remain
in force: session cost and token usage are unavailable. The first reported
schema failure is `case.generation.cost_usd: expected number`. No
schema was relaxed for these drafts. A separate content audit is necessary
because the formal validator stops content checks after a schema failure.

Exact session model identity, revision, sampling parameters, API response IDs,
token usage and session cost were unavailable and remain unknown. Medium is a
user-reported selection, not provider telemetry. The prepared prompt hash is
preserved, but this was not an isolated API invocation: session context also
influenced drafting. Recorded timestamps describe batch preparation and
materialization. External API requests and external API cost were both zero.

The complete V4 test suite passed: 51 tests using
`python3 -m unittest discover -s tests -p 'test_v4*.py'`.
The initial pilot, V2 pilot, prepared source inventory, splits, requests and
holdout artifacts were preserved: all 89 captured file hashes matched. Per-file preservation hashes, target length
statistics, rejection reasons and artifact hashes are in the ignored run.

The earlier staged API-runner changes remain separate pending work. This
documentation commit neither executes nor approves that runner, and introduces
no generation framework, training code or cluster scripts.
