# V4 500-example hybrid expansion preparation and 20-case trial dry-run

Completed 24 October 2026 on `v4-finetuning-workstream`.

All expansion artifacts, screened real-source records, authored fictional records, frozen holdout fingerprints, split assignments, and prepared teacher requests reside under the ignored local directory `runs/v4-expansion-500/`. Both `runs/v4-pilot-50/` and `runs/v4-pilot-50-v2/` remain completely unchanged and verified against their SHA-256 manifests. Zero paid model calls have been made.

## 1. Development corpus composition (500 pairs, 750 sources)

- **Retained V2 pilot baseline**: 50 pairs (`40 train`, `10 validation`) from 25 fictional candidate families and 50 authored JDs in `runs/v4-pilot-50-v2/`, with split assignments preserved via `python3 -m v4_data split --previous runs/v4-pilot-50-v2/assignments.jsonl`.
- **New fictional candidate families**: 175 families (`160 train`, `15 validation`) producing 350 pairs (`320 train`, `30 validation`).
  - Constructed as structured extractive fact ledgers first (`fictional-ledger-inventory.json`) and rendered deterministically into `original_resume` Markdown before JD pairing.
  - Spans 20 occupational domains, 56.57% Singapore representation (`99/175` families across Singapore districts and polytechnics/universities) and 43.43% international (`76/175` across UK, Ireland, Australia, New Zealand, Canada, Malaysia, Germany, and the Netherlands).
  - Experience block distribution: 123 families with 3 experience blocks, 44 with 2 blocks (including 12 entry-level intern/graduate profiles), and 8 genuinely sparse 1-block profiles restricted to `validation`.
- **Screened and sanitized Djinni candidate families**: 50 development families (`45 train`, `5 validation`) producing 100 pairs (`90 train`, `10 validation`), plus 5 held-out benchmark families (`55` total accepted from `1,000` selectively streamed rows of `lang-uk/recruitment-dataset-candidate-profiles-english` at revision `86255174c6c378a2b52cd7e81d79002c64d899a6`, licence `MIT`).
  - Screening rejected 945 of 1,000 rows (`328` unresolved PII/links/handles/names, `291` insufficient structured employment/skill claims, `168` `<1.5` years experience or aspirational/condition-heavy text, `72` length out of bounds, `69` non-English/Cyrillic script, `17` domain balance cap).
  - Desired future work (`Looking For`) and occupation title labels (`Position`) are never used to infer skills; missing education remains omitted rather than filled with placeholders.
- **Job description mix (500 development JDs + 40 held-out JDs)**:
  - **50 retained V2 authored JDs** (`40 train`, `10 validation`).
  - **275 new authored JDs** (`253 train`, `22 validation`), comprising 225 base authored JDs and 50 authored JDs replacing the up-to-50 USAJOBS slice (`https://data.usajobs.gov/api/search` returns `HTTP 401 Unauthorized` without a registered `Authorization-Key` and introduces federal civil-service/GS-grade skew; Careers@Gov remains excluded).
  - **175 screened and sanitized Djinni JDs** (`157 train`, `18 validation`), plus 15 held-out benchmark JDs (`190` total accepted from `1,000` selectively streamed rows of `lang-uk/recruitment-dataset-job-descriptions-english` at revision `b56a6054c10f1266a141e37c8df66c79ff2863af`, licence `MIT`). Screening rejected 810 of 1,000 rows (`270` non-English/Cyrillic, `217` length out of bounds after boilerplate removal, `168` domain cap, `137` PII/links/handles, `12` excluded domains, `6` near-duplicates).
- **Split isolation**:
  - Final development split: **450 `train`** and **50 `validation`** requests in `runs/v4-expansion-500/prepare/requests.jsonl`.
  - All stretch, poor-fit, sparse, and prompt-injection/noisy evaluation cases are restricted to `validation`.

## 2. Frozen held-out benchmark exclusion manifest

- Reserved 60 `final_benchmark` sources (`20` candidate families: `15` fictional + `5` screened Djinni; `40` JD families: `25` authored + `15` screened Djinni) in `runs/v4-expansion-500/holdout-sources.jsonl`.
- Frozen via `python3 -m v4_data freeze` into `runs/v4-expansion-500/holdout-freeze/holdout.jsonl` (`SHA-256: 97130abd3c8595b6b34a8385e1f3de8d95ba143faac6c09c58cfa4601a899459`) prior to preparing the 500 development requests.
- Verified zero exact, lineage, or word-trigram shingle (`>= 0.75`) collisions across all 750 development sources via `python3 -m v4_data prepare --holdout runs/v4-expansion-500/holdout-freeze/holdout.jsonl`.

## 3. 20-case model-efficiency trial dry-run bundle and cost estimate

- Prepared 20 representative new requests in `runs/v4-expansion-500/trial-20/dry-run-requests.jsonl` (`14` fictional + `6` Djinni candidates; `10` authored + `10` Djinni JDs; `16 train`, `4 validation`; `7` close, `10` partial, `1` stretch, `1` sparse, `1` noisy prompt-injection JD).
- Token estimate across the 20 requests (`runs/v4-expansion-500/trial-20/cost-estimate.json`, prompt `teacher-v1` `SHA-256: 7f543e9fb7e8fd006f6089744a5d8d41d2a31d444ac3ed5d04a366ee0c1b63c9`):
  - **Input tokens**: `18,210` total (`910.5` mean/request; `26,177` max with 25% single-repair cap and 15% tokenization margin).
  - **Billed output + High reasoning tokens**: `36,703` total (`13,703` visible JSON output + `23,000` High reasoning tokens; `1,835.2` mean/request; `52,761` max with repair cap and margin).
- Proposed teacher models and budget assumptions:
  - **Primary teacher (`sol`, High reasoning)**: assumed `$2.50 / 1M` input tokens, `$15.00 / 1M` output+reasoning tokens -> **`$0.5961` expected** (20 requests), **`$0.8569` maximum budget** (up to 25 requests with at most 1 repair attempt).
  - **Comparison teacher (`gemini-4-argon`, High reasoning)**: assumed `$2.00 / 1M` input tokens, `$12.00 / 1M` output+reasoning tokens -> **`$0.4769` expected** (20 requests), **`$0.6855` maximum budget** (up to 25 requests).
  - **Combined dual-model comparison trial** (40 base requests, max 50 with repairs): **`$1.0730` expected**, **`$1.5424` maximum budget**.
- **Blocked pending explicit user approval**: No paid API calls have been executed.
