# V4 local data foundation

This is an offline, standard-library-only preparation and validation package. It
does not load a model, call Ollama or a remote teacher, download data, export a
training set, or start training. Its only generator is an extractive renderer for
two tiny fictional software-test cases. Those outputs are not SFT examples or an
unseen benchmark. Teacher integration and independently reviewed corpus acceptance
remain later work.

## Quick verification

From the repository root, using Python 3.10 or newer:

```sh
python3 -m unittest discover -s tests -p 'test_v4_*.py' -v
python3 -m v4_data demo --out runs/v4-demo --at 2026-10-24T00:00:00Z
```

Choose an unused output directory. The demo creates two cases from four source
records and two ledgers, with zero model requests and zero API cost. It writes
split assignments, pair plans, teacher requests, cases, validation results and a
manifest. Both valid cases must be `needs_review`, never accepted for training.
All demo families are explicitly reserved for `train` to make software testing
predictable; this is not the experimental corpus split.

`--at` fixes UTC timestamps. With identical sources, policy, timestamp and code,
requests, target text and evidence mappings are deterministic. Case/run IDs also
include run identity; a different output directory denotes a different run. The
manifest records the Git commit, a digest of the executing package's Python
files, input/output byte hashes, schema/policy hashes, source IDs, counts and
actual preparation cost. Failed writes leave an incomplete directory without a
manifest; it must not be consumed as a completed run. Existing runs are never
overwritten.

## Contracts and storage

`schemas/v4/records.schema.json` is a JSON Schema 2020-12 definitions document.
Validate each row against its named `$defs` entry, not against the unconstrained
document root. The built-in validator implements the restricted vocabulary used
by these definitions and rejects unknown keywords. Records reject unknown fields,
duplicate JSON keys, non-finite numbers and incompatible schema versions.

| Definition | Purpose |
|---|---|
| `source` | Candidate or JD, stable record/family/template IDs, parent and related families, exact sanitized text, raw/text/normalized hashes, source revision, license, restrictions, admission and privacy review |
| `ledger`, `fact`, `span` | Candidate-only fact index, bound to the candidate text hash; category, confidence and exact visible source spans |
| `pair`, `assignment`, `policy` | Existing graph edges or new pair plans, per-source component/split assignments, seeded allocation policy |
| `holdout` | Evaluator-owned benchmark IDs, family IDs, hashes and lexical fingerprints; no benchmark text |
| `request` | Versioned prompt, visible inputs, checked fact index and assignment digest |
| `case`, `evidence_mapping`, `provenance` | Original inputs, Markdown target, output-to-fact mappings, prompt/model revisions, request hash, parameters, retries, timestamps, usage and lineage |
| `validation`, `rejection` | Case-bound validator version, deterministic findings and structured rejection/review reasons |
| `manifest`, `file` | Versioned run inventory and hashes; official training-base identity explicitly marked unpinned and not ready |

All records use schema version `1.0`; one record per UTF-8 JSONL line. Small policy,
reservation and run manifest files use JSON. Character offsets are Unicode code
points with an exclusive end, not UTF-8 byte or UTF-16 offsets. `sha256` fields use
lowercase hex. Text hashes preserve exact text; normalized hashes use NFKC,
case folding and joined word tokens (`nfkc-words-v1`). Structured hashes use
sorted-key compact JSON with Unicode preserved and non-finite numbers forbidden.

Source IDs are stable, author-assigned identifiers with explicit versions. Keep
family IDs stable across edits, translations, paraphrases and source mirrors;
use globally namespaced IDs to avoid accidental merges. Register changed source
text as a new record with `parent_ids`. All local parent references must resolve,
and cycles are rejected. Pair, request, case and generation-family IDs are derived
from their identity inputs. Changes to facts require a new ledger version and
digest; facts may not supply information absent from the visible original resume.
The initial ledger contract deliberately requires claims to equal their evidence
spans; semantic normalization needs a separately reviewed extension.

Source licenses are recorded, not inferred or reassigned. `pending` and `excluded`
sources fail admission; `local-approved` requires a recorded decision and passed
privacy review. `fixture-only` records have no training/evaluation admission.
The tiny fixtures declare `LicenseRef-Local-Test-Fixture`, not a license for a
mixed-source corpus. Fingerprints and hashes are not anonymization.

Private/raw/generated data belong under ignored `data/` subdirectories; run
products belong under `runs/`. Only `data/fixtures/v4/*.jsonl` is allowlisted for
version control. Weight formats, caches and common secret files are ignored.
Existing tracked baseline fixtures and caches are preserved; ignore rules do not
untrack existing files or prevent an explicit `git add -f`. Review staged paths
before every commit. No acquired corpora or model weights belong in Git.

## Split before pairing

```sh
python3 -m v4_data split \
  --sources data/private/sources.jsonl \
  --out runs/v4-split
```

The splitter first joins shared candidate/JD/template families, declared related
families, parent lineage, matching raw hashes or upstream source identities,
exact/near copies, and all supplied existing pair edges.
It allocates entire connected components using the policy seed and a stable hash
of each component's smallest source ID. Default weights are 80% train, 10%
validation and 10% synthetic test. These are probabilities over components;
counts can vary, especially with small pools or giant components. Final benchmark
membership is explicit and never a random allocation.

Use `--existing-pairs data/private/old-pairs.jsonl` for paired source corpora. These
are pre-existing graph relationships, not the new pair plan. Retain and supply
the same edges to later `prepare`, `freeze` and `validate` commands. A missing
edge file is detected when it changes recorded component membership.

Use `--reservations data/private/reservations.json` for evaluator-selected final
families or development-only pilot families, for example:

```json
{"candidate-heldout-v1": "final_benchmark", "jd-heldout-v1": "final_benchmark"}
```

Every member of the reserved component inherits its split. When expanding a
corpus, supply `--previous runs/v4-split/assignments.jsonl` and the complete old
plus new source inventory. Frozen source splits survive additions. Changes to
old source content, family identity or policy fail; a new relation joining two
different frozen splits fails rather than silently moving examples. Resolve such
a collision by quarantining the bridging material and reviewing the inventory.
Do not omit `--previous` to bypass a collision.

After assignment, construct new pairs with
`v4_data.splitting.pair_sources(candidate_id, jd_id, sources, assignments)` and
write them with `write_jsonl(..., "pair")`. This computes the stable pair ID and
blocks cross-split or final-benchmark pairings. `prepare_requests` rechecks the
whole graph before using any such plan. Never randomly split generated rows.

Duplicate detection combines normalized exact matches, containment, word-trigram
Jaccard similarity and character-sequence similarity. The default near-duplicate
threshold is 0.82. These are conservative lexical safeguards, not reliable
semantic paraphrase/translation detection. Such variants must share declared
families; independent review remains necessary. The all-pairs check is quadratic
and intended for small local inventories. Profile and replace it with a tested
candidate-retrieval stage before larger corpus preparation; do not relax the
leakage checks to meet a throughput target.

## Legacy exclusions and final benchmark isolation

`data/pairs.csv`, its original resume, JD and keyword file are blocked by the
pinned inventory in `configs/v4/legacy-exclusions.json`. Admission checks paths,
IDs, raw hashes, lineage metadata, normalized content and near copies, including
renamed inputs. This applies before every split, including final-benchmark
reservations. Missing or changed protected files stop admission until their
exclusion inventory is explicitly reviewed. These baseline files remain usable
by the existing application, which this package does not modify.

No actual final benchmark is frozen yet. An evaluator must independently author
and admit it, reserve its families before generation, and freeze its protocol:

```sh
python3 -m v4_data freeze \
  --sources data/private/evaluator-sources.jsonl \
  --assignments data/private/evaluator-assignments.jsonl \
  --protocol data/private/evaluation-protocol.md \
  --out runs/v4-holdout
```

The resulting `holdout.jsonl` contains hashes, IDs and hashed word trigrams, not
benchmark text. Non-fixture request preparation requires this snapshot. It blocks
matching IDs/lineage/families, exact hashes and substantial lexical overlap (75%
of the smaller fingerprint set). Keep benchmark texts, annotations and the
protocol evaluator-owned; supply the generator only the exclusion snapshot and
a separate development inventory. Generator preparation refuses an inventory
containing any `final_benchmark` assignment. Software fixtures cannot be frozen
as a final benchmark. A local hash protects integrity checks, not authenticity
against an operator deliberately replacing a manifest.

## Request preparation and validation

```sh
python3 -m v4_data prepare \
  --sources data/private/development-sources.jsonl \
  --ledgers data/private/ledgers.jsonl \
  --assignments data/private/development-assignments.jsonl \
  --pairs data/private/planned-pairs.jsonl \
  --holdout runs/v4-holdout/holdout.jsonl \
  --out runs/v4-requests
```

Preparation checks admission, split membership, benchmark exclusions, visible
facts and contact-pattern flags before writing any prompt. It creates requests
only and incurs no teacher cost. A future backend must report its proposed case
count, maximum attempts, token budget, pricing and expected cost, and obtain
confirmation before paid calls or bulk generation. No paid or local model
backend is included in this foundation.

The versioned teacher prompt handles sparse evidence and treats source text as
data. The application's existing shared prompt is unchanged. Before experimental
comparisons, reconcile its length rule and sparse-input exception consistently
across all compared prompts, then freeze those revisions.

Import future responses as archival `case` JSONL with `offline-response`
provenance, exact request hash, pinned teacher revision, reported usage and a
separate evidence map. Attempts are capped at two; the second must reference the
first attempt in the same batch and pair. Then validate:

```sh
python3 -m v4_data validate \
  --sources data/fixtures/v4/sources.jsonl \
  --ledgers data/fixtures/v4/ledgers.jsonl \
  --assignments runs/v4-demo/assignments.jsonl \
  --pairs runs/v4-demo/pairs.jsonl \
  --cases runs/v4-demo/cases.jsonl \
  --out runs/v4-demo-validation
```

For non-fixture data also pass `--holdout`. Checks cover source/target/request
hashes, family and split consistency, ledger binding, provenance, retry lineage,
valid evidence spans and IDs, complete mapping of short claims/skills, verbatim
support, unsupported numbers/dates in cited evidence, section structure, contact
patterns and cross-split target similarity. Batch-level invalid source inventories
stop the command. Case findings are structured in `validation.jsonl`.

Exit codes are 0 for a completed run, 1 for invalid inputs or an incomplete run,
and 2 for a completed validation run containing rejected cases. Exit 0 does not
mean training acceptance. Every deterministically passing case remains
`needs_review`; no acceptance or training-export path is implemented. Lexical
checks cannot establish semantic support, employer/role relationships,
proficiency, negation, relevance, completeness or complete de-identification.
Independent review and the planned human pilot audit remain required.

The future trainable base is the official `Qwen/Qwen3.5-4B` checkpoint, not
`qwen3.5:4b-q4_K_M`. Manifests explicitly mark its revision
`UNPINNED-NOT-READY`; no weights or training configuration are materialized by
this package. Cluster access, LoRA, merge/conversion/quantization and Ollama import
remain outside this milestone.
