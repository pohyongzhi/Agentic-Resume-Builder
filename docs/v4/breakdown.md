# V4 workstream breakdown

Last updated: 27 October 2026

## 1. Executive summary

The V4 workstream produced a reproducible, evidence-constrained resume-tailoring dataset and a working LoRA adaptation of the official `Qwen/Qwen3.5-4B` model. Work progressed through research, schema design, two 50-case pilots, public-source screening, a 500-case dataset, an editorial rewrite, a 50-case constraint augmentation, single-GPU training, adapter verification, model merging, GGUF conversion, local Ollama import, and controlled application-level comparisons.

The current training corpus is `runs/v4-dataset-550-v4-1/`:

| Item | Result |
| --- | ---: |
| Total accepted examples | 550 |
| Training examples | 490 |
| Validation examples | 60 |
| Preserved V3 core | 500 |
| V4.1 constraint additions | 50 |
| Added candidate families | 25 |
| Added split | 40 train / 10 validation |
| Paid generation calls | 0 |
| Frozen-holdout overlaps | 0 |

The current trained candidate is imported locally as `agentic-resume-v4-1:q4_K_M`. It is not yet the application default. The application still uses `qwen3.5:4b-q4_K_M` so that the established baseline remains available while V4.1 is evaluated and refined.

The strongest measured outcome is improved constraint retention. V4.1 preserved automation percentages, actual titles, limited proficiency, temporary authority, and manager approval boundaries more reliably than V3. Its remaining defect is occasional negative wording in the summary, such as explicitly stating what the candidate does not own. The recommended next iteration is a narrow V4.2 data correction followed by retraining and the same fixed comparison protocol.

## 2. Scope and controls

The workstream followed these controls throughout:

- Development was isolated on branch `v4-finetuning-workstream`; nothing was pushed.
- Raw public data, generated datasets, caches, logs, checkpoints, adapters, merged weights, and GGUF files stayed outside ordinary Git or under ignored `runs/` paths.
- The official Hugging Face base was used for training. The quantized Ollama baseline was never used as a training source.
- `data/pairs.csv` was hash-pinned and excluded from training, validation, and final evaluation.
- Candidate families, JD families, near duplicates, and connected pair components were prevented from crossing data splits.
- Frozen final-evaluation sources were excluded before pairing and were not opened for model selection.
- Candidate facts came only from the supplied resume. Job descriptions controlled emphasis but were not treated as evidence.
- Unsupported titles, employers, dates, numbers, qualifications, proficiency, ownership, supervision, and approval relationships were prohibited.
- No paid model API was called. Missing session metadata was recorded as unavailable rather than fabricated.
- Cluster credentials and account secrets were not written to the repository or model artifacts.

## 3. Chronological development

### Phase 1 — research and training design

The first phase documented the learning objective, public-data options, privacy risks, licensing limits, split policy, evidence requirements, evaluation plan, LoRA approach, and eventual Ollama deployment path.

Key decisions were:

1. Train a resume rewriter, not a fit classifier or a generator of new candidate facts.
2. Use an archival case containing lineage and evidence metadata, then export only conversational prompt/completion records for training.
3. Split source families before pairing rather than randomly splitting completed rows.
4. Reserve final evaluation sources before development generation.
5. Use completion-only loss so the model learns the tailored resume rather than the system prompt, JD, or original resume.
6. Start from official post-trained `Qwen/Qwen3.5-4B` weights and deploy only after verified merge and conversion.

Primary documents: `docs/v4/README.md`, `docs/v4/datasets.md`, and `docs/v4/training-plan.md`.

### Phase 2 — schemas, fixtures, splitting, and validation

Versioned JSON/JSONL contracts were created for:

- candidate and JD source records;
- candidate fact ledgers;
- SFT cases and output-span-to-fact-ID evidence mappings;
- split assignments and connected components;
- generation provenance;
- validation findings and rejection reasons;
- source manifests and frozen benchmark exclusions.

Stable IDs, family IDs, hashes, timestamps, prompt versions, model metadata, source revisions, licenses, and lineage were included. Tiny fictional fixtures supported tests without committing real or bulk data.

The validators check schema conformance, evidence spans, entities, titles, employers, dates, numbers, qualifications, skill proficiency, ownership language, unsupported JD copying, PII patterns, exact duplicates, near duplicates, chronological issues, family leakage, connected-component leakage, and frozen-holdout overlap.

### Phase 3 — initial and revised 50-case pilots

The initial pilot at `runs/v4-pilot-50/` established the end-to-end QA process. It was retained unchanged after review.

The accepted quality reference became `runs/v4-pilot-50-v2/`:

| Metric | V2 pilot result |
| --- | ---: |
| Cases | 50 |
| Candidate families | 25 |
| JDs per family | 2 |
| Split | 40 train / 10 validation |
| Evidence mappings | 613 |
| Factual checks passed | 50/50 |
| Targets repaired | 16 targets through 18 edits |
| Targets rejected | 0 |
| Target length | 138–293 words |
| Median target length | 261.5 words |

Twenty-four candidate families had two or three substantial experience blocks; one was intentionally sparse. Poor-fit and injection cases were kept in validation. All targets were reviewed for factual support and application usefulness. Two poor-fit cases remained honest resumes but could not credibly meet their advertised jobs, which was the intended QA behavior.

The pilot covered 25 primary candidate roles, including software, QA, data, finance, administration, customer success, IT support, research, marketing, design, purchasing, logistics, facilities, HR, hospitality, retail, events, operations, and sustainability. The two-JD design yielded 50 distinct target job titles.

### Phase 4 — preliminary hybrid expansion and 20-case gate

An early 500-case hybrid plan combined fictional candidates, screened Djinni profiles, authored JDs, and screened Djinni JDs. It also created a frozen holdout manifest and a representative 20-case generation gate. This phase was a preparation and diagnosis stage, not the final source inventory.

The 20-case dry-run bundle covered 14 fictional and six Djinni candidates, ten authored and ten Djinni JDs, and 16 training plus four validation assignments. It included close, partial, stretch, sparse, and noisy prompt-injection cases.

A paid API trial was costed but never executed:

| Item | Planned value |
| --- | ---: |
| Model | `gpt-6.1-sol` |
| Reasoning | High |
| Requests | 20 |
| Expected input tokens | 18,210 |
| Expected output and reasoning tokens | 36,703 |
| Expected cost | $0.4035 |
| Maximum estimated cost before margin | $0.5800 |
| Hard execution budget | $0.65 |
| Actual paid calls and cost | 0 / $0 |

Session-authored trial drafts exposed weak source pairings, unrelated occupations, ambiguous timelines, thin profiles, and residual privacy concerns. Only seven of the 20 drafts were immediately content-ready; thirteen required source, pairing, or privacy review. This finding caused the workstream to rebuild the 450-case source inventory instead of scaling a weak pairing plan.

### Phase 5 — audited public-source inventory for 450 additional cases

The replacement inventory at `runs/v4-inventory-450-v1/inventory-044/` used selectively accessed, screened, and sanitized Djinni English datasets.

| Source | Pinned revision | Card license | Screened | Retained | Decision |
| --- | --- | --- | ---: | ---: | --- |
| Djinni candidate profiles | `86255174c6c378a2b52cd7e81d79002c64d899a6` | MIT | 4,200 | 225 | Included for local development |
| Djinni job descriptions | `b56a6054c10f1266a141e37c8df66c79ff2863af` | MIT | 7,400 | 261 | Included for local development |
| datasetmaster/resumes | reviewed 25 Oct 2026 | MIT card declaration | — | 0 | Rejected: real/synthetic separation and upstream provenance unclear |
| candidate-matching-synthetic | reviewed 25 Oct 2026 | MIT card declaration | — | 0 | Rejected: narrow templates and weak evidence quality |
| RJDB | reviewed 25 Oct 2026 | CC BY-SA 4.0 | — | 0 | Rejected: unnecessary share-alike and upstream constraints |
| USAJOBS | reviewed 25 Oct 2026 | public-data terms with exceptions | — | 0 | Rejected: access requirements and avoidable federal-domain skew |

Selective Hugging Face Dataset Server slices were used instead of downloading the complete datasets. The cards declare MIT, but the provenance, consent, and redistribution status of every original free-text fragment cannot be independently established. The retained records are therefore approved only for local development. Publication or redistribution requires a renewed license, privacy, attribution, and source review.

Filtering removed or rejected direct identifiers, contact details, links, handles, employer and product names, recruiter text, sensitive attributes, non-English remnants, aspirations, inconsistent experience totals, unsupported occupation evidence, company marketing, prompt injections, duplicate text, and malformed records.

The final source inventory contained:

| Metric | Result |
| --- | ---: |
| New pairs | 450 |
| New split | 410 train / 40 validation |
| Candidate families | 225 |
| Candidate/JD edges | 450 |
| Candidate families per occupation | 75 frontend / 75 Java / 75 QA |
| Pairs per occupation | 150 frontend / 150 Java / 150 QA |
| Pair score, minimum / median / maximum | 0.716667 / 0.9 / 1.0 |
| Skill coverage, minimum / median | 0.363636 / 0.8 |
| Experience shortfalls | 0 |
| Exact or near duplicates selected | 0 |
| Cross-split family leaks | 0 |
| Frozen-holdout overlaps | 0 |

Every candidate was paired with two distinct JDs from the same occupation family. Each pair shared at least three explicit skills, covered at least 30% of JD skills, avoided unsupported seniority increases, and satisfied stated minimum-experience requirements. Thirty stratified final pairs, including borderline examples, were manually reviewed and all 30 were accepted.

### Phase 6 — assembly of the 500-case V2 dataset

The 50-case fictional pilot and the audited 450-pair expansion were combined in `runs/v4-dataset-500-v2/`.

| Component | Train | Validation | Total |
| --- | ---: | ---: | ---: |
| V2 fictional pilot | 40 | 10 | 50 |
| Screened Djinni expansion | 410 | 40 | 450 |
| Combined V2 dataset | 450 | 50 | 500 |

The accepted expansion consisted of a reviewed 20-case gate and a 430-case bulk pass. Every generated claim was linked to candidate evidence. The resulting audit reported:

- 4,146 evidence mappings and zero invalid spans;
- zero target PII-pattern findings;
- zero negative missing-qualification or aspiration statements in the accepted expansion;
- zero candidate-family or JD-family split overlap;
- zero exact or near-duplicate target overlap across splits;
- zero exact frozen-holdout collisions;
- twelve exact target duplicate groups within a split, each limited to the same candidate paired with two closely related JDs.

All 500 records were content-ready and authorized for local experimental training. They were not independently human-certified case by case and were not cleared for redistribution. Exact session model revisions and per-case token counts were unavailable and remained marked unavailable.

### Phase 7 — V3 editorial target polish

The expansion targets were useful but too extractive and repetitive. The V3 editorial pass at `runs/v4-dataset-500-v3-r4/` preserved all source resumes, JDs, family assignments, splits, and the original 50-case pilot while rewriting the other 450 targets.

| Quality measure on 450 rewritten targets | Before | After |
| --- | ---: | ---: |
| Exact duplicate target groups | 12 | 0 |
| Source-substring bullets | 2,317 / 2,632 (88.03%) | 5 / 2,544 (0.20%) |
| Generic `documented work in` summaries | 424 | 0 |
| First-person targets | 191 | 0 |
| Raw source labels | 9 | 0 |
| Skill lists with 15+ entries | 22 | 0 |
| Technology-only bullets | 43 | 0 |
| Median target length | 118 words | 117 words |

A stratified 45-record review covered frontend, Java, and QA examples. Fourteen sampled records received narrow factual or copy-edit repairs and were rechecked. All 45 then passed factual-support, usefulness, and material-improvement review. Deterministic content validation returned zero errors.

The final 500-case V3 dataset remained 450 train / 50 validation. Of the 450 rewritten records, 120 recorded a known session model label and 330 retained unavailable model identity metadata. No paid API call was made.

### Phase 8 — reproducible Qwen3.5-4B LoRA package

The training package pinned the official base model rather than the local Ollama quantization:

| Item | Pinned value |
| --- | --- |
| Model | `Qwen/Qwen3.5-4B` |
| Hugging Face revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Python | 3.11.11 |
| PyTorch | 2.14.1 |
| Transformers | 5.20.2 |
| PEFT | 0.21.2 |
| TRL | 1.16.0 |
| Accelerate | 1.15.0 |
| datasets | 5.1.0 |
| safetensors | 0.8.0 |
| sentencepiece | 0.2.2 |
| tokenizers | 0.23.3 |
| huggingface-hub | 2.3.0 |
| Jinja2 | 3.1.6 |

Tokenizer and label audits established that:

- the official template supports `enable_thinking=false`;
- EOS is `<|im_end|>` with token ID `248046`;
- padding is `<|endoftext|>` with token ID `248044`;
- system, user, non-thinking prefix, and padding tokens receive label `-100`;
- only assistant resume tokens and the terminal EOS are supervised;
- every 450/50 V3 record fits the 4,096-token budget without truncation;
- the V3 training split ranged from 698 to 1,322 tokens, with a 903 median;
- the V3 validation split ranged from 671 to 1,336 tokens, with a 984 median.

The actual model index resolved 248 permitted language-backbone projections:

| Module group | Targeted projections |
| --- | ---: |
| Full self-attention | 32 |
| DeltaNet linear attention | 120 |
| MLP | 96 |
| Total | 248 |

Vision modules, embeddings, `lm_head`, and MTP modules were excluded. Training fails if the expected modules do not resolve.

The fixed training recipe was BF16 LoRA rank 16, alpha 32, dropout 0.05, no bias, learning rate `1e-4`, AdamW, weight decay 0.01, cosine decay, 0.05 warmup ratio, three epochs, microbatch one, gradient accumulation 16, 4,096-token sequences, gradient checkpointing, cache disabled, packing disabled, and seed 42.

The package added tokenizer, label-mask, module, smoke-test, full-training, validation-only, adapter-reload, adapter-comparison, and verified-merge commands. Output directories must be new, preventing accidental overwrite.

### Phase 9 — initial training and V3 model assessment

The first full run proved the training path and recorded a validation loss of `0.2291505`, but it used the earlier, more extractive V2 targets. After the V3 editorial rewrite, full training recorded:

| Run | Training loss | Validation loss | Notes |
| --- | ---: | ---: | --- |
| Initial V2-target run | 0.1974 | 0.2292 | Pipeline proof on easier, more extractive targets |
| Unadapted base on V3 validation | — | 1.0251 | Same V3 validation distribution |
| V3 polished-target run | 0.6629 | 0.6434 | Harder and more varied targets; deterministic reload passed |

The V2 and V3 validation losses are not directly comparable because their targets differ materially. The relevant V3 reference is the unadapted base on the V3 validation set.

A six-case local comparison between the established Ollama baseline and the V3 model preferred V3 on four cases, the baseline on one, and produced one mixed result. V3 was safer on poor-fit, sparse, and noisy-source inputs. It still omitted a 10% automation limit in a QA case and exposed a terminal `<|im_end|>` token. The inference boundary was subsequently hardened to strip raw control tokens.

### Phase 10 — V4.1 constraint augmentation

The V4.1 dataset added 50 fictional cases from 25 new candidate families while preserving the 500 V3 records byte for byte. The additions explicitly exercised:

- percentages and quantitative scope limits;
- basic or intermediate proficiency;
- actual-title preservation when the JD used a more senior title;
- manager, lead, counsel, or product-owner approval boundaries;
- temporary and acting duties;
- supervision, ownership, hiring, budget, release, and decision limits.

The additions spanned 25 candidate roles, including QA, data reporting, data engineering, frontend, IT support, security operations, procurement, supply chain, sales operations, marketing, laboratory operations, finance, bookkeeping, HR, project coordination, teaching support, mechanical maintenance, warehouse operations, construction administration, compliance, UX design, customer support, and general operations.

All 50 additions received complete manual review:

| V4.1 addition check | Result |
| --- | ---: |
| Factual support | 50/50 |
| Constraint preservation | 50/50 |
| Useful application resume | 50/50 |
| Repaired | 0 |
| Rejected | 0 |
| Unsupported numeric strings | 0 |
| Unsupported experience headings | 0 |
| Candidate-family split leaks | 0 |
| Exact base overlaps | 0 |
| Near-duplicate candidate pairs | 0 |
| Frozen-holdout overlaps | 0 |
| Maximum holdout five-word-shingle Jaccard | 0.009146 |

The resulting `runs/v4-dataset-550-v4-1/` exports contain 490 training and 60 validation conversations using the V1.1 student prompt.

### Phase 11 — SMU smoke test and full V4.1 training

The cluster environment used Python 3.11.11 with PyTorch `2.14.1+cu130` and the pinned package set. Training ran on one allocated NVIDIA RTX Pro 6000 Blackwell Max-Q MIG instance with approximately 24 GB available memory. Reference PyTorch implementations were used for the optional DeltaNet acceleration kernels; they were correct but slower.

The 12-example, 200-step overfit smoke test passed decisively:

| Smoke-test metric | Result |
| --- | ---: |
| Initial loss | 1.3826933 |
| Final loss | 0.00044697 |
| Final/initial ratio | 0.0003233 |
| Required maximum ratio | 0.5 |
| Memorization gate | Passed |
| Train runtime | 414 seconds |
| Adapter reload token identity | Passed |

The full V4.1 run completed 93 optimizer steps across three epochs:

| Full-training metric | Result |
| --- | ---: |
| Training examples | 490 |
| Validation examples | 60 |
| Optimizer steps | 93 |
| Epochs | 3 |
| Training runtime | 2,571 seconds (42 min 51 sec) |
| Training loss | 0.6223 |
| Validation loss | 0.5822657 |
| Validation runtime | 25.738 seconds |
| Validation throughput | 2.331 examples/sec |
| Deterministic adapter reload | Passed |

V4.1 validation loss was approximately 9.5% lower than the V3 polished-target run (`0.5823` versus `0.6434`). This is encouraging but is not a complete quality measure; the application-level comparison remains more important for factual behavior.

### Phase 12 — merge, conversion, quantization, and local import

The final adapter was reloaded against the exact pinned base and merged into BF16 weights. The adapter and merged model produced identical token IDs for the 128-token verification sample. The merged output contained three safetensors shards and remained outside Git.

Conversion proceeded as follows:

1. Direct Ollama conversion paths were investigated. The required Q4_K_M path was not available through the tested built-in safetensors route, and MLX-based alternatives were unsuitable on the Linux node.
2. Official `llama.cpp` was pinned at commit `a8276b70ccf7fd66a7f31ec1db2761d947794b9e`.
3. The merged checkpoint was converted to a text-only F16 GGUF and quantized to Q4_K_M.
4. The first GGUF retained NextN metadata and failed local loading because Ollama expected a missing `blk.32.attn_norm.weight` tensor.
5. Conversion was repeated with `--no-nextn`, producing a compatible Q4_K_M artifact.

The corrected file is approximately 2.7 GB and has SHA-256:

```text
30b3ac98715ad1e0aee49477606840e00d0e4339725f8c31e6f99ac56596f804
```

It is archived locally at:

```text
/Users/kester/Desktop/Agentic-Resume-Builder-Models/v4-1-q4_K_M/agentic-resume-v4-1-no-nextn-q4_K_M.gguf
```

The model was imported as `agentic-resume-v4-1:q4_K_M`. A local load test returned the requested exact phrase and did not expose a raw control token.

### Phase 13 — local V3 versus V4.1 assessment

Seven fixed validation cases were run through the V3 and V4.1 Ollama models with the production prompt and deterministic decoding. The frozen final benchmark remained unopened. Because V3 used NVFP4 and V4.1 used Q4_K_M, this was a deployment comparison rather than a pure adapter ablation.

| Automated check | V3 | V4.1 |
| --- | ---: | ---: |
| Required headings | 7/7 | 7/7 |
| No unsupported numeric strings | 6/7 | 7/7 |
| No raw terminal control token | 0/7 | 7/7 |
| Strict constraint phrase set | 6/7 | 6/7 |
| Mean output length | 154.1 words | 150.7 words |
| Mean request time | 22.74 seconds | 19.60 seconds |

Manual findings were:

- V4.1 was preferred on the two targeted QA cases because it preserved the actual title, manual scope, percentage limit, and QA-lead authority.
- V4.1 also preserved intermediate English, procurement authority, senior-engineer approvals, compliance sampling, counsel decisions, and acting-role limits.
- Three cases were ties because both models preserved the critical evidence.
- V3 was preferred on one Senior Data Engineer case because V4.1 inserted an awkward negative summary sentence about responsibilities the candidate did not own.
- Both models still adopted the target QA title in one older case where the source did not explicitly provide a title.

Prompt-only attempts did not reliably remove the negative summary framing. Inspection traced the behavior to repeated `without` constructions in several V4.1 training summaries. The production prompt changes were reverted so the model comparison remained controlled.

## 4. Current dataset composition and role coverage

The current 550-case set has three layers:

| Layer | Candidate families | Examples | Role coverage |
| --- | ---: | ---: | --- |
| V2 fictional pilot | 25 | 50 | 25 diverse business, technical, service, operations, and entry-level roles |
| Screened Djinni expansion | 225 | 450 | 150 frontend, 150 Java, 150 QA pairs |
| V4.1 fictional augmentation | 25 | 50 | 25 diverse constraint-heavy roles |
| Total | 275 | 550 | Broad pilot/augmentation coverage with a technology-heavy 450-case core |

The 450-case public-data core is deliberately concentrated in three occupation families because these were the public records that passed the evidence, privacy, chronology, compatibility, and duplication gates at the required volume. The fictional pilot and V4.1 augmentation supply broader industries and Singapore-oriented contexts.

This concentration is an important reporting limitation: the model has strong exposure to frontend, Java, and QA tailoring, plus smaller controlled coverage of finance, operations, support, procurement, data, HR, marketing, design, education, compliance, facilities, hospitality, retail, research, laboratory, and logistics work. It should not be presented as uniformly validated across every occupation.

## 5. What was verified

The workstream verified the following end-to-end properties:

- source and case schemas are versioned and hashable;
- records carry lineage, split, provenance, and evidence metadata;
- private and generated data remain ignored by Git;
- candidate and JD families do not cross train/validation boundaries;
- the frozen holdout does not overlap development sources;
- completion-only labels mask every prompt and padding token;
- all audited examples fit the sequence budget;
- expected Qwen3.5 language modules resolve and forbidden modules are excluded;
- a fixed smoke subset can be memorized;
- saved adapters reload deterministically;
- merged-model output matches adapter output for the verification sample;
- the corrected GGUF loads in local Ollama;
- V4.1 improves the targeted factual constraints on the fixed development comparison.

The complete V4 test suite recorded 96 passing tests after the V4.1 package changes. Tests cover contracts, splitting, holdout exclusions, validation, prompt masking, target resolution, non-overwrite behavior, and training preflight checks.

## 6. Limitations and honest interpretation

The current results should be interpreted with these limits:

1. The final benchmark is still unopened. All model judgments are development-set results.
2. The V4.1 A/B test contains seven cases, so it demonstrates specific behavior changes rather than broad statistical superiority.
3. V3 and V4.1 used different deployment quantizations, making their Ollama comparison practical but not a pure adapter-only experiment.
4. The public-data core is concentrated in frontend, Java, and QA roles.
5. Djinni dataset cards declare MIT, but original free-text consent and provenance could not be independently established. The data and derivatives remain local-development only.
6. Most bulk V2/V3 records did not receive independent case-by-case human certification. The V2 pilot, stratified V3 sample, all V4.1 additions, and flagged cases did receive direct review as documented.
7. Validation loss measures agreement with reference wording and is not a direct factuality score. Higher or lower loss alone cannot decide deployment quality.
8. V4.1 can write unnecessary negative responsibility statements in summaries. These are usually factual but reduce resume usefulness.
9. One older QA case still permits adoption of a target title when the source has no explicit title.
10. Optional accelerated DeltaNet kernels were not installed, increasing training time without changing the intended computation.

## 7. Current release state

| Artifact | State |
| --- | --- |
| Baseline `qwen3.5:4b-q4_K_M` | Current application default and rollback |
| V3 adapted model | Retained comparison model |
| V4.1 adapter and merge evidence | Verified and archived |
| V4.1 Q4_K_M GGUF | Converted, checksum-verified, and imported locally |
| V4.1 Ollama tag | `agentic-resume-v4-1:q4_K_M` |
| V4.1 production promotion | Not approved yet |
| Frozen final benchmark | Unopened |
| GitHub publication | Not performed |

## 8. Recommended next steps

The smallest high-value next iteration is V4.2:

1. Repair only the constraint examples that place missing authority in the summary. Preserve the same facts but express authority limits in the relevant experience bullet.
2. Add or repair a small number of title-preservation examples where the source lacks an explicit title and the JD title is more senior or specialized.
3. Keep all 500 V3 core records, splits, and holdout artifacts unchanged.
4. Re-run deterministic validation and manual review for every changed case.
5. Train with the same pinned base, environment, LoRA settings, seed, and completion mask.
6. Repeat the smoke, reload, merge, conversion, and fixed seven-case comparison.
7. Expand the development comparison with more role-balanced, constraint-specific cases before considering promotion.
8. Lock the recipe and evaluation protocol before opening the final benchmark once.
9. Promote a new Ollama tag only if factual safety, usefulness, formatting, and latency all pass; keep the current baseline as rollback.

More resumes per JD are not the immediate priority. The present defect is targeted wording learned from target examples, so correcting a small number of high-impact targets is more efficient than adding undirected volume.

## 9. Evidence and reproducibility index

### Tracked design and review documents

- `docs/v4/README.md` — repository scope and safety findings.
- `docs/v4/datasets.md` — public dataset research and source decisions.
- `docs/v4/training-plan.md` — data, training, evaluation, and deployment design.
- `docs/v4/data-pipeline.md` — local data pipeline and validation behavior.
- `docs/v4/pilot-50.md` — initial pilot record.
- `docs/v4/pilot-50-v2.md` — accepted 50-case pilot summary.
- `docs/v4/expansion-500.md` — preliminary hybrid expansion and 20-case costed gate.
- `docs/v4/session-trial-20.md` — trial findings that triggered source-inventory replacement.
- `docs/v4/inventory-450-review.md` — final 450-pair source audit.
- `docs/v4/target-polish-v3.md` — V3 editorial quality audit.
- `docs/v4/training-package.md` — pinned LoRA training package.

### Ignored local dataset evidence

- `runs/v4-pilot-50-v2/` — accepted 50-case fictional reference.
- `runs/v4-inventory-450-v1/inventory-044/` — screened source inventory and pairing audit.
- `runs/v4-dataset-500-v2/` — first complete 500-case training package.
- `runs/v4-dataset-500-v3-r4/` — polished 500-case package.
- `runs/v4-dataset-550-v4-1/` — current 550-case constraint-augmented package.

### Ignored local training and evaluation evidence

- `runs/v4-1-server-results/` — V4.1 smoke, training, reload, and merge evidence.
- `runs/v4-ollama-ab-v3/assessment.md` — baseline versus V3 assessment.
- `runs/v4-ollama-ab-v4-1/assessment.md` — V3 versus V4.1 assessment.
- `/Users/kester/Desktop/Agentic-Resume-Builder-Models/v4-1-q4_K_M/` — local GGUF archive and checksums.

### Main implementation commits

| Commit | Contribution |
| --- | --- |
| `78876c6` | V4 research and fine-tuning design |
| `85242b3` | Versioned data contracts and fixtures |
| `d0fff48` | Family-safe splits and benchmark exclusions |
| `f06920c` | Offline generation and validation pipeline |
| `77a8fd0` | Initial 50-case pilot |
| `bc886de` | Revised 50-case V2 pilot |
| `4837505` | Hybrid expansion preparation and 20-case dry run |
| `8df2594` | Session-trial and source-quality findings |
| `21f5565` | Public-source inventory |
| `9708a68`–`f8dd657` | Successive sanitization, privacy, identity, and source-gate hardening |
| `a6f47c2` | Qwen3.5 LoRA training package |
| `a487e36` | Transformers warmup compatibility fix |
| `5633fd7` | Polished V3 dataset configuration |
| `83f659c` | Deterministic adapter comparison |
| `035a7c8` | Verified adapter merge path |
| `a1a4406` | Inference constraint hardening |
| `90692ce` | Staged V4.1 remote workflow |
| `83fddff` | V4.1 constraint training package |
| `a281139` | Remote training gate correction |

## 10. Final status

The workstream has completed dataset construction, local experimental training, adapter verification, merging, quantization, import, and a controlled development comparison. The resulting V4.1 model is operational and materially better on the constraints it was designed to address. It remains a candidate rather than the application default because one learned writing pattern still reduces resume quality. The next work should be a narrow data correction and controlled V4.2 retrain, not a broad rewrite or unstructured dataset expansion.
