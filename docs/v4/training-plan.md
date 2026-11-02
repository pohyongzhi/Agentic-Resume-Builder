# V4 synthetic-data and training design

Proposed experiments, not tested training settings. Read [scope and repository findings](README.md) and [source decisions](datasets.md) first.

## Learning objective

Input: original resume and target JD. Output: a complete, concise Markdown resume emphasizing supported relevant experience. Every candidate claim must follow from the resume actually supplied at inference time. JD requirements guide selection and ordering; they cannot supply missing candidate facts. Do not train a fit classifier, recruiter decision maker or a generator of fictional achievements under the same target.

Start from the official post-trained [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), pinned to a commit. Further SFT on a post-trained checkpoint is appropriate. The Ollama Q4_K_M copy is an inference artifact; download the original trainable weights for LoRA. Pin tokenizer/template and verify provenance of the current Ollama baseline later.

## Synthetic example construction

1. **Register sources.** Record source URL/revision, source ID, declared license, upstream restrictions, intended local use, attribution, retrieval date, language and real/synthetic status. Inspect samples before bulk acquisition. Keep raw sources and generated corpora outside ordinary Git.
2. **Sanitize and deduplicate.** Remove direct identifiers and irrelevant sensitive attributes. Check residual free text and unique identifying combinations. Retain a local lineage ID; a hash is not anonymization. Group exact and near-duplicates across all source families, including mirrors. Detect prompt-like instructions in documents and treat all resume/JD text as data.
3. **Partition source families before pairing.** Allocate candidate clusters and JD/template clusters separately to train/validation/synthetic-test pools. Pair only within a pool. Source paraphrases, teacher retries and altered requirements inherit the original split. The protected final benchmark is a separate pool unavailable to generators and tuners.
4. **Prepare candidate facts independently of the JD.** For fictional records, author a plausible fact ledger with explicit skills/proficiency, responsibilities, employers, dates, education and optional metrics. Render an original resume and verify every usable fact is visible in it. For existing profiles, extract facts with verbatim evidence spans. Never enrich a real candidate using occupation stereotypes. An aspiration such as 'looking for Kubernetes work' does not demonstrate Kubernetes experience.
5. **Choose a JD with controlled gaps.** Include close, partial and poor fits, junior candidates facing senior requirements, missing certifications, missing tools, ambiguous dates, modest proficiency, sparse education and transferable experience. Do not always select jobs after constructing a perfect candidate.
6. **Generate the target.** Give the teacher only the visible source resume, JD, rules, and an evidence index derived from that resume. Request a Markdown target and a separate claim-to-source map. The final assistant target is only the resume, without the map, scores, reasoning or editorial commentary.
7. **Validate independently.** Check entities, numbers, dates, proficiency, ownership, negation and role relationships; then assess semantic support and relevance with a separate review pass. Require evidence for headers, summaries, skills and education as well as bullets. A matching number somewhere else in the resume is not support for a new achievement. Reject unsupported claims, copied JD requirements and fabricated metrics. Human-review uncertain cases; a teacher's own approval does not count as independent validation.
8. **Accept or repair once.** One evidence-constrained retry is a practical starting limit; reject after a second failure. Record rejection reasons and actual costs. Human-audit all 500 pilot outputs, double-review at least 100 stratified examples, and adjudicate disagreements. Freeze validator rules after development calibration, before final testing.

For sparse inputs, preserve the available evidence and allow a shorter output. A section with no supplied information may state 'Not provided' consistently with the agreed output contract. Do not expand a short profile into a long biography, invent education, or promote 'basic Python' to proficiency. Any future prompt clarification must apply equally to base and adapted models.

### Proposed corpus scale and coverage

Use 500 accepted development-only pairs to test the process. Once it is stable, target 5,000 total development examples including eligible pilot rows: about 4,000 training, 500 validation and 500 synthetic-test rows, achieved through whole-group assignment rather than slicing rows. Pilot examples exposed while designing rules belong in train/dev, never in a purported untouched test set. Exact counts may vary to preserve disjoint groups.

After the pilot, manually review every flagged/high-risk output and a reproducible stratified 10% sample of other accepted outputs. A confirmed fabrication triggers quarantine and investigation of the associated generation batch, not merely removal of the sampled row. Sampling does not prove all unsampled outputs are error-free. Keep the 500-example synthetic test unopened until the recipe is locked, alongside the separately authored final benchmark.

For the 5,000-row design, target at least 2,500 candidate families and 1,250 JD families, normally at most two targets per candidate and four per JD. Use fictional resumes for roughly 80% and audited Djinni profiles for roughly 20%; compare the mixture with a fictional-only ablation. Proposed task mix: 40% close fit (2,000 rows), 25% partial fit (1,250), 15% poor fit/career transition (750), 15% sparse/entry-level cases (750), and 5% noisy-format/injected-instruction cases (250). Use one tenth of these counts for the pilot. These are mutually exclusive primary buckets; assign overlapping stress tags separately. Keep normal well-supported examples so the model does not learn to omit everything.

Balance across data/analytics, software/IT, business/operations, sales/support, finance and adjacent graduate roles. Use independently authored Singapore-oriented fictional scenarios where licensed local postings are unavailable. Broaden to other sectors only with adequate facts and meaningful validation. Use a small matched pilot to choose the teacher; verify its terms permit the intended downstream fine-tuning and estimate total cost before generation. Record model/version, prompt hash, parameters and usage. No provider or spending is selected here.

### Proposed archival JSONL schema

Each row is one accepted example. Required groups:

| Field | Contents / constraint |
|---|---|
| `schema_version`, `example_id`, `split` | Stable version and ID; split assigned by lineage. |
| `candidate_cluster_id`, `jd_cluster_id`, `generation_family_id` | All variants stay in one split. |
| `sources` | Resume/JD source IDs, URLs, revisions, licenses, usage decision, retrieval date, synthetic flags. |
| `input.original_resume`, `input.target_jd` | Sanitized exact text the student will receive. |
| `facts` | Fact IDs, normalized claim, source character spans, evidence text and confidence. Only facts visible in `input.original_resume` are eligible. |
| `target.resume_md` | The only supervised assistant answer. |
| `target_evidence` | Output spans mapped to source fact IDs; relation/proficiency checks. Metadata, never student input. |
| `quality` | PII checks, duplicate checks, deterministic findings, independent support/relevance review, human review and acceptance decision. |
| `generation` | Teacher ID/revision, prompt version/hash, parameters, retries, time and usage. |
| `hashes` | Original and sanitized source hashes, target hash, normalization version. |

Training export uses conversational prompt/completion JSONL:

```json
{
  "prompt": [
    {"role": "system", "content": "<versioned rules shared with inference>"},
    {"role": "user", "content": "JOB DESCRIPTION:\n<target JD>\n\nCANDIDATE'S ORIGINAL RESUME:\n<original resume>\n\nTailor the resume to this job."}
  ],
  "completion": [
    {"role": "assistant", "content": "<accepted Markdown resume>"}
  ]
}
```

The angle-bracket values are schema placeholders, not training examples. Export only prompt/completion to the trainer. Keep facts, source licenses, judge feedback and IDs in the archival record. A source ledger unavailable to the deployed model must never become a hidden dependency of the correct answer.

[TRL supports conversational prompt/completion data and completion-only loss](https://huggingface.co/docs/trl/sft_trainer). Use the model's chat template, non-thinking configuration, correct EOS and verified prompt/output boundaries. Inspect labelled tokens manually: system/user/padding labels must be masked and resume/EOS tokens supervised. Avoid assuming that a template's assistant mask works without testing it.

## Leakage protection and evaluation

Before teacher generation, independently author and freeze a final benchmark, proposed 100 candidate families × 2 JDs = 200 pairs, with a separate evaluator-owned fact/requirement annotation file. Include at least 40 Singapore-context pairs. This is additional to the 5,000 development examples and remains an academic benchmark; report uncertainty, domain limits and grouped results. Keep the original one-pair fixture as development only. Prefer more independent candidates over many pairings of the same person. If review capacity requires a smaller benchmark, decide its size before any model selection and disclose the resulting uncertainty.

Hash final candidate and JD files, build exact/near-duplicate exclusion manifests and exclude all their derivatives from training, validation, teacher demonstrations, prompt tuning and validator calibration. The generator can receive exclusion hashes and a filter service, not final benchmark texts. Use separate development annotations to build validators. Document historical exposure and replace any benchmark case used for debugging. Freeze file hashes plus evaluation protocol before training.

Split candidate identity clusters and JD/company-template families jointly. For existing paired data, connected components of the candidate/JD graph must remain together; where this creates a giant component, discard crossing pairs or keep the corpus out of the evaluation pools. For newly generated pairs, assign source pools first. Detect cross-source copies, translation/paraphrase families and template reuse. A random row split is inadequate.

Report paired comparisons with candidate-cluster bootstrap intervals. If JDs are reused across candidate families, also account for JD clustering (for example a two-way clustered sensitivity analysis); do not treat every pairing as independent:

- Unsupported claims per resume and per claim, plus severity (fabricated employer, degree, dates, metric or skill). Human adjudication is the reference for the final sample.
- Coverage of JD requirements actually supported by the original resume; preserve supported skill specificity and proficiency.
- Blinded reviewer preference for usefulness, clarity and relevance, with factuality a prerequisite.
- Format completeness, unsupported omission of important facts, output length, inference latency and tokens.
- Existing NLI, embedding similarity and keyword scores as secondary diagnostics, with their limitations documented.

Pilot gates: no known unsupported claim survives acceptance; all accepted targets have evidence maps; no split/holdout collision; no direct PII in accepted text; at least 95% of the human-audited pilot meets relevance/format expectations after factuality screening. If fewer pass, revise generation and filtering, not the target quota. Zero observed errors is not a statistical guarantee. For the eventual model, pre-register a non-inferiority tolerance for factual errors on development data before the sweep; choose among eligible checkpoints by relevance/coverage. If no model passes, retain the base model.

## L40S training proposal

The official [4B configuration](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json) has a hybrid language stack and a vision component. [Transformers documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5) distinguishes text-only and conditional-generation classes. Prototype the pinned checkpoint load first. Either use its verified text-only loading path or the full model with vision weights frozen; verify no silent language-weight mismatch and keep an export-compatible layout.

These are starting hypotheses for one 48GB L40S, not guarantees of memory fit or optimality:

| Parameter | First experiment | Selection rationale |
|---|---|---|
| Method/precision | BF16 LoRA, frozen original weights | Practical first test on 48GB; QLoRA NF4/double quantization with BF16 compute is a memory fallback, separately recorded. |
| LoRA rank / alpha | `r=16`, `alpha=32` | Modest starting capacity; compare rank 32 with alpha 64, holding scaling convention fixed. |
| Targets | Language attention, DeltaNet projections and MLP linear layers | Inspect actual module names; avoid an attention-only recipe copied from another architecture. Exclude vision, output head and embeddings by default. |
| Dropout / bias | `0.05`, `none` | Conservative small-data regularization; reconsider only after learning curves. |
| Learning rate | `1e-4` | Compare `5e-5` and `1e-4` at each rank first. Consider `2e-4` only if evidence suggests underfitting. |
| Optimizer | AdamW, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01` | Transparent baseline; record parameter decay groups. |
| Schedule | Cosine, 5% warmup, gradient clip `1.0` | Freeze within the first sweep. |
| Duration | Up to 3 epochs, inspect each epoch | Select early checkpoint; avoid assuming more epochs help. |
| Effective batch | 16 examples; microbatch 1 × accumulation 16 initially | Try microbatch 2 × accumulation 8 only after profiling, maintaining effective batch. Log actual supervised tokens per update. |
| Sequence budget | Start by testing 4,096 total tokens; consider 8,192 after length analysis | Includes prompt and answer. Quarantine overlength rows or preserve evidence through reviewed restructuring; never silently truncate facts or targets. |
| Memory options | Gradient checkpointing; cache off during training | Benchmark implementation and kernel support. |
| Packing | Off for initial runs | Hybrid recurrent state and attention must both reset between examples before packing is enabled. |
| Loss | Completion-only token cross-entropy | Verify masks and EOS. No teacher reasoning in targets. |
| Seeds | 42 for sweep; confirm winner with 43 and 44 if budget allows | Record dataset order and library/driver versions; quantify variance. |

[PEFT allows explicit LoRA target selection](https://huggingface.co/docs/peft/package_reference/lora). On this architecture, audit the resolved module list instead of applying `all-linear` to the whole multimodal model. Candidate suffixes include `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`, scoped to the language backbone. Verify against the installed implementation and export support. Do not change vocabulary or train the vision tower for text-only resume tailoring.

Experiment order: 8–16-example overfit/debug run → 500-example pilot (roughly 450 train / 50 validation, whole families) → four rank/LR combinations at the same data budget → carry the top two stable recipes to the full training pool → choose using the 500-example validation pool → seed confirmations → lock → one final-test evaluation. The small pilot validation pool diagnoses obvious instability; it cannot establish a robust optimum. Preserve all pilot-family split assignments during scaling. Save at epoch boundaries and at intervals appropriate to small dataset size (roughly quarter-epoch), not a fixed interval longer than the whole run. Keep an untouched base-model control and a fictional-only data ablation. Use validation to decide whether expanding beyond 5,000 examples is worth the cost.

No credible exact runtime is available yet. Measure 50–100 representative optimizer steps including warmup and maximum-length batches; estimate total runtime from measured step time plus evaluation/export overhead. Profile peak memory with a safety margin. Log supervised tokens, examples, optimizer steps and GPU hours so configurations are comparable.

## Later Slurm and Ollama implementation

1. Obtain SMU partition/account/QoS, allocation syntax, job time/memory limits, storage and environment policy. Use VPN/SSH at that stage. Keep GPU work on allocated compute nodes; do not assume login-node GPU/network availability.
2. Build and pin a compatible environment: Python, PyTorch/CUDA, Transformers, PEFT, TRL, Accelerate and optional quantization/DeltaNet kernels. Inspect licenses and compatible binary builds. Record exact versions/revisions; do not leave dependencies on unpinned `main`.
3. Stage the official checkpoint and approved data into permitted storage. Use hashes, a data manifest and an offline cache where compute nodes lack internet. Keep telemetry/logging local by default.
4. Submit a one-GPU smoke job, verify actual L40S allocation, masks, finite gradients, trainable-module inventory, save/reload equivalence and peak memory. Establish export viability before spending on a long sweep.
5. Train and save adapter weights/configuration, base revision, tokenizer/template, selected hyperparameters, dataset hashes, logs, metrics and code commit. Retain separate resume checkpoints if needed; never commit large weights to ordinary Git.
6. Reload the exact original base in appropriate precision, merge the selected adapter, check outputs, convert with a pinned compatible llama.cpp build and quantize to Q4_K_M. Validate DeltaNet adapter merging/conversion explicitly. A text-only training wrapper may require reconstructing the original checkpoint layout for conversion; do not assume generic Qwen2 export scripts work.
7. Import the prepared GGUF into a distinct local Ollama model. Current [Ollama import guidance](https://docs.ollama.com/import) documents GGUF import; do not assume direct Qwen3.5 adapter import is supported. If export fails, keep a Transformers + PEFT inference path as the development fallback and disclose the runtime change.
8. Create base and adapted GGUF controls through the same conversion/quantization pipeline. Match chat templates, non-thinking mode, context, sampling and output limits. Check pre/post-quantization factuality on development cases. Add `v4_lora` via the shared tracking path, keeping other workflows intact and giving each experiment unique result identity.
9. After the recipe is locked, run the frozen final benchmark once, report all cases, and keep adapter and results local. Copying artifacts from the cluster requires the normal permitted transfer path; no GitHub push is part of this workstream.
