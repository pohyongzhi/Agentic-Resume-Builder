# V4 fine-tuning research and design

Status: local data foundation implemented and a 50-case fictional development pilot drafted, 24 October 2026. Pilot acceptance remains pending independent review, a frozen benchmark exclusion manifest and complete generation metadata. No external corpus acquisition, paid generation, GPU training, application changes, or remote publication has been performed.

The source, licensing and synthetic-design reviews are complete. A fresh independent final judging pass remains pending; the cross-source challenge is documented with its independence limitation. Recommendations remain provisional until that review and the later data pilot.

## Scope and repository

Active checkout: `/Users/kester/Desktop/Agentic-Resume-Builder`, local branch `v4-finetuning-workstream`, based on `Baseline` commit `60132d2181df5005af834771ceee6d04b304af86`. Keep changes local. Commit coherent milestones with one-line messages. Do not push. An earlier unchanged clone exists in the ChatGPT project directory; it is not the active checkout.

The current phase adds a local data preparation and validation foundation to the documented source research, licensing/privacy assessment, synthetic-data design, evaluation policy, and proposed training/export plan. VPN, SSH, cluster setup, paid generation, bulk downloads and training are later phases. This is an academic project; prefer sources that support future reuse, but record any restrictions rather than assuming academic use clears them.

- [Dataset evidence and decisions](datasets.md)
- [Synthetic data, evaluation and training plan](training-plan.md)
- [Local pipeline, contracts, split safeguards and verification](data-pipeline.md)
- [50-example development pilot and validation limits](pilot-50.md)

## Repository findings

The configuration selects `qwen3.5:4b-q4_K_M`, non-thinking mode, temperature 0 and a 16,384-token inference context. `common/llm.py` constructs the default Ollama client; environment configuration can override its endpoint. This establishes the intended local inference path, not proof that Ollama is currently running or that its installed weights have been identified.

`baseline_tailor.py` sends the original resume and JD in one request. `common/prompts.py` requires factual support and four Markdown sections. `common/data.py` loads evaluation pairs from `data/pairs.csv`. `run_all.py` scores outputs and appends results; `eval/metrics.py` provides embedding similarity, keyword coverage and chunk-based NLI faithfulness. Only `baseline` and `w4_ats_optimiser` are registered. The latter is an ATS loop, not a fine-tuned model. Use a distinct future identifier such as `v4_lora` to avoid overwriting it.

### Decision: current data are development fixtures

There is exactly one pair: a fictional 136-word resume and a 92-word JD explicitly labelled a sample. The README calls these placeholders and proposes a future 20-pair benchmark. They have already been inspected during development. Preserve these files as smoke/development fixtures and exclude them from SFT. Do not present them as an unseen final benchmark. No final benchmark is frozen yet.

Existing fixture SHA-256 values, for future exclusion checks:

| File | SHA-256 |
|---|---|
| `data/pairs.csv` | `d7ddd709ac41e15fb13b57c78b334154a8b871dc309f2962c85ab160bf49c2b6` |
| `data/resumes/cand_01.md` | `694ed0c4e8b2227c5a94a84f878221140c9ce9f814435df2770c6b0be6e78821` |
| `data/jds/jd_01.md` | `7cd59e0c9422c7e5def8af1d30f8212c261baec5d719242ad736362c14302b6d` |
| `data/jds/jd_01.keywords.json` | `953ef6222effdbeec56acde5e24aa85381829af2c7c35cfee255643c4c8546e9` |

### Measurement issues to address during implementation

1. A 350–500-word preference is inappropriate for some sparse inputs. Add a consistent evidence-limited length exception to all compared prompts before freezing experiments. Missing education must remain unknown, not be supplied by a teacher. Avoid changing the baseline prompt only for the adapter.
2. NLI ignores headings and short lines, so invented short skill lists, dates and titles can escape scoring. Its maximum entailment across 60-word chunks can also miss evidence or accept partial support. It is one signal, not an acceptance certificate. Verify the loaded model's label mapping and calibrate on human-labelled claims.
3. Keyword coverage rewards JD requirements even when the candidate lacks them. The sample labels Power BI and Tableau as synonyms although competence in one does not prove competence in the other. Retain legacy scores for comparison, but add supported-requirement coverage and unsupported-claim measurements.
4. Result caching uses workflow/pair/repeat, omitting model digest, checkpoint and prompt revision. Future runs need unique experiment IDs or separate result paths to prevent stale baseline/adapter results being skipped or mixed.
5. Repeats at temperature 0 are not independent candidate samples. Report uncertainty over candidate/JD groups; measure repeated timings separately.
6. SFT changes model weights. The comparison is an adaptation experiment, not solely a comparison of agentic techniques with identical weights. Report training and data-generation costs separately from inference costs.

## Recommended strategy

Build a new corpus of faithful full-resume rewrites. Public matching scores, paired documents and resume classifications are useful source material but generally provide no correct tailored target. Search results did not establish a ready-to-use, reliably factual and clearly licensed rewrite corpus; this is a finding within the inspected sources, not proof none exists.

The preferred pilot combines controlled fictional candidate facts with selected, privacy-checked Djinni English profiles and JDs. Use O*NET/ESCO for occupational coverage and skill normalization, not as evidence of a candidate's qualifications. Inspect direct resume-rewriting datasets as references only until their rights and factuality issues are resolved. See the evidence table for conditional alternatives.

Initial proposal: 500 accepted development examples, then approximately 5,000 accepted examples if the pilot passes quality gates. Quantity is a planning target, not a demonstrated optimum. Document the real/synthetic mix and all teacher provenance. Select hyperparameters through validation experiments; the final test must never guide model selection.

## Milestones and completion criteria

| Stage | Deliverable and gate |
|---|---|
| Research/design (documented) | Source inventory, reviewed shortlist, synthetic recipe, split protocol, architecture-aware training plan; independent final adjudication remains pending. |
| Data implementation (foundation complete) | Versioned contracts, source admission, component splitting, holdout fingerprints, offline fixture generation and deterministic validation. Real-source cleaning/admission, teacher integration, independent acceptance and a separately frozen final benchmark remain pending. |
| Generation pilot (50-case draft complete) | 50 fictional development cases with evidence mappings are available locally; none has passed the full archival acceptance gate. The proposed 500 accepted cases, independent review and human audit remain later milestones. Teacher choice and budget must be settled before any paid request. |
| Cluster smoke run (later) | Approved Slurm environment, one GPU allocation, correct loss masks, finite gradients, adapter reload and export checks. |
| Controlled training (later) | Small validation sweep, locked winner, seed repeat, documented throughput/memory and cost. |
| Final evaluation/integration (later) | Base versus adapter comparison on untouched final cases; same export pipeline and inference settings; additional workflow without replacing baseline. |

Cluster-specific partition, account/QoS, modules/container policy, storage quotas, time limit and networking must be obtained before producing a runnable site-specific submission. VPN/SSH access is only needed then. No credentials belong in this repository.
