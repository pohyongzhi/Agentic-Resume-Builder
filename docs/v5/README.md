# V5 workflow comparison

V5 integrates the selected V4.1 fine-tuned model into the current multi-workflow application. It
does not change product version `1.0.0`, replace the base model or rename historical training
artifacts.

## Current workflow map

| No. | Runtime ID | Technique | Model | Status |
| ---: | --- | --- | --- | --- |
| 1 | `w1_single_pass` | One prompt, no tools or harness | `qwen3.5:4b-q4_K_M` | Implemented |
| 2 | `w2_baseline` | Parallel extraction, creator, validation and rendering | `qwen3.5:4b-q4_K_M` | Implemented |
| 3 | `w3_critic` | Fact, recruiter and formatting critics with bounded revision | `qwen3.5:4b-q4_K_M` | Implemented |
| 4 | `w4_evidence` | Source-span grounding and claim verification | `qwen3.5:4b-q4_K_M` | Implemented |
| 5 | `w5_fine_tuned` | One-pass generation with the selected V4.1 model | `agentic-resume-v4-1:q4_K_M` | Implemented |
| 6 | `w6_best_of_n` | Three strategy drafts and a two-order selector | `qwen3.5:4b-q4_K_M` | Implemented |
| 7 | `w7_plan_execute` | Planner and section writers | base model | Planned; not registered |

The older stable ID `fine_tuned_small_generator` remains in historical release records only.
Current executable code uses `w5_fine_tuned` so its number matches the project workflow table.
The historical ATS optimiser is not part of this comparison.

## Model routing contract

- `common.llm.chat()` uses `config.BASE_MODEL` when no model is supplied.
- Only W5 passes `config.FINE_TUNED_MODEL` explicitly.
- Every model call logs the actual selected tag.
- The batch runner stores the expected and actual workflow model in its cache identity and results.
- The web application opens on W1, never W5.

Tests use a fake Ollama client to prove identical resume/JD inputs route W1 to the base model and W5
to the fine-tuned model. Live output quality must still be measured with the same frozen dataset and
shared evaluator used for other workflows.

## Fine-tuned model distribution

The runnable Q4_K_M GGUF is public and downloadable without an account:

```text
echokes/agentic-resume-finetuned-v1-GGUF
```

The exact account-free import commands are in `HOW_TO_RUN.md`. The private adapter and local-only
training data are not required to run W5. Training provenance, hashes and limitations remain in
`release/v1/` and `docs/v4/`.

## Known limitation

The fine-tuned model is functional but has not shown a consistent overall advantage in the small
development comparison. Some outputs used undesirable negative framing. Treat W5 as one experimental
workflow and compare it fairly rather than presenting it as the product baseline.
