# V5 seven-variant experiment

V5 expands the comparison design from the V1 six-workflow registry to the seven variants specified in the project architecture. It does not change product version `1.0.0` or rename the historical V4.1 model experiment.

## Stable mapping

| No. | Stable ID | Intended architecture | V5 status |
| ---: | --- | --- | --- |
| 1 | `one_pass` | Raw resume and JD to one base-model call | Implemented |
| 2 | `agentic_baseline` | Resume parser and JD retriever feed a creator that uses validation and rendering tools | Planned |
| 3 | `multi_critic_review` | Fact-checker, recruiter and formatting critics review a draft within a bounded revision loop | Planned |
| 4 | `evidence_grounded_generation` | Every generated claim cites source evidence and a verifier rejects unsupported claims | Planned |
| 5 | `fine_tuned_small_generator` | Selected V4.1 LoRA-derived model performs one-pass tailoring | Implemented |
| 6 | `multi_draft_selector` | Several strategies produce drafts; a selector chooses the best | Planned |
| 7 | `planner_section_writers` | Planner maps requirements to evidence; section writers draft in parallel; assembler verifies consistency | Planned |

## Migration from V1

- V1 `baseline` becomes V5 `one_pass` with identical runtime behavior.
- V5 reserves `agentic_baseline` for the separate multi-component design; no existing one-pass code is silently relabelled as agentic.
- The selected V4.1 fine-tuned model moves from registry position 4 to V5 position 5 without changing its Ollama tag, weights or provenance.
- `multi_critic_review`, `evidence_grounded_generation`, `multi_draft_selector` and `planner_section_writers` retain their stable IDs but move to their V5 positions.
- The historical ATS optimiser example remains excluded from the comparison.

## Fairness rules

1. Identical resume/JD inputs and shared base prompt.
2. Base model for Variants 1, 2, 3, 4, 6 and 7.
3. Fine-tuned model only for Variant 5.
4. Every model call goes through `common.llm.chat()` and records the selected model, tokens and time.
5. Only implemented variants are executable.
6. Planned variants must be implemented as their documented technique, not as aliases to one-pass generation.
7. The frozen holdout remains closed until all intended variants and the evaluation protocol are finalized.

## Completed V5 foundation

The V5 foundation is implemented on branch `v5-workflow`:

- the ordered seven-variant registry is the single source of identity, model route and implementation status;
- the web application defaults to Variant 1 and displays only implemented variants;
- planned variants fail closed in the evaluation CLI;
- Variant 1 uses `qwen3.5:4b-q4_K_M` through the shared wrapper;
- Variant 5 explicitly uses `agentic-resume-v4-1:q4_K_M` through the same wrapper;
- every model call records its actual model tag;
- product version `1.0.0`, experiment V5 and historical model experiment V4.1 remain separate;
- the complete repository suite passes 100 tests.

A live identical-input test produced the four required Markdown headings for both implemented variants. Variant 1 traced the original base tag, Variant 5 traced the fine-tuned tag, and neither output contained a raw terminal control token.

## Team distribution

Source code is available from the GitHub `v5-workflow` branch. The selected runnable model is publicly downloadable without a Hugging Face account from:

```text
echokes/agentic-resume-finetuned-v1-GGUF
```

Its Hub-side size and SHA-256 match the local release record. The portable Modelfile was tested through a fresh temporary Ollama import and deterministic resume generation. The LoRA adapter remains private at `echokes/agentic-resume-finetuned-v1`, and the training dataset was not uploaded.

Detailed teammate commands are maintained in `HOW_TO_RUN.md`. The application currently exposes Variant 1 and Variant 5; this is intentional and not evidence that the five planned techniques are complete.

## Next implementation order

Implement Variant 2 first because it establishes reusable parsing, validation and rendering contracts. Then implement evidence grounding, the critic loop, multi-draft selection and planner/section writers. Each variant requires unit tests, identical-input routing tests and development-set review before the frozen holdout is opened.
