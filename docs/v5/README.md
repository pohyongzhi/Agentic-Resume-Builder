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

## Next implementation order

Implement Variant 2 first because it establishes reusable parsing, validation and rendering contracts. Then implement evidence grounding, the critic loop, multi-draft selection and planner/section writers. Each variant requires unit tests, identical-input routing tests and development-set review before the frozen holdout is opened.
