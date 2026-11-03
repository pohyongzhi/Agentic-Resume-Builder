# Known limitations

- Only Workflow 1 and Workflow 4 are implemented. The other four stable IDs are experiment slots, not working techniques.
- V4.1 improved selected constraint-retention cases but was not consistently superior to the earlier comparison model. It produced undesirable negative framing in one reviewed summary.
- The seven-case local comparison is a development check, not a statistically representative quality benchmark. V3 and V4.1 also used different quantizations, so it is not a pure adapter ablation.
- The frozen holdout remains unopened for model selection and is not reported as a final benchmark result.
- The selected GGUF and V4.1 adapter are available outside Git in the local model store. The GGUF is publicly mirrored for school-project use; the adapter remains private. The local absolute manifest paths are machine-specific.
- Generated resumes require human factual and editorial review before use.
- The training dataset and sanitized public-source derivatives remain private and local-only. Dataset-card licenses were recorded, but original free-text provenance, consent and redistribution rights could not be independently guaranteed. The public GGUF has not undergone a formal legal or model-memorization review.
- Local paths in the manifest are machine-specific, and Ollama tags are not themselves portable model files.
