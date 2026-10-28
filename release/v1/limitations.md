# Known limitations

- Only Workflow 1 and Workflow 4 are implemented. The other four stable IDs are experiment slots, not working techniques.
- V4.1 improved selected constraint-retention cases but was not consistently superior to the earlier comparison model. It produced undesirable negative framing in one reviewed summary.
- The seven-case local comparison is a development check, not a statistically representative quality benchmark. V3 and V4.1 also used different quantizations, so it is not a pure adapter ablation.
- The frozen holdout remains unopened for model selection and is not reported as a final benchmark result.
- The selected GGUF is available locally, but `adapter_model.safetensors` is not present in the verified local V4.1 evidence. Adapter configuration and training metadata are preserved; reproducing or recovering the adapter weights is still required for a complete weight-level archive.
- Generated resumes require human factual and editorial review before use.
- Sanitized public-source derivatives remain local-only. Dataset-card licenses were recorded, but original free-text provenance, consent and redistribution rights could not be independently guaranteed.
- Local paths in the manifest are machine-specific, and Ollama tags are not themselves portable model files.
