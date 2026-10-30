# Agentic Resume Builder 1.0.0

This local release establishes the six-workflow experiment contract while retaining Workflow 1 as the default control. Workflow 4 is the selected V4.1 fine-tuned candidate; it does not replace the baseline.

The release is reproducible from tracked code and configuration plus the ignored local artifacts recorded in `manifest.json` and verified in `checksums.sha256`. Large datasets and model weights are intentionally excluded from Git.

## Verification

From the repository root:

```bash
shasum -a 256 -c release/v1/checksums.sha256
python -m unittest discover -s tests -v
python run_all.py --workflows baseline fine_tuned_small_generator --limit 1 --repeats 1
```

The selected V4.1 tag remains historical and recoverable as `agentic-resume-v4-1:q4_K_M`. Rolling back the application means selecting `baseline`; it never requires deleting either model tag.

The recoverable cleanup archive is `/Users/kester/Desktop/Agentic-Resume-Builder-Archive-2026-10-28`. It contains the pre-cleanup inventory, a complete Git bundle, the former staged patch and copied cleanup candidates with SHA-256 verification.

## Model hosting

The selected artifacts are backed up on Hugging Face:

- Private adapter: `echokes/agentic-resume-finetuned-v1`
- Public runnable GGUF: `echokes/agentic-resume-finetuned-v1-GGUF`

Their pinned Hub revisions and verified large-file hashes are recorded in `manifest.json`. The dataset was not uploaded. The GGUF is public for school-project distribution; the adapter remains private.
