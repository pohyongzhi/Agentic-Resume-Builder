# V4 target polish V3

The accepted polished dataset is stored locally under the ignored path `runs/v4-dataset-500-v3-r4/`. It contains 500 SFT examples split into 450 training and 50 validation records. The original 50-case V2 pilot is preserved; the other 450 targets were rewritten without changing their source resumes, job descriptions, family assignments or splits.

The editorial pass removed repetitive source-copy construction while preserving evidence. Across the 450 rewritten targets, exact duplicate target groups fell from 12 to zero and normalized source-substring bullets fell from 2,317 of 2,632 (88.03%) to 5 of 2,544 (0.20%). Generic `documented work in` summaries, first-person targets, raw source labels, overlong skill lists and technology-only bullets were eliminated. Median target length remained essentially unchanged at 117 words, indicating editing rather than padding.

A stratified 45-record review covered frontend, Java and QA targets. Fourteen sampled records received narrow factual or copy-edit repairs and were rechecked. The final sample passed factual-support, application-usefulness and material-improvement review for all 45 records. The five remaining source-substring flags were individually inspected and are concise, supported facts. Deterministic content validation reports zero errors.

Generation metadata is recorded conservatively. Of the 450 polished records, 120 record `gpt-6-astra` with Low reasoning and an unavailable exact revision. The remaining 330 record an unavailable session model identity, reasoning setting and revision. Per-case token usage is unavailable. No paid API request was made.

Use `configs/v4/qwen35-4b-lora-v3.json` for the next experiment. Verify the local file contract with:

```bash
python3 -c 'from pathlib import Path; from v4_training.config import load_config; from v4_training.data import verify_training_inputs; c=load_config("configs/v4/qwen35-4b-lora-v3.json"); print(verify_training_inputs(c, Path.cwd()))'
```

On a node with the pinned tokenizer already cached, run the tokenizer and label audits before training:

```bash
python3 -m v4_training audit-tokenizer --config configs/v4/qwen35-4b-lora-v3.json --cache-dir "$HOME/v4-training/hf-cache" --local-files-only --examples 3
python3 -m v4_training audit-labels --config configs/v4/qwen35-4b-lora-v3.json --cache-dir "$HOME/v4-training/hf-cache" --local-files-only --examples 3
```
