# W5 merge status

Updated: 2026-11-02

## Completed

- Merged `origin/main` at `37ee76b` into local `v5-workflow` without rewriting history.
- Kept current main implementations of W1, W2, W3, W4 and W6.
- Added executable `w5_fine_tuned` using `agentic-resume-v4-1:q4_K_M`.
- Kept `qwen3.5:4b-q4_K_M` as the shared default and only allowed W5 to override it.
- Logged the actual model tag for every shared model call and batch result.
- Added W5 to the web selector while keeping W1 as the default.
- Removed obsolete duplicate workflow placeholders.
- Preserved the original V4 exclusion fixtures under `data/legacy/` while retaining main's current
  evaluation dataset at `data/pairs.csv`.
- Updated setup and model-distribution documentation.
- Passed `327` tests and `21` subtests. The branch-owned code/documentation whitespace check
  passed; merged main evaluation artifacts retain pre-existing Markdown hard breaks and CSV line
  endings that a whole-index `git diff --check` reports as trailing whitespace.

## Remaining before merging to main

1. Commit the resolved merge on `v5-workflow`.
2. Push `v5-workflow` and open or update its pull request.
3. Confirm CI passes.
4. Run one live identical-input W1/W5 smoke comparison when Ollama is running and both tags are
   installed. Unit routing is verified, but this merge session did not restart Ollama.
5. Merge the pull request only after reviewing the 83-file branch diff, which includes the retained
   V4 dataset/training provenance and reusable training code.

## Evaluation status

The shared evaluator and faithfulness work from current main are present. Historical W5 development
evidence covers seven cases and is not a complete workflow evaluation. A full frozen-set W5 run is
still required before making comparative quality claims.
