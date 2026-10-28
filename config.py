"""One place for every setting. Keep these IDENTICAL for all workflows so the comparison is fair."""

import os

DEFAULT_MODEL = "agentic-resume-v4-1:q4_K_M"
BASE_MODEL = "qwen3.5:4b-q4_K_M"
MODEL = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)  # writer model for every workflow (Ollama tag)
EMBED_MODEL = "nomic-embed-text"  # embedding model for cosine similarity (ollama pull nomic-embed-text)

TEMPERATURE = 0.0     # low randomness so differences come from the technique, not luck
NUM_CTX = 16384       # Ollama's default context is small and silently cuts off long prompts. Always set this.
THINK = False         # Qwen thinking mode. Same for all workflows (or make it its own experiment).
MAX_STEPS = 8         # step budget for agent loops (Topic 7: budgets stop runaway loops)

# NLI model used as an independent referee for faithfulness (runs on CPU, separate from the writer).
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
ENTAIL_THRESHOLD = 0.5   # a claim counts as "supported" if entailment prob >= this

DATA_DIR = "data"
RESULTS_FILE = "results/results.jsonl"
