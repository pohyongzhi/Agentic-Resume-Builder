"""Shared product and experiment settings."""

PRODUCT_NAME = "Agentic Resume Builder"
PRODUCT_VERSION = "1.0.0"
EXPERIMENT_VERSION = "V5"

BASE_MODEL = "qwen3.5:4b-q4_K_M"
FINE_TUNED_MODEL = "agentic-resume-v4-1:q4_K_M"

# Backwards-compatible name for code that expects the shared/default writer.
# It must remain the original base model; V5 Variant 5 opts into FINE_TUNED_MODEL.
MODEL = BASE_MODEL
EMBED_MODEL = "nomic-embed-text"  # embedding model for cosine similarity (ollama pull nomic-embed-text)

TEMPERATURE = 0.0     # low randomness so differences come from the technique, not luck
NUM_CTX = 16384       # Ollama's default context is small and silently cuts off long prompts. Always set this.
NUM_PREDICT = 2048    # hard output ceiling for a complete evidence-limited resume
THINK = False         # Qwen thinking mode. Same for all workflows (or make it its own experiment).
MAX_STEPS = 8         # step budget for agent loops (Topic 7: budgets stop runaway loops)

# NLI model used as an independent referee for faithfulness (runs on CPU, separate from the writer).
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
ENTAIL_THRESHOLD = 0.5   # a claim counts as "supported" if entailment prob >= this

DATA_DIR = "data"
RESULTS_FILE = "results/results.jsonl"
