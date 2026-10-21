"""One place for every setting. Keep these IDENTICAL for all workflows so the comparison is fair."""

MODEL = "qwen3.5:4b-q4_K_M"      # writer model for every workflow (Ollama tag)
EMBED_MODEL = "nomic-embed-text"  # embedding model for cosine similarity (ollama pull nomic-embed-text)

TEMPERATURE = 0.0     # low randomness so differences come from the technique, not luck
SEED = 42             # fixed seed passed to Ollama so runs are reproducible
NUM_CTX = 16384       # Ollama's default context is small and silently cuts off long prompts. Always set this.
THINK = False         # Qwen thinking mode. Same for all workflows (or make it its own experiment).
MAX_STEPS = 8         # step budget for agent loops (Topic 7: budgets stop runaway loops)

# Harness around the resume creator (common/harness.py)
MAX_TOOL_CALLS = 6        # creator tool calls allowed before the run fails with TOOL_BUDGET_EXCEEDED
OUTPUT_DIR = "results/outputs"   # rendered resumes can only be saved inside this folder

# Extraction thresholds (common/tools.py)
PDF_MIN_CHARS_PER_PAGE = 50   # below this a PDF is treated as scanned (no text layer)
JD_MIN_CHARS = 300            # a fetched job page with less readable text than this counts as a failed fetch
FETCH_TIMEOUT = 15            # seconds

# NLI model used as an independent referee for faithfulness (runs on CPU, separate from the writer).
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
ENTAIL_THRESHOLD = 0.5   # a claim counts as "supported" if entailment prob >= this

DATA_DIR = "data"
RESULTS_FILE = "results/results.jsonl"
