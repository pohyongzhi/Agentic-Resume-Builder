import os
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
SEED = 42             # fixed seed passed to Ollama so runs are reproducible
NUM_CTX = 16384       # Ollama's default context is small and silently cuts off long prompts. Always set this.
NUM_PREDICT = 2048    # hard output ceiling for a complete evidence-limited resume
THINK = False         # Qwen thinking mode. Same for all workflows (or make it its own experiment).
MAX_STEPS = 8         # step budget for agent loops (Topic 7: budgets stop runaway loops)
MAX_CRITIC_ROUNDS = 2  # W3 may revise at most twice after the initial three-critic review
W3_GENERATION_VERSION = 4
W3_STRUCTURED_RETRIES = 1
W3_DRAFT_MAX_TOKENS = 4096
W3_CRITIC_MAX_TOKENS = 1536

# Harness around the resume creator (common/harness.py)
MAX_TOOL_CALLS = 6        # creator tool calls allowed before the run fails with TOOL_BUDGET_EXCEEDED
OUTPUT_DIR = "results/outputs"   # rendered resumes can only be saved inside this folder

# Extraction thresholds (common/tools.py)
PDF_MIN_CHARS_PER_PAGE = 50   # below this a PDF is treated as scanned (no text layer)
JD_MIN_CHARS = 300            # a fetched job page with less readable text than this counts as a failed fetch
FETCH_TIMEOUT = 15            # seconds

# W6 Best-of-N: strategy prompts provide diversity while the shared TEMPERATURE stays identical
# to the baseline. Choose 1-6; the default uses the three required strategies.
W6_NUM_CANDIDATES = 3

# NLI model used as an independent referee for faithfulness (runs on CPU, separate from the writer).
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
ENTAIL_THRESHOLD = 0.5   # a claim counts as "supported" if entailment prob >= this
TASK_FAITHFULNESS_THRESHOLD = 0.8  # minimum claim support for evaluator task success
TASK_MIN_WORDS = 200
TASK_MAX_WORDS = 650

# Evaluator configuration. These settings are deliberately separate from MODEL and do not
# affect generation cache identity.
JUDGE_PROVIDER = "ollama"
JUDGE_MODEL = "gemma4:e4b"
JUDGE_TEMPERATURE = 0.0
JUDGE_SEED = 42
JUDGE_NUM_CTX = 16384
JUDGE_THINK = False
JUDGE_REASONING_EFFORT = "medium"
JUDGE_MAX_COMPLETION_TOKENS = 2048
# Minimum spacing between hosted judge requests. Override with HOSTED_MIN_INTERVAL_SECONDS
# (e.g. 55 on the Groq free tier, whose per-minute token limit counts max_completion_tokens).
JUDGE_MIN_REQUEST_INTERVAL_SECONDS = float(os.environ.get("HOSTED_MIN_INTERVAL_SECONDS", "15"))
JUDGE_MAX_RETRIES = 2
JUDGE_PROMPT_VERSION = "resume-rubric-v4"
JUDGE_SCHEMA_VERSION = "1.0"
# 2.3 adds supportable-requirement coverage and per-component identities to evaluation records.
EVALUATOR_SCHEMA_VERSION = "2.3"
NLI_EVALUATOR_VERSION = "claim-evidence-nli-v4"
PROTECTED_FIELD_EVALUATOR_VERSION = "protected-fields-v2"
FABRICATION_BAIT_EVALUATOR_VERSION = "keyword-bait-v2"
NLI_EVIDENCE_LIMIT = 3

# Supportable-requirement coverage (eval/requirement_coverage.py). The provider/model default to
# the judge; the CLI can override them. Semantic decisions use the same structured-output settings.
REQUIREMENT_COVERAGE_EVALUATOR_VERSION = "supportable-requirement-coverage-v1"
REQUIREMENT_COVERAGE_PROMPT_VERSION = "supportable-requirements-v1"
REQUIREMENT_COVERAGE_PROVIDER = JUDGE_PROVIDER
REQUIREMENT_COVERAGE_MODEL = JUDGE_MODEL

# Prompt-injection benchmark (eval/injection.py). Generation uses the workflow's own settings.
INJECTION_ATTACK_VERSION = "prompt-injection-canary-v1"
INJECTION_SAMPLE_VERSION = "injection-sample-v1"
INJECTION_SEED = 42

# Pairwise position-swap comparison (eval/pairwise.py). Uses the judge provider/model.
PAIRWISE_EVALUATOR_VERSION = "pairwise-position-swap-v1"
PAIRWISE_PROMPT_VERSION = "pairwise-v1"
PAIRWISE_SCHEMA_VERSION = "1"
# Hosted reasoning models spend completion tokens on reasoning before the JSON answer;
# 2048 truncated pairwise verdicts (five dimensions plus explanation) in the Groq pilot.
PAIRWISE_MAX_COMPLETION_TOKENS = 4096
# Pairwise verdicts are a forced choice per dimension; low reasoning keeps hosted token use
# within free-tier limits. Position-swap consistency is reported as the stability check.
PAIRWISE_REASONING_EFFORT = "low"
PAIRING_POLICY_VERSION = "same-dataset-pair-repeat-v1"
POSITION_ORDER_VERSION = "challenger-first-then-baseline-first-v1"
PAIRWISE_BASELINE_WORKFLOW = "w2_baseline"
PAIRWISE_FAILURE_JUDGE = "judge_failed"
PAIRWISE_FAILURE_INVALID = "invalid_output"
PAIRWISE_FAILURE_GENERATION = "generation_unavailable"

# Judge-versus-human agreement (eval/human_review.py).
AGREEMENT_SCHEMA_VERSION = "1.1"
HUMAN_AGREEMENT_VERSION = "judge-human-agreement-v1"
MIN_AGREEMENT_PAIRS = 10  # below this, agreement is reported as insufficient_labels
CLAIM_LABELS = ("supported", "contradicted", "unsupported")

# Offline ATS-style compatibility proxy (eval/ats.py). Any scoring-rule change must bump
# OFFLINE_PROXY_VERSION.
ATS_SCHEMA_VERSION = "1.1"
OFFLINE_PROXY_VERSION = "offline-ats-proxy-v1"
ATS_STATUSES = ("scored", "unscored", "failed")
COMPOSITE_WEIGHTS = {
    "parseability": 0.20,
    "section_completeness": 0.25,
    "formatting_compatibility": 0.15,
    "supported_requirement_coverage": 0.40,
}
FORMAT_HAZARD_PENALTY = 0.34

# Input robustness (eval/robustness.py). v2 adds noisy, format and out-of-scope variant families
# and scores against the clean original resume; its artifacts live under
# robustness/<identity_hash>/. v1 artifacts (robustness-sample-v1, meaning-preserving-v1) stay
# under robustness/ as history.
ROBUSTNESS_SEED = 42
ROBUSTNESS_SAMPLE_VERSION = "robustness-sample-v2"
ROBUSTNESS_VARIANT_VERSION = "robustness-variants-v2"
ROBUSTNESS_TYPO_RATE = 0.03          # share of eligible (lowercase, unprotected) prose words
ROBUSTNESS_STABILITY_TOLERANCE = 0.05  # max degradation of a [0, 1] metric for a stable run
REPORT_BOOTSTRAP_SEED = 42
REPORT_BOOTSTRAP_SAMPLES = 2000

DATA_DIR = "data"
RESULTS_FILE = "results/results.jsonl"
