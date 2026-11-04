#!/usr/bin/env bash
# Run one workflow's share of the team evaluation with the owner's own Groq key.
#
#   scripts/team_eval.sh <workflow> <step>
#
# Steps, run in this order (each is resumable; rerun the same command to continue):
#   free      local metrics on all 100 outputs + requirement coverage on gpt-oss-20b
#   rubric    rubric judge (gpt-oss-120b) on the 30 pairs in eval_plan/rubric30.txt
#   pairwise  position-swapped comparison against w2_baseline on eval_plan/pairwise10.txt
#   status    show how far each step has got
#
# Needs GROQ_API_KEY in .env (never commit it). Config hashes are read from
# results/results.jsonl (the newest generation of each workflow), so everyone
# evaluates the same shared outputs with the same settings.
set -euo pipefail
cd "$(dirname "$0")/.."

WF="${1:-}"; STEP="${2:-}"
WORKFLOWS="w1_single_pass w2_baseline w3_critic w4_evidence w5_fine_tuned w6_best_of_n w7_plan_execute"
if [[ -z "$WF" || -z "$STEP" || " $WORKFLOWS " != *" $WF "* ]]; then
  echo "usage: scripts/team_eval.sh <workflow> <free|rubric|pairwise|status>"; echo "workflows: $WORKFLOWS"; exit 2
fi

PY="${PYTHON:-python}"
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
if [[ "$STEP" != status && -z "${GROQ_API_KEY:-}" ]]; then echo "GROQ_API_KEY is not set (put it in .env)"; exit 2; fi
# Free-tier keys allow 8k tokens a minute; one request reserves prompt + 4,096 completion tokens.
export HOSTED_MIN_INTERVAL_SECONDS="${HOSTED_MIN_INTERVAL_SECONDS:-55}"
export PYTHONDONTWRITEBYTECODE=1
# TEST_PAIRS=<file> limits every step to the pairs in that file (for a dry run before the real one).
RUBRIC_PAIRS="${TEST_PAIRS:-eval_plan/rubric30.txt}"; PAIRWISE_PAIRS="${TEST_PAIRS:-eval_plan/pairwise10.txt}"
FREE_LIMIT=(); if [[ -n "${TEST_PAIRS:-}" ]]; then FREE_LIMIT=(--pair-ids-file "$TEST_PAIRS"); fi

hash_of() {
  "$PY" - "$1" <<'EOF'
import json, sys
wf, latest = sys.argv[1], None
for line in open("results/results.jsonl"):
    row = json.loads(line)
    if row.get("workflow") == wf:
        latest = row.get("config_hash")
print(latest or "")
EOF
}
H="$(hash_of "$WF")"; H2="$(hash_of w2_baseline)"
if [[ -z "$H" ]]; then echo "No generated outputs for $WF in results/results.jsonl. Pull the shared outputs first."; exit 2; fi

COMMON=(--judge-provider groq --judge-model openai/gpt-oss-120b
        --coverage-provider groq --coverage-model openai/gpt-oss-20b
        --skip-robustness --skip-injection --skip-human-review --skip-reports)

case "$STEP" in
  free)
    "$PY" evaluate.py score --workflows "$WF" --config-hash "$WF=$H" --skip-judge --skip-pairwise "${FREE_LIMIT[@]}" "${COMMON[@]}"
    ;;
  rubric)
    "$PY" evaluate.py score --workflows "$WF" --config-hash "$WF=$H" --skip-coverage --skip-pairwise \
      --pair-ids-file "$RUBRIC_PAIRS" "${COMMON[@]}"
    ;;
  pairwise)
    if [[ "$WF" == w2_baseline ]]; then echo "w2_baseline is the baseline; it has no pairwise step."; exit 0; fi
    "$PY" evaluate.py pairwise --workflows w2_baseline "$WF" --config-hash "w2_baseline=$H2" --config-hash "$WF=$H" \
      --pairwise-challengers "$WF" --skip-judge --skip-coverage --pair-ids-file "$PAIRWISE_PAIRS" "${COMMON[@]}"
    ;;
  status)
    "$PY" - "$WF" <<'EOF'
import json, pathlib, sys
wf = sys.argv[1]
for root in sorted(pathlib.Path("results/evaluations").glob("*/*/*")):
    recs = list((root / "records" / wf).glob("*/repeat_0.json"))
    pw = list(root.glob(f"pairwise/*/records/{wf}/*/repeat_0.json"))
    if not recs and not pw:
        continue
    judged = sum(1 for f in recs if json.loads(f.read_text()).get("judge_status") == "scored")
    cov = sum(1 for f in recs if json.loads(f.read_text()).get("requirement_coverage_status") == "scored")
    pws = sum(1 for f in pw if json.loads(f.read_text()).get("status") == "scored")
    print(f"{root}: {len(recs)} scored outputs, {cov} with coverage, {judged} judged; pairwise {pws}/{len(pw)} scored")
EOF
    ;;
  *) echo "unknown step: $STEP (free, rubric, pairwise, status)"; exit 2 ;;
esac
