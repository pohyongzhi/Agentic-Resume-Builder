#!/usr/bin/env bash
set -euo pipefail

# Staged local-to-SMU workflow for the V4.1 dataset and Qwen3.5-4B LoRA run.
# Authentication is handled by ssh. Never place a password in this file.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_TARGET="${V4_SSH_TARGET:-kester.yeo.2024@10.193.104.102}"
REMOTE_REPO="Agentic-Resume-Builder"
REMOTE_STORAGE="v4-training"
CONFIG_REL="configs/v4/qwen35-4b-lora-v4-1.json"
DATA_REL="runs/v4-dataset-550-v4-1"
TRAIN_REL="${DATA_REL}/sft-train-490.jsonl"
VALIDATION_REL="${DATA_REL}/sft-validation-60.jsonl"
PROMPT_REL="${DATA_REL}/student-v1-1.txt"
MANIFEST_REL="${DATA_REL}/training-manifest.json"
MODEL_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

SMOKE_NAME="v4-overfit-smoke-v4-1-blackwell-001"
TRAIN_NAME="v4-pilot-training-v4-1-blackwell-001"
MERGE_NAME="v4-merged-v4-1-bf16-001"
SMOKE_SESSION="v4-1-smoke"
TRAIN_SESSION="v4-1-train"
MERGE_SESSION="v4-1-merge"

CONTROL_DIR=""
CONTROL_SOCKET=""

usage() {
  cat <<'EOF'
Usage: scripts/v4_1_remote.sh ACTION

Actions:
  check             Verify that the local V4.1 package exists and has 490/60 rows.
  upload            Upload only training code, config, and the four dataset artifacts.
  smoke             Start the 12-example smoke test in remote tmux + interactive srun.
  train             Start full training after the smoke-result gate passes.
  merge             Merge the trained adapter after training metrics exist.
  status            Show Slurm jobs, tmux sessions, results, and recent logs.
  attach-smoke      Attach to the smoke tmux session.
  attach-train      Attach to the training tmux session.
  attach-merge      Attach to the merge tmux session.
  download-results  Download logs and JSON evidence, excluding weights/checkpoints.
  download-merged   Download the merged BF16 model to the Desktop model archive.

Run actions in this order:
  check -> upload -> smoke -> status -> train -> status -> merge -> status

The script prompts through normal ssh authentication and never stores a password.
EOF
}

local_preflight() {
  local required=(
    "$ROOT/$CONFIG_REL"
    "$ROOT/$TRAIN_REL"
    "$ROOT/$VALIDATION_REL"
    "$ROOT/$PROMPT_REL"
    "$ROOT/$MANIFEST_REL"
    "$ROOT/requirements-v4-training.txt"
    "$ROOT/v4_training/__main__.py"
  )
  local path
  for path in "${required[@]}"; do
    [[ -f "$path" ]] || {
      echo "Missing required V4.1 artifact: $path" >&2
      echo "The targeted V4.1 dataset must be completed before cluster work starts." >&2
      return 1
    }
  done

  local train_count validation_count
  train_count="$(wc -l < "$ROOT/$TRAIN_REL" | tr -d ' ')"
  validation_count="$(wc -l < "$ROOT/$VALIDATION_REL" | tr -d ' ')"
  [[ "$train_count" == "490" ]] || {
    echo "Expected 490 training records, found $train_count" >&2
    return 1
  }
  [[ "$validation_count" == "60" ]] || {
    echo "Expected 60 validation records, found $validation_count" >&2
    return 1
  }

  python3 - "$ROOT/$CONFIG_REL" "$TRAIN_REL" "$VALIDATION_REL" \
    "$MANIFEST_REL" "$PROMPT_REL" "$MODEL_REVISION" <<'PY'
import json
import sys

path, train, validation, manifest, prompt, revision = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    config = json.load(handle)
expected = {
    "train": train,
    "validation": validation,
    "manifest": manifest,
    "system_prompt": prompt,
}
if config.get("data") != expected:
    raise SystemExit(f"Config data paths do not match: {config.get('data')!r}")
if config.get("model", {}).get("revision") != revision:
    raise SystemExit("Pinned model revision is missing or incorrect")
print("Local preflight passed: 490 train, 60 validation, pinned model revision present.")
PY
}

open_master() {
  [[ -n "$CONTROL_SOCKET" && -S "$CONTROL_SOCKET" ]] && return
  CONTROL_DIR="$(mktemp -d /tmp/v4-ssh.XXXXXX)"
  CONTROL_SOCKET="$CONTROL_DIR/s"
  echo "Opening SSH connection to $SSH_TARGET"
  ssh -M -S "$CONTROL_SOCKET" -o ControlPersist=600 -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=6 -fnNT "$SSH_TARGET"
}

close_master() {
  if [[ -n "$CONTROL_SOCKET" && -S "$CONTROL_SOCKET" ]]; then
    ssh -S "$CONTROL_SOCKET" -O exit "$SSH_TARGET" >/dev/null 2>&1 || true
  fi
  if [[ -n "$CONTROL_DIR" && -d "$CONTROL_DIR" ]]; then
    rmdir "$CONTROL_DIR" 2>/dev/null || true
  fi
}

remote() {
  open_master
  ssh -S "$CONTROL_SOCKET" -o ServerAliveInterval=30 "$SSH_TARGET" "$@"
}

upload() {
  local_preflight
  open_master
  remote "mkdir -p \"\$HOME/$REMOTE_REPO/v4_training\" \"\$HOME/$REMOTE_REPO/configs/v4\" \"\$HOME/$REMOTE_REPO/$DATA_REL\""
  local transport="ssh -S $CONTROL_SOCKET -o ServerAliveInterval=30"
  rsync -av --exclude '__pycache__' --exclude '*.pyc' -e "$transport" \
    "$ROOT/v4_training/" "$SSH_TARGET:~/$REMOTE_REPO/v4_training/"
  rsync -av -e "$transport" \
    "$ROOT/requirements-v4-training.txt" \
    "$SSH_TARGET:~/$REMOTE_REPO/requirements-v4-training.txt"
  rsync -av -e "$transport" \
    "$ROOT/$CONFIG_REL" "$SSH_TARGET:~/$REMOTE_REPO/configs/v4/"
  rsync -av -e "$transport" \
    "$ROOT/$TRAIN_REL" "$ROOT/$VALIDATION_REL" "$ROOT/$PROMPT_REL" \
    "$ROOT/$MANIFEST_REL" "$SSH_TARGET:~/$REMOTE_REPO/$DATA_REL/"
  remote "wc -l \"\$HOME/$REMOTE_REPO/$TRAIN_REL\" \"\$HOME/$REMOTE_REPO/$VALIDATION_REL\""
  echo "Upload complete. No raw sources, caches, credentials, or model files were transferred."
}

start_stage() {
  local stage="$1"
  open_master
  remote bash -s -- "$stage" "$CONFIG_REL" "$MODEL_REVISION" \
    "$SMOKE_NAME" "$TRAIN_NAME" "$MERGE_NAME" \
    "$SMOKE_SESSION" "$TRAIN_SESSION" "$MERGE_SESSION" <<'REMOTE'
set -euo pipefail

stage="$1"
config_rel="$2"
revision="$3"
smoke_name="$4"
train_name="$5"
merge_name="$6"
smoke_session="$7"
train_session="$8"
merge_session="$9"

repo="$HOME/Agentic-Resume-Builder"
storage="$HOME/v4-training"
python_bin="$storage/venvs/qwen35/bin/python"

[[ -x "$python_bin" ]] || { echo "Missing training Python: $python_bin" >&2; exit 1; }
[[ -f "$repo/$config_rel" ]] || { echo "Missing remote config: $repo/$config_rel" >&2; exit 1; }

snapshot="$(find "$HOME/.cache" "$storage" -type d -path "*/models--Qwen--Qwen3.5-4B/snapshots/$revision" 2>/dev/null | head -1)"
[[ -n "$snapshot" ]] || { echo "Pinned Qwen3.5 snapshot is not present on the server" >&2; exit 1; }
cache="$(dirname "$(dirname "$(dirname "$snapshot")")")"

mkdir -p "$storage/runs" "$storage/logs"

case "$stage" in
  smoke)
    session="$smoke_session"
    output="$storage/runs/$smoke_name"
    log="$storage/logs/$smoke_name.log"
    [[ ! -e "$output" ]] || { echo "Refusing to overwrite $output" >&2; exit 1; }
    payload="module load Python/3.11.11-GCCcore-13.3.0; cd '$repo'; export HF_HUB_CACHE='$cache' HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false; '$python_bin' -m v4_training audit-tokenizer --config '$config_rel' --cache-dir '$cache' --local-files-only --examples 3; '$python_bin' -m v4_training audit-labels --config '$config_rel' --cache-dir '$cache' --local-files-only --examples 3; '$python_bin' -m v4_training overfit-smoke --config '$config_rel' --cache-dir '$cache' --output-dir '$output' --examples 12 --max-steps 200 2>&1 | tee '$log'"
    time_limit="01:00:00"
    ;;
  train)
    session="$train_session"
    output="$storage/runs/$train_name"
    log="$storage/logs/$train_name.log"
    smoke_result="$storage/runs/$smoke_name/smoke-result.json"
    [[ -f "$smoke_result" ]] || { echo "Smoke result is missing: $smoke_result" >&2; exit 1; }
    "$python_bin" -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["memorization_gate_passed"] and x["reload_token_ids_identical"], x' "$smoke_result"
    [[ ! -e "$output" ]] || { echo "Refusing to overwrite $output" >&2; exit 1; }
    payload="module load Python/3.11.11-GCCcore-13.3.0; cd '$repo'; export HF_HUB_CACHE='$cache' HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false; '$python_bin' -m v4_training train --config '$config_rel' --cache-dir '$cache' --output-dir '$output' 2>&1 | tee '$log'"
    time_limit="02:00:00"
    ;;
  merge)
    session="$merge_session"
    output="$storage/runs/$merge_name"
    log="$storage/logs/$merge_name.log"
    adapter="$storage/runs/$train_name/adapter-final"
    metrics="$storage/runs/$train_name/validation-metrics.json"
    [[ -d "$adapter" && -f "$metrics" ]] || { echo "Completed training adapter or metrics are missing" >&2; exit 1; }
    [[ ! -e "$output" ]] || { echo "Refusing to overwrite $output" >&2; exit 1; }
    payload="module load Python/3.11.11-GCCcore-13.3.0; cd '$repo'; export HF_HUB_CACHE='$cache' HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false; '$python_bin' -m v4_training reload-test --config '$config_rel' --cache-dir '$cache' --adapter '$adapter' --output-dir '$storage/runs/v4-reload-v4-1-blackwell-001'; '$python_bin' -m v4_training merge-adapter --config '$config_rel' --cache-dir '$cache' --adapter '$adapter' --output-dir '$output' --verification-tokens 128 2>&1 | tee '$log'"
    time_limit="01:00:00"
    ;;
  *) echo "Unknown remote stage: $stage" >&2; exit 2 ;;
esac

if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 1
fi

printf -v launch '%q ' srun --account=is469 --partition=student --qos=studentqos \
  --gres=gpu:rtx_pro_6000_blackwell_max:1 --cpus-per-task=4 --mem=32G \
  --time="$time_limit" --pty bash -lc "$payload"
tmux new-session -d -s "$session" "$launch"
echo "Started tmux session: $session"
echo "Output: $output"
echo "Log: $log"
REMOTE
}

status() {
  remote bash -s -- "$SMOKE_NAME" "$TRAIN_NAME" "$MERGE_NAME" <<'REMOTE'
set -u
storage="$HOME/v4-training"
echo "=== SLURM ==="
squeue -u "$USER" -o "%.18i %.10T %.20N %.30R" || true
echo "=== TMUX ==="
tmux list-sessions 2>/dev/null || echo "No tmux sessions"
echo "=== RESULTS ==="
for name in "$@"; do
  if [[ -d "$storage/runs/$name" ]]; then
    echo "$name"
    find "$storage/runs/$name" -maxdepth 2 -type f \( -name '*result.json' -o -name '*metrics.json' \) -print
  fi
done
echo "=== RECENT LOGS ==="
for name in "$@"; do
  log="$storage/logs/$name.log"
  if [[ -f "$log" ]]; then
    echo "--- $log"
    tail -n 20 "$log"
  fi
done
REMOTE
}

attach_session() {
  local session="$1"
  open_master
  ssh -tt -S "$CONTROL_SOCKET" "$SSH_TARGET" "tmux attach -t '$session'"
}

download_results() {
  open_master
  local destination="$ROOT/runs/v4-1-server-results"
  mkdir -p "$destination"
  local transport="ssh -S $CONTROL_SOCKET -o ServerAliveInterval=30"
  local name
  for name in "$SMOKE_NAME" "$TRAIN_NAME" "v4-reload-v4-1-blackwell-001"; do
    rsync -av --partial --exclude 'checkpoint-*' --exclude '*.safetensors' \
      -e "$transport" "$SSH_TARGET:~/$REMOTE_STORAGE/runs/$name/" \
      "$destination/$name/" || true
  done
  rsync -av --partial -e "$transport" \
    "$SSH_TARGET:~/$REMOTE_STORAGE/logs/${SMOKE_NAME}.log" \
    "$SSH_TARGET:~/$REMOTE_STORAGE/logs/${TRAIN_NAME}.log" \
    "$destination/" || true
  echo "Downloaded evidence to $destination"
}

download_merged() {
  open_master
  local destination="$HOME/Desktop/Agentic-Resume-Builder-Models/$MERGE_NAME"
  mkdir -p "$destination"
  local transport="ssh -S $CONTROL_SOCKET -o ServerAliveInterval=30"
  rsync -avh --partial --progress -e "$transport" \
    "$SSH_TARGET:~/$REMOTE_STORAGE/runs/$MERGE_NAME/" "$destination/"
  echo "Downloaded merged model to $destination"
}

trap close_master EXIT

action="${1:-}"
case "$action" in
  check) local_preflight ;;
  upload) upload ;;
  smoke) start_stage smoke ;;
  train) start_stage train ;;
  merge) start_stage merge ;;
  status) status ;;
  attach-smoke) attach_session "$SMOKE_SESSION" ;;
  attach-train) attach_session "$TRAIN_SESSION" ;;
  attach-merge) attach_session "$MERGE_SESSION" ;;
  download-results) download_results ;;
  download-merged) download_merged ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
