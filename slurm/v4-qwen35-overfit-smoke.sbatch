#!/usr/bin/env bash
# One-GPU template. Supply the site-specific values on the sbatch command line.
# Required placeholders: SMU_ACCOUNT, SMU_PARTITION, SMU_QOS, SMU_TIME_LIMIT,
# SMU_GPU_GRES, SMU_STORAGE_ROOT, SMU_PROJECT_ROOT, and SMU_PYTHON.
#SBATCH --job-name=v4-qwen35-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail

: "${V4_STORAGE_ROOT:?Set V4_STORAGE_ROOT to private cluster storage}"
: "${V4_PROJECT_ROOT:?Set V4_PROJECT_ROOT to the repository checkout}"
: "${V4_PYTHON:?Set V4_PYTHON to the pinned environment python executable}"
: "${SLURM_JOB_ID:?This template must run as a Slurm job}"

export HF_HOME="${V4_STORAGE_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${V4_STORAGE_ROOT}/logs" "${V4_STORAGE_ROOT}/runs"
cd "${V4_PROJECT_ROOT}"

"${V4_PYTHON}" -m v4_training overfit-smoke \
  --config configs/v4/qwen35-4b-lora.json \
  --cache-dir "${HF_HUB_CACHE}" \
  --output-dir "${V4_STORAGE_ROOT}/runs/v4-overfit-smoke-${SLURM_JOB_ID}" \
  --examples 12
