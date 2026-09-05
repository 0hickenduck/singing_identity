#!/usr/bin/env bash
set -euo pipefail

# Resume the repaired 200-row-per-singer Stage 1 run on local NVMe.
# Run from the repository root inside tmux on the same valkyrie node:
#
#   tmux new -s singing_stage1_resume
#   bash scripts/resume_stage1_repaired_200_on_localdisk.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  elif [[ -x /tmp/singing_identity_cuda_venv/bin/python ]]; then
    PYTHON_BIN="/tmp/singing_identity_cuda_venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

WORK_ROOT="${WORK_ROOT:-/localdisk/bowen/singing_identity}"
CACHE_ROOT="${CACHE_ROOT:-/localdisk/bowen/.cache/singing_identity}"
RUN_NAME="${RUN_NAME:-stage1_repaired_200_fresh_local}"
RUN_ROOT="${RUN_ROOT:-${WORK_ROOT}/runs/${RUN_NAME}}"
GTSINGER_ROOT="${GTSINGER_ROOT:-${WORK_ROOT}/data/gtsinger_domain_eval}"
FEATURE_ROOT="${FEATURE_ROOT:-${WORK_ROOT}/features/${RUN_NAME}}"
REPORT_OUT="${REPORT_OUT:-${REPO_ROOT}/results/${RUN_NAME}_report.md}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}_resume.log}"
ALLOW_HOME_HEAVY_IO_FLAG=()
if [[ "${FEATURE_ROOT}" == /home/bowen/* ]]; then
  ALLOW_HOME_HEAVY_IO_FLAG=(--allow-home-heavy-io)
fi

mkdir -p "${RUN_ROOT}" "${FEATURE_ROOT}" "${CACHE_ROOT}" "${LOG_DIR}" /localdisk/bowen/tmp

export XDG_CACHE_HOME="${CACHE_ROOT}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_DATASETS_CACHE="${CACHE_ROOT}/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/huggingface/transformers"
export TORCH_HOME="${CACHE_ROOT}/torch"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export UV_CACHE_DIR="${CACHE_ROOT}/uv"
export UV_LINK_MODE=copy
export NPM_CONFIG_CACHE="${CACHE_ROOT}/npm"
export CUDA_CACHE_PATH="${CACHE_ROOT}/nv/ComputeCache"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export TMPDIR="${TMPDIR:-/localdisk/bowen/tmp}"
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1

{
  echo "== Stage 1 localdisk resume =="
  date
  hostname
  echo "repo=${REPO_ROOT}"
  echo "python=${PYTHON_BIN}"
  echo "run_name=${RUN_NAME}"
  echo "run_root=${RUN_ROOT}"
  echo "feature_root=${FEATURE_ROOT}"
  echo "gtsinger_root=${GTSINGER_ROOT}"
  echo "cache_root=${CACHE_ROOT}"
  if ((${#ALLOW_HOME_HEAVY_IO_FLAG[@]})); then
    echo "allow_home_heavy_io=1 (explicitly reusing an existing NFS feature cache)"
  fi
  echo
  echo "== Lab preflight =="
} | tee -a "${LOG_FILE}"

STRICT="${STRICT:-1}" PREPARE_DIRS=0 bash "${REPO_ROOT}/scripts/check_lab_environment.sh" 2>&1 | tee -a "${LOG_FILE}"

{
  echo
  echo "== Verify local Stage 1 state =="
} | tee -a "${LOG_FILE}"

test -d "${RUN_ROOT}" || { echo "Missing RUN_ROOT=${RUN_ROOT}" | tee -a "${LOG_FILE}"; exit 2; }
test -d "${FEATURE_ROOT}" || { echo "Missing FEATURE_ROOT=${FEATURE_ROOT}" | tee -a "${LOG_FILE}"; exit 2; }

{
  echo
  echo "== Resume Stage 1 without --force =="
  date
} | tee -a "${LOG_FILE}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" scripts/run_stage1_overnight.py \
  --models acoustic,wavlm,hubert,mert \
  --python "${PYTHON_BIN}" \
  --device cuda \
  --gtsinger-root "${GTSINGER_ROOT}" \
  --run-root "${RUN_ROOT}" \
  --report "${REPORT_OUT}" \
  --feature-root "${FEATURE_ROOT}" \
  --max-rows-per-singer 200 \
  --smoke-job-timeout-sec 1800 \
  --extract-job-timeout-sec 43200 \
  --probe-job-timeout-sec 14400 \
  --local-cache-root "${CACHE_ROOT}" \
  --fast-no-wav-stats \
  "${ALLOW_HOME_HEAVY_IO_FLAG[@]}" 2>&1 | tee -a "${LOG_FILE}"
