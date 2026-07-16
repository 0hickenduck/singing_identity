#!/usr/bin/env bash
set -euo pipefail

# Clean Stage 1 rerun where dataset, caches, features, and run outputs are born
# on local NVMe. Run inside tmux on a valkyrie node.

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

RUN_NAME="${RUN_NAME:-stage1_repaired_200_fresh_local}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-1}"
WORK_ROOT="${WORK_ROOT:-/localdisk/bowen/singing_identity}"
CACHE_ROOT="${CACHE_ROOT:-/localdisk/bowen/.cache/singing_identity}"
DATA_ROOT="${DATA_ROOT:-${WORK_ROOT}/data/gtsinger_domain_eval}"
RUN_ROOT="${RUN_ROOT:-${WORK_ROOT}/runs/${RUN_NAME}}"
FEATURE_ROOT="${FEATURE_ROOT:-${WORK_ROOT}/features/${RUN_NAME}}"
STATUS_DIR="${STATUS_DIR:-${WORK_ROOT}/status/${RUN_NAME}}"
REPORT_OUT="${REPORT_OUT:-${REPO_ROOT}/results/${RUN_NAME}_report.md}"
LOG_DIR="${LOG_DIR:-${WORK_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"

mkdir -p "${DATA_ROOT}" "${RUN_ROOT}" "${FEATURE_ROOT}" "${STATUS_DIR}" "${CACHE_ROOT}" "${LOG_DIR}" /localdisk/bowen/tmp

for heavy_path in "${DATA_ROOT}" "${RUN_ROOT}" "${FEATURE_ROOT}" "${CACHE_ROOT}" "${LOG_DIR}"; do
  fs_type="$(findmnt -T "${heavy_path}" -n -o FSTYPE 2>/dev/null || true)"
  if [[ "${fs_type}" == nfs* && "${ALLOW_NFS_HEAVY_IO:-0}" != "1" ]]; then
    echo "Refusing heavy I/O on NFS path: ${heavy_path} (${fs_type})."
    echo "Use /localdisk/bowen/... or set ALLOW_NFS_HEAVY_IO=1 only after explicit approval."
    exit 2
  fi
done

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
  echo "== Fresh Stage 1 on localdisk =="
  date
  hostname
  echo "repo=${REPO_ROOT}"
  echo "python=${PYTHON_BIN}"
  echo "data_root=${DATA_ROOT}"
  echo "run_root=${RUN_ROOT}"
  echo "feature_root=${FEATURE_ROOT}"
  echo "cache_root=${CACHE_ROOT}"
  echo "download_workers=${DOWNLOAD_WORKERS}"
  echo "status_dir=${STATUS_DIR}"
  echo "report=${REPORT_OUT}"
  echo
  echo "== Lab preflight =="
} | tee -a "${LOG_FILE}"

STRICT="${STRICT:-1}" PREPARE_DIRS=0 bash "${REPO_ROOT}/scripts/check_lab_environment.sh" 2>&1 | tee -a "${LOG_FILE}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" scripts/data_prep/repair_gtsinger_and_rerun_stage1.py \
  --repo-root "${REPO_ROOT}" \
  --python "${PYTHON_BIN}" \
  --local-dir "${DATA_ROOT}" \
  --status-dir "${STATUS_DIR}" \
  --models acoustic,wavlm,hubert,mert \
  --device cuda \
  --stage1-run-root "${RUN_ROOT}" \
  --stage1-feature-root "${FEATURE_ROOT}" \
  --stage1-report "${REPORT_OUT}" \
  --max-rows-per-singer 200 \
  --smoke-job-timeout-sec 1800 \
  --extract-job-timeout-sec 43200 \
  --probe-job-timeout-sec 14400 \
  --local-cache-root "${CACHE_ROOT}" \
  --max-workers "${DOWNLOAD_WORKERS}" \
  --max-download-attempts 96 \
  --retry-wait-sec 1800 2>&1 | tee -a "${LOG_FILE}"
