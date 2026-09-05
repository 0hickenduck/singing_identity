#!/usr/bin/env bash
set -euo pipefail

# Keep the repo current before any long run (sequential-editing workflow).
git -C "$(dirname "$0")/.." pull --ff-only || echo "WARNING: git pull --ff-only failed; repo may be stale or diverged."

# Lightweight preflight for the Gavo lab cluster. This script is intentionally
# read-mostly: it reports node, mount, cache, and helper-tool state before long
# jobs start.

STRICT="${STRICT:-1}"
PREPARE_DIRS="${PREPARE_DIRS:-0}"

failures=0

note() {
  printf '%s\n' "$*"
}

warn() {
  printf 'WARN: %s\n' "$*" >&2
}

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  failures=$((failures + 1))
}

check_command() {
  local name="$1"
  local required="$2"
  if command -v "${name}" >/dev/null 2>&1; then
    printf '%-12s %s\n' "${name}" "$(command -v "${name}")"
  else
    printf '%-12s MISSING\n' "${name}"
    if [[ "${required}" == "required" ]]; then
      fail "required command '${name}' is missing from PATH"
    fi
  fi
}

check_mount() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    findmnt -T "${path}" -o TARGET,SOURCE,FSTYPE,SIZE,AVAIL,USE% || fail "findmnt failed for ${path}"
  else
    warn "${path} does not exist"
  fi
}

check_cache_var() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "${value}" ]]; then
    warn "${name} is unset"
    return
  fi
  printf '%-24s %s\n' "${name}" "${value}"
  case "${value}" in
    /home/bowen/*|/work/bowen/*)
      fail "${name} points to NFS-like storage: ${value}"
      ;;
  esac
}

host="$(hostname)"

note "== Host =="
note "${host}"
case "${host}" in
  valkyrie*|gavo*) ;;
  athena*) fail "running on athena; use athena only as a jump host" ;;
  *) warn "hostname is not a recognized valkyrie/gavo compute node" ;;
esac

if [[ "${PREPARE_DIRS}" == "1" ]]; then
  mkdir -p /localdisk/bowen/singing_identity/{data,runs,features,logs,status,checkpoints}
  mkdir -p /localdisk/bowen/.cache/{huggingface,torch,uv,pip,nv,torch_extensions}
  mkdir -p /localdisk/bowen/tmp
fi

note
note "== Mounts =="
check_mount /home/bowen
check_mount /work/bowen
check_mount /localdisk
check_mount /localdisk/bowen

note
note "== Tools =="
check_command all_gpus required
check_command all_cpus required
check_command uv required
check_command tmux required
check_command rg optional
check_command cpz optional
check_command rmz optional
check_command dust optional
check_command marimo optional

note
note "== Resource Snapshot =="
if command -v all_gpus >/dev/null 2>&1; then
  all_gpus || fail "all_gpus returned a non-zero exit code"
else
  warn "skipping GPU snapshot because all_gpus is missing"
fi
if command -v all_cpus >/dev/null 2>&1; then
  all_cpus || fail "all_cpus returned a non-zero exit code"
else
  warn "skipping CPU snapshot because all_cpus is missing"
fi

note
note "== Cache Environment =="
check_cache_var XDG_CACHE_HOME
check_cache_var HF_HOME
check_cache_var HF_DATASETS_CACHE
check_cache_var HUGGINGFACE_HUB_CACHE
check_cache_var TRANSFORMERS_CACHE
check_cache_var TORCH_HOME
check_cache_var PIP_CACHE_DIR
check_cache_var UV_CACHE_DIR
printf '%-24s %s\n' "UV_LINK_MODE" "${UV_LINK_MODE:-unset}"
check_cache_var CUDA_CACHE_PATH
check_cache_var TORCH_EXTENSIONS_DIR
check_cache_var TMPDIR

note
note "== PATH =="
note "${PATH}"
case ":${PATH}:" in
  *":${HOME}/.local/bin:"*) ;;
  *) warn "${HOME}/.local/bin is not on PATH" ;;
esac

if ((failures > 0)); then
  warn "${failures} preflight issue(s) found"
  if [[ "${STRICT}" == "1" ]]; then
    exit 1
  fi
fi
