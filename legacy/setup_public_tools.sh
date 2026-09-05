#!/usr/bin/env bash
set -euo pipefail

# Install public, user-local tools used by this project. This does not install
# lab-private helpers such as all_gpus, all_cpus, cpz, or rmz.

BIN_DIR="${BIN_DIR:-${HOME}/.local/bin}"
CACHE_ROOT="${CACHE_ROOT:-/localdisk/bowen/.cache}"
INSTALL_MARIMO="${INSTALL_MARIMO:-0}"

mkdir -p "${BIN_DIR}" "${CACHE_ROOT}/uv" "${CACHE_ROOT}/pip" "${CACHE_ROOT}/huggingface" /localdisk/bowen/tmp

export PATH="${BIN_DIR}:${PATH}"
export XDG_CACHE_HOME="${CACHE_ROOT}"
export UV_CACHE_DIR="${CACHE_ROOT}/uv"
export UV_LINK_MODE=copy
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export TMPDIR="/localdisk/bowen/tmp"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to install uv." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv into ${BIN_DIR}..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${BIN_DIR}:${PATH}"
else
  echo "uv already available at $(command -v uv)"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv installation finished, but uv is still not on PATH." >&2
  echo "Add this to shell startup: export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
  exit 1
fi

if [[ "${INSTALL_MARIMO}" == "1" ]]; then
  echo "Installing/updating marimo through uv tool..."
  uv tool install marimo
else
  echo "Skipping permanent marimo tool install. Use INSTALL_MARIMO=1 to install it."
fi

cat <<'EOF'

Public tool setup complete.

Still install or locate these lab-private helpers through the lab-approved path:
  all_gpus all_cpus cpz rmz

Recommended shell startup line:
  export PATH="$HOME/.local/bin:$PATH"
EOF
