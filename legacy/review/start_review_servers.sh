#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE_DIR="${REVIEW_STATE_DIR:-/localdisk/bowen/singing_identity/review_servers}"
STATE_FILE="${REVIEW_STATE_FILE:-$STATE_DIR/review_servers.tsv}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/localdisk/bowen/.cache/uv}"
TIMEOUT_SEC="${REVIEW_START_TIMEOUT_SEC:-120}"

usage() {
  cat <<'EOF'
Usage:
  scripts/review/start_review_servers.sh PORT:NOTEBOOK [PORT:NOTEBOOK ...]

Example:
  scripts/review/start_review_servers.sh \
    2719:notebooks/human_check.py \
    2720:notebooks/prompt_review.py

Starts each notebook as:
  uv run --with marimo marimo edit NOTEBOOK --host 127.0.0.1 --port PORT --headless --no-token

State is written to:
  /localdisk/bowen/singing_identity/review_servers/review_servers.tsv
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

session_name() {
  local port="$1"
  printf 'review_marimo_%s' "$port"
}

port_pids() {
  local port="$1"
  ss -H -ltnp 2>/dev/null \
    | awk -v port=":${port}" '$4 ~ port "$" {print $0}' \
    | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
    | sort -u
}

port_listening() {
  local port="$1"
  ss -H -ltn 2>/dev/null | awk -v port=":${port}" '$4 ~ port "$" {found=1} END {exit found ? 0 : 1}'
}

kill_port_processes() {
  local port="$1"
  mapfile -t pids < <(port_pids "$port")
  if ((${#pids[@]} == 0)); then
    return 0
  fi
  echo "Stopping existing listener(s) on port $port: ${pids[*]}"
  kill -TERM "${pids[@]}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! port_listening "$port"; then
      return 0
    fi
    sleep 0.5
  done
  mapfile -t pids < <(port_pids "$port")
  if ((${#pids[@]} > 0)); then
    echo "Force-stopping listener(s) on port $port: ${pids[*]}"
    kill -KILL "${pids[@]}" 2>/dev/null || true
  fi
}

parse_pair() {
  local pair="$1"
  [[ "$pair" == *:* ]] || die "expected PORT:NOTEBOOK, got: $pair"
  local port="${pair%%:*}"
  local notebook="${pair#*:}"
  [[ "$port" =~ ^[0-9]+$ ]] || die "invalid port in pair: $pair"
  [[ -n "$notebook" ]] || die "empty notebook path in pair: $pair"
  if [[ "$notebook" != /* ]]; then
    notebook="$REPO_ROOT/$notebook"
  fi
  [[ -f "$notebook" ]] || die "notebook does not exist: $notebook"
  printf '%s\t%s\n' "$port" "$notebook"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -eq 0 ]]; then
  usage
  exit 0
fi

need_cmd uv
need_cmd tmux
need_cmd ss

mkdir -p "$STATE_DIR" "$UV_CACHE_DIR"

declare -a ports=()
declare -a notebooks=()
while (($#)); do
  parsed="$(parse_pair "$1")"
  ports+=("$(cut -f1 <<<"$parsed")")
  notebooks+=("$(cut -f2- <<<"$parsed")")
  shift
done

declare -A seen_ports=()
for port in "${ports[@]}"; do
  [[ -z "${seen_ports[$port]:-}" ]] || die "duplicate port: $port"
  seen_ports[$port]=1
done

echo "Starting review servers from repo: $REPO_ROOT"
echo "Using UV_CACHE_DIR=$UV_CACHE_DIR"

: > "$STATE_FILE.tmp"
for i in "${!ports[@]}"; do
  port="${ports[$i]}"
  notebook="${notebooks[$i]}"
  session="$(session_name "$port")"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Killing existing tmux session: $session"
    tmux kill-session -t "$session" || true
  fi
  kill_port_processes "$port"

  echo "Starting $session on port $port -> $notebook"
  tmux new-session -d -s "$session" \
    "cd '$REPO_ROOT' && export UV_CACHE_DIR='$UV_CACHE_DIR' && export UV_LINK_MODE=copy && uv run --with marimo marimo edit '$notebook' --host 127.0.0.1 --port '$port' --headless --no-token"
  printf '%s\t%s\t%s\n' "$port" "$session" "$notebook" >> "$STATE_FILE.tmp"
done
mv "$STATE_FILE.tmp" "$STATE_FILE"

deadline=$((SECONDS + TIMEOUT_SEC))
declare -a ready=()
declare -a pending=("${ports[@]}")
while ((${#pending[@]} > 0)); do
  next_pending=()
  for port in "${pending[@]}"; do
    if port_listening "$port"; then
      ready+=("$port")
    else
      next_pending+=("$port")
    fi
  done
  pending=("${next_pending[@]}")
  ((${#pending[@]} == 0)) && break
  if ((SECONDS >= deadline)); then
    echo
    echo "FAILED: timed out waiting for review servers after ${TIMEOUT_SEC}s"
    echo "Ready ports: ${ready[*]:-(none)}"
    echo "Pending ports: ${pending[*]}"
    echo
    for port in "${pending[@]}"; do
      session="$(session_name "$port")"
      echo "--- tmux tail: $session ---"
      tmux capture-pane -pt "$session" -S -80 2>/dev/null | tail -n 40 || true
    done
    exit 1
  fi
  sleep 1
done

echo
echo "Review servers ready:"
while IFS=$'\t' read -r port session notebook; do
  [[ -n "$port" ]] || continue
  echo "  port $port -> $notebook"
  echo "    session: $session"
  echo "    url: http://127.0.0.1:$port"
done < "$STATE_FILE"
echo
echo "Local tunnel example:"
printf '  powershell -File C:\\Users\\libowen\\bin\\open_lab_tunnel.ps1 -Ports '
(IFS=,; echo "${ports[*]}")
