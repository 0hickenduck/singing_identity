#!/usr/bin/env bash
set -euo pipefail

STATE_DIR="${REVIEW_STATE_DIR:-/localdisk/bowen/singing_identity/review_servers}"
STATE_FILE="${REVIEW_STATE_FILE:-$STATE_DIR/review_servers.tsv}"

usage() {
  cat <<'EOF'
Usage:
  scripts/review/stop_review_servers.sh [PORT ...]

Without arguments, stops the review servers recorded in:
  /localdisk/bowen/singing_identity/review_servers/review_servers.tsv

With ports, stops sessions named review_marimo_PORT and any remaining listener
on those ports.
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

stop_session_and_port() {
  local port="$1"
  local session="${2:-$(session_name "$port")}"

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Killing tmux session: $session"
    tmux kill-session -t "$session" || true
  else
    echo "No tmux session found: $session"
  fi

  for _ in $(seq 1 20); do
    if ! port_listening "$port"; then
      echo "Port $port released"
      return 0
    fi
    sleep 0.5
  done

  mapfile -t pids < <(port_pids "$port")
  if ((${#pids[@]} > 0)); then
    echo "Port $port still has listener(s), terminating: ${pids[*]}"
    kill -TERM "${pids[@]}" 2>/dev/null || true
    sleep 2
  fi
  if port_listening "$port"; then
    mapfile -t pids < <(port_pids "$port")
    if ((${#pids[@]} > 0)); then
      echo "Force-stopping listener(s) on port $port: ${pids[*]}"
      kill -KILL "${pids[@]}" 2>/dev/null || true
      sleep 1
    fi
  fi

  if port_listening "$port"; then
    echo "FAILED: port $port is still in use"
    return 1
  fi
  echo "Port $port released"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

need_cmd tmux
need_cmd ss

declare -a ports=()
declare -A sessions=()

if (($# > 0)); then
  for port in "$@"; do
    [[ "$port" =~ ^[0-9]+$ ]] || die "invalid port: $port"
    ports+=("$port")
    sessions[$port]="$(session_name "$port")"
  done
else
  [[ -f "$STATE_FILE" ]] || die "no state file found and no ports provided: $STATE_FILE"
  while IFS=$'\t' read -r port session _notebook; do
    [[ -n "${port:-}" ]] || continue
    ports+=("$port")
    sessions[$port]="${session:-$(session_name "$port")}"
  done < "$STATE_FILE"
fi

if ((${#ports[@]} == 0)); then
  echo "No review servers to stop."
  exit 0
fi

failed=0
echo "Stopping review servers:"
for port in "${ports[@]}"; do
  if ! stop_session_and_port "$port" "${sessions[$port]}"; then
    failed=1
  fi
done

if [[ -f "$STATE_FILE" && $# -eq 0 ]]; then
  rm -f "$STATE_FILE"
fi

echo
if ((failed)); then
  echo "FAILED: one or more review ports are still in use."
  exit 1
fi
echo "All requested review servers stopped."
