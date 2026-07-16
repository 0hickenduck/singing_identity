#!/usr/bin/env bash
set -u

REMOTE_TIMEOUT="${REMOTE_TIMEOUT:-8}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-3}"
SHOW_NIC=0

usage() {
  cat <<'USAGE'
Usage:
  scripts/valkyrie-status.sh [--nic] [host...]

Examples:
  scripts/valkyrie-status.sh
  scripts/valkyrie-status.sh valkyrie02 valkyrie03
  scripts/valkyrie-status.sh --nic

Environment:
  REMOTE_TIMEOUT=8          Max seconds per host query
  SSH_CONNECT_TIMEOUT=3     Max seconds for SSH connection setup
USAGE
}

hosts=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --nic)
      SHOW_NIC=1
      shift
      ;;
    *)
      hosts+=("$1")
      shift
      ;;
  esac
done

if [[ ${#hosts[@]} -eq 0 ]]; then
  hosts=(valkyrie01 valkyrie02 valkyrie03 valkyrie04 valkyrie05 valkyrie06 valkyrie07 valkyrie08)
fi

ssh_opts=(
  -o BatchMode=yes
  -o ConnectTimeout="$SSH_CONNECT_TIMEOUT"
  -o StrictHostKeyChecking=accept-new
  -o LogLevel=ERROR
)

remote_script='
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH

host=$(hostname -s 2>/dev/null || hostname)
load=$(cut -d " " -f 1-3 /proc/loadavg 2>/dev/null || printf "-")
users=$(who 2>/dev/null | awk "{print \$1}" | sort -u | paste -sd, -)
user_count=$(who 2>/dev/null | awk "{print \$1}" | sort -u | wc -l | tr -d " ")
tmux_count=$(tmux ls 2>/dev/null | wc -l | tr -d " ")
uptime_pretty=$(uptime -p 2>/dev/null || printf "-")

gpu_summary="-"
suggestion="OK"
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_summary=$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null |
    awk -F, "
      BEGIN { busy=0; free=0 }
      {
        name=\$1; used=\$2; total=\$3; util=\$4
        gsub(/^ +| +$/, \"\", name)
        gsub(/^ +| +$/, \"\", used)
        gsub(/^ +| +$/, \"\", total)
        gsub(/^ +| +$/, \"\", util)
        if (used + 0 < 1024 && util + 0 < 20) free++
        if (used + 0 > 8192 || util + 0 > 50) busy++
        part=sprintf(\"%s %s/%sMiB %s%%\", name, used, total, util)
        out = out ? out \"; \" part : part
      }
      END {
        if (out == \"\") out=\"nvidia-smi failed\"
        print out
        if (busy > 0) exit 10
        if (free > 0) exit 20
      }")
  gpu_state=$?
  if [ "$gpu_state" -eq 10 ]; then
    suggestion="BUSY"
  elif [ "$gpu_state" -eq 20 ] && [ "$user_count" -eq 0 ]; then
    suggestion="GOOD"
  elif [ "$gpu_state" -eq 20 ]; then
    suggestion="CHECK"
  else
    suggestion="OK"
  fi
fi

if [ -z "$users" ]; then
  users="-"
fi

nic_summary="-"
if [ "${SHOW_NIC:-0}" = "1" ]; then
  nic_summary=$(lspci 2>/dev/null |
    awk "/Ethernet controller|Network controller/ { sub(/^[^ ]+ /, \"\"); print }" |
    paste -sd "; " -)
  if [ -z "$nic_summary" ]; then
    nic_summary="-"
  fi
fi

if [ "${SHOW_NIC:-0}" = "1" ]; then
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$host" "$suggestion" "$users" "$load" "$tmux_count" "$gpu_summary" "$nic_summary" "$uptime_pretty"
else
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$host" "$suggestion" "$users" "$load" "$tmux_count" "$gpu_summary" "$uptime_pretty"
fi
'

if [[ "$SHOW_NIC" -eq 1 ]]; then
  printf '%-12s %-8s %-14s %-18s %-6s %-55s %-45s %s\n' HOST PICK USERS LOAD TMUX GPU NIC UPTIME
else
  printf '%-12s %-8s %-14s %-18s %-6s %-55s %s\n' HOST PICK USERS LOAD TMUX GPU UPTIME
fi

for host in "${hosts[@]}"; do
  output=$(
    timeout "$REMOTE_TIMEOUT" ssh "${ssh_opts[@]}" "$host" "SHOW_NIC=$SHOW_NIC; export SHOW_NIC; $remote_script" 2>&1
  )
  status=$?
  if [[ "$SHOW_NIC" -eq 1 ]]; then
    line=$(printf '%s\n' "$output" | awk -F '\t' 'NF >= 8 { last=$0 } END { print last }')
  else
    line=$(printf '%s\n' "$output" | awk -F '\t' 'NF >= 7 { last=$0 } END { print last }')
  fi

  if [[ "$status" -eq 0 && -n "$line" ]]; then
    if [[ "$SHOW_NIC" -eq 1 ]]; then
      IFS=$'\t' read -r rhost pick users load tmux_count gpu nic uptime_pretty <<<"$line"
      printf '%-12s %-8s %-14s %-18s %-6s %-55.55s %-45.45s %s\n' "$rhost" "$pick" "$users" "$load" "$tmux_count" "$gpu" "$nic" "$uptime_pretty"
    else
      IFS=$'\t' read -r rhost pick users load tmux_count gpu uptime_pretty <<<"$line"
      printf '%-12s %-8s %-14s %-18s %-6s %-55.55s %s\n' "$rhost" "$pick" "$users" "$load" "$tmux_count" "$gpu" "$uptime_pretty"
    fi
  else
    reason=$(printf '%s\n' "$output" |
      grep -v 'npmrc file' |
      grep -v 'globalconfig' |
      grep -v 'nvm use --delete-prefix' |
      sed '/^$/d' |
      head -1)
    if [[ -z "$reason" && "$status" -eq 124 ]]; then
      reason="timeout after ${REMOTE_TIMEOUT}s"
    elif [[ -z "$reason" ]]; then
      reason="ssh failed"
    fi

    if [[ "$SHOW_NIC" -eq 1 ]]; then
      printf '%-12s %-8s %-14s %-18s %-6s %-55s %-45s %s\n' "$host" DOWN "-" "-" "-" "$reason" "-" "-"
    else
      printf '%-12s %-8s %-14s %-18s %-6s %-55s %s\n' "$host" DOWN "-" "-" "-" "$reason" "-"
    fi
  fi
done
