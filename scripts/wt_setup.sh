#!/usr/bin/env bash
# One-time setup for a fresh worktree. Idempotent. Run by Orca hook or by hand.
set -euo pipefail
case "$(hostname)" in athena*) echo "refusing to run on athena"; exit 1;; esac
findmnt -T /localdisk/bowen -o TARGET,SOURCE,FSTYPE,AVAIL || { echo "/localdisk/bowen not mounted on $(hostname)"; exit 1; }
mkdir -p /localdisk/bowen/singing_identity/{data,features,runs,logs,checkpoints} /localdisk/bowen/.cache /localdisk/bowen/tmp
uv sync
echo "worktree ready: $(pwd) @ $(git rev-parse --short HEAD) on $(hostname)"
