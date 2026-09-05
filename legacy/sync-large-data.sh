#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/sync-large-data.sh push <user@host:/absolute/project/path> [local_project_root]
  scripts/sync-large-data.sh pull <user@host:/absolute/project/path> [local_project_root]

Environment:
  SYNC_PATHS   Space-separated paths to sync. Default: "data outputs"
  DRY_RUN      Set to 1 to print what would sync without writing.
  SYNC_DELETE  Set to 1 to pass --delete to rsync. Default: off.

Examples:
  DRY_RUN=1 scripts/sync-large-data.sh push bowen@valkyrie08:/home/bowen/bowen_lab/projects/arti6_linearvc
  SYNC_PATHS="data/vtuber_curated_conservative data/manifests" scripts/sync-large-data.sh pull bowen@valkyrie08:/home/bowen/bowen_lab/projects/arti6_linearvc
USAGE
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 2
fi

direction="$1"
remote_root="${2%/}"
local_root="${3:-$(pwd)}"
local_root="${local_root%/}"

if [[ "$direction" != "push" && "$direction" != "pull" ]]; then
  usage
  exit 2
fi

if [[ ! -d "$local_root/.git" ]]; then
  echo "Local project root does not look like a git repository: $local_root" >&2
  exit 1
fi

read -r -a sync_paths <<< "${SYNC_PATHS:-data outputs}"

rsync_opts=(-avh --progress)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  rsync_opts+=(--dry-run)
fi
if [[ "${SYNC_DELETE:-0}" == "1" ]]; then
  rsync_opts+=(--delete)
fi

rsync_opts+=(
  --exclude '.git/'
  --exclude '.venv/'
  --exclude '__pycache__/'
  --exclude '.huggingface/'
  --exclude 'hf_cache/'
  --exclude 'wandb/'
  --exclude 'mlruns/'
  --exclude 'cookies*.txt'
)

for rel_path in "${sync_paths[@]}"; do
  rel_path="${rel_path%/}"
  [[ -z "$rel_path" ]] && continue

  case "$direction" in
    push)
      src="$local_root/$rel_path/"
      dst="$remote_root/$rel_path/"
      ;;
    pull)
      src="$remote_root/$rel_path/"
      dst="$local_root/$rel_path/"
      ;;
  esac

  echo "Syncing $direction: $rel_path"
  mkdir -p "$local_root/$rel_path"
  rsync "${rsync_opts[@]}" "$src" "$dst"
done
