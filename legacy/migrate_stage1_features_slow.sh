#!/usr/bin/env bash
set -euo pipefail

# Slowly migrate the repaired Stage 1 feature cache from /home to localdisk.
# Run inside tmux. This is intentionally throttled because the source is NFS.

SRC="${SRC:-/home/bowen/bowen_lab/projects/singing_identity_stage1_features/repaired_200_allmodels}"
DST="${DST:-/localdisk/bowen/singing_identity/features/repaired_200_allmodels}"
LOG_FILE="${LOG_FILE:-/localdisk/bowen/singing_identity/logs/migrate_stage1_features_slow.log}"
BWLIMIT_KB="${BWLIMIT_KB:-2048}"
SLEEP_SEC="${SLEEP_SEC:-60}"
DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-}"

mkdir -p "${DST}" "$(dirname "${LOG_FILE}")"

run_rsync() {
  local src="$1"
  local dst="$2"
  local -a cmd=(
    rsync
    -a
    --partial
    --info=progress2,stats2
    --bwlimit="${BWLIMIT_KB}"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    cmd+=(-n)
  fi
  cmd+=("${src}" "${dst}")
  if command -v ionice >/dev/null 2>&1; then
    ionice -c2 -n7 nice -n 10 "${cmd[@]}"
  else
    nice -n 10 "${cmd[@]}"
  fi
}

{
  echo "== Slow Stage 1 feature migration =="
  date
  hostname
  echo "src=${SRC}"
  echo "dst=${DST}"
  echo "bwlimit_kb=${BWLIMIT_KB}"
  echo "sleep_sec=${SLEEP_SEC}"
  echo "dry_run=${DRY_RUN}"
  echo
} | tee -a "${LOG_FILE}"

shopt -s nullglob
if [[ -n "${ONLY}" ]]; then
  read -r -a names <<< "${ONLY}"
  children=()
  for name in "${names[@]}"; do
    children+=("${SRC}/${name}")
  done
else
  children=("${SRC}"/*)
fi
if ((${#children[@]} == 0)); then
  echo "No feature subdirectories found under ${SRC}" | tee -a "${LOG_FILE}"
  exit 1
fi

for child in "${children[@]}"; do
  if [[ ! -d "${child}" ]]; then
    echo "Skipping missing feature subdirectory ${child}" | tee -a "${LOG_FILE}"
    continue
  fi
  name="$(basename "${child}")"
  target="${DST}/${name}"
  mkdir -p "${target}"
  {
    echo
    echo "== Migrating ${name} =="
    date
  } | tee -a "${LOG_FILE}"
  run_rsync "${child}/" "${target}/" 2>&1 | tee -a "${LOG_FILE}"
  {
    echo "== Finished ${name}; sleeping ${SLEEP_SEC}s =="
    date
  } | tee -a "${LOG_FILE}"
  sleep "${SLEEP_SEC}"
done

{
  echo
  echo "== Slow Stage 1 feature migration complete =="
  date
} | tee -a "${LOG_FILE}"
