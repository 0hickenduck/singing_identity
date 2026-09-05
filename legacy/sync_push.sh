#!/bin/bash
# sync_push.sh — commit and push code/docs/compact results to GitHub.
# Usage: bash scripts/sync_push.sh "short message"
# Mechanically skips any file larger than MAX_KB; large artifacts belong on
# /localdisk, never in the repo. Safe to run any time; no-op when clean.
set -euo pipefail
cd "$(dirname "$0")/.."
MAX_KB=2048

git pull --ff-only

skipped=()
while IFS= read -r f; do
    if [ -f "$f" ] && [ "$(du -k -- "$f" | cut -f1)" -gt "$MAX_KB" ]; then
        skipped+=("$f")
    else
        git add -- "$f"
    fi
done < <(git ls-files -om --exclude-standard)

if [ "${#skipped[@]}" -gt 0 ]; then
    echo "SKIPPED files over ${MAX_KB}KB (keep these on /localdisk, not in git):"
    printf '  %s\n' "${skipped[@]}"
fi

if git diff --cached --quiet; then
    echo "Nothing to commit."
else
    git commit -m "sync: ${1:-update code/docs/results} ($(date +%F))"
fi

git push
echo "Pushed to $(git remote get-url origin)"
