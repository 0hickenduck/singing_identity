#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper. The lab /work path has been observed as NFS on
# valkyrie03, so the canonical entrypoint now uses /localdisk/bowen.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${REPO_ROOT}/scripts/resume_stage1_repaired_200_on_localdisk.sh" "$@"
