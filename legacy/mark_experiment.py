#!/usr/bin/env python3
"""Project-local wrapper for the shared Codex experiment marker."""

from __future__ import annotations

import subprocess
import sys


PROJECT_CONTEXT = "/home/bowen/bowen_lab/projects/singing_identity/.codex/project_context.json"
MARKER_SCRIPT = "/home/bowen/bowen_lab/codex_hooks/mark_experiment.py"


def main() -> int:
    command = [
        sys.executable,
        MARKER_SCRIPT,
        "--project-context",
        PROJECT_CONTEXT,
        *sys.argv[1:],
    ]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
