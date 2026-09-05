#!/usr/bin/env python3
"""Thin entrypoint to audit and summarize Stage 1 experiment results."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.audit_stage1_results import main

if __name__ == "__main__":
    main()
