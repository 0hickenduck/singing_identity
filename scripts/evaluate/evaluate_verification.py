#!/usr/bin/env python3
"""Thin entrypoint to run identity residual verification evaluation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from legacy.probing.run_identity_residual_final_validation import main

if __name__ == "__main__":
    main()
