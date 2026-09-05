#!/usr/bin/env python3
"""Validates that benchmark runs satisfy the experiment provenance contract.

Checks:
  1. run_id and folder structure exist
  2. resolved config.json exists
  3. full 40-char Git commit and dirty flag are recorded
  4. If dirty, git_patch.diff artifact exists
  5. Dataset manifest and manifest_sha256 exist
  6. Execution identity (command, seed, host, python) is recorded
  7. metrics.json exists with valid schema
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from singing_identity.utils.provenance import validate_run_provenance


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and validate experiment run provenance.")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Run director(ies) to audit")
    parser.add_argument("--strict", action="store_true", help="Exit with error if any run fails or is partial")
    args = parser.parse_args()

    all_passed = True
    for run_dir in args.run_dirs:
        val = validate_run_provenance(run_dir)
        status = val["status"].upper()
        icon = "✓" if val["valid"] else "✗"
        print(f"\n[{icon}] Run: {run_dir.name} ({status})")
        print(f"    Path: {run_dir}")

        if val["warnings"]:
            for w in val["warnings"]:
                print(f"    [WARN]  {w}")
        if val["errors"]:
            for e in val["errors"]:
                print(f"    [ERROR] {e}")
            all_passed = False

        if args.strict and status != "COMPLETE":
            all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
