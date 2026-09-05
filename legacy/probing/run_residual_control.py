#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_utils import ExperimentError, write_json, write_table  # noqa: E402
from probing.run_mode_probe import run_probe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the mandatory F0/energy/duration/voiced-ratio residualized mode probe."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--extractor", required=True)
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--run-id", default="residual_control_mode_probe")
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--predictions-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--voiced-only", action="store_true")
    parser.add_argument("--shuffle-labels", action="store_true")
    parser.add_argument("--binary-singing", action="store_true")
    args = parser.parse_args()
    args.residualize_nuisance = True
    try:
        result = run_probe(args)
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_json(result["metrics"], args.metrics_out)
    write_table(result["predictions"], args.predictions_out)
    print(f"residualized auc: {result['metrics']['metrics']['auc']:.4f}")
    print(f"wrote metrics -> {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
