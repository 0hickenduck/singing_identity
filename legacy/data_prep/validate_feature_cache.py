#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_utils import (  # noqa: E402
    ExperimentError,
    feature_path,
    read_table,
    validate_feature_npz,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate cached .npz feature files against a manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--extractor", required=True)
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows = read_table(args.manifest)
    checked = []
    try:
        for row in rows:
            path = feature_path(args.feature_root, args.extractor, args.checkpoint_hash, args.layer, row["utt_id"])
            if not path.exists():
                raise ExperimentError(f"Missing feature file: {path}")
            frames, dim = validate_feature_npz(path)
            checked.append({"utt_id": row["utt_id"], "frames": frames, "dim": dim})
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    dims = sorted({item["dim"] for item in checked})
    report = {
        "num_files": len(checked),
        "dims": dims,
        "min_frames": min(item["frames"] for item in checked) if checked else 0,
        "max_frames": max(item["frames"] for item in checked) if checked else 0,
    }
    write_json(report, args.report)
    print(f"validated {len(checked)} feature files -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
