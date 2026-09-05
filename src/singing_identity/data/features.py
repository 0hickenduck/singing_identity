#!/usr/bin/env python3
"""Feature cache inspection and validation utilities."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from singing_identity.utils.research_utils import (
    ExperimentError,
    feature_path,
    read_table,
    validate_feature_npz,
    write_json,
)


def validate_manifest_features(
    manifest_path: Path,
    feature_root: Path,
    extractor: str,
    checkpoint_hash: str,
    layer: str,
) -> dict[str, Any]:
    """Validate all features referenced by manifest exist and are well-formed."""
    rows = read_table(manifest_path)
    checked = []
    for row in rows:
        path = feature_path(feature_root, extractor, checkpoint_hash, layer, row["utt_id"])
        if not path.exists():
            raise ExperimentError(f"Missing feature file: {path}")
        frames, dim = validate_feature_npz(path)
        checked.append({"utt_id": row["utt_id"], "frames": frames, "dim": dim})

    dims = sorted({item["dim"] for item in checked})
    return {
        "num_files": len(checked),
        "dims": dims,
        "min_frames": min(item["frames"] for item in checked) if checked else 0,
        "max_frames": max(item["frames"] for item in checked) if checked else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate cached .npz feature files against a manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--extractor", required=True)
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = validate_manifest_features(
            args.manifest, args.feature_root, args.extractor, args.checkpoint_hash, args.layer
        )
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    write_json(report, args.report)
    print(f"validated {report['num_files']} feature files -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
