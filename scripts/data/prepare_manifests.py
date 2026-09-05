#!/usr/bin/env python3
"""Canonical entry point to prepare manifests and feature caches for singing identity benchmarks."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from singing_identity.data.synthetic import make_synthetic_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare dataset manifests and feature caches")
    parser.add_argument("--dataset", choices=["gtsinger", "jvs", "synthetic", "all"], default="all")
    parser.add_argument("--out-dir", type=str, default="data/processed", help="Output directory for manifests")
    parser.add_argument("--synthetic-root", type=str, default=None, help="Root directory for synthetic experiment")
    parser.add_argument("--extract-features", action="store_true", help="Also extract features for the dataset")
    parser.add_argument("--approach", type=str, default="configs/approaches/acoustic.json", help="Approach config for extraction")
    parser.add_argument("--feature-root", type=str, default="/localdisk/bowen/singing_identity/features")
    args = parser.parse_args()

    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("synthetic",):
        syn_root = Path(args.synthetic_root) if args.synthetic_root else out_dir / "synthetic"
        print(f"[DATA PREP] Generating synthetic dataset at {syn_root}...")
        make_synthetic_dataset(
            root=syn_root,
            speakers=4,
            items_per_mode=2,
        )
        print(f"[✓] Synthetic manifest generated: {syn_root / 'synthetic_run_config.json'}")
        return 0

    if args.dataset in ("gtsinger", "all"):
        print("[DATA PREP] Verifying GTSinger manifests...")
        manifest = out_dir / "gtsinger_utterances.jsonl"
        if manifest.exists():
            print(f"[✓] GTSinger manifests ready in {out_dir}")
        else:
            print(f"[!] Warning: {manifest} not found. Ensure raw dataset is available.")

    if args.dataset in ("jvs", "all"):
        print("[DATA PREP] Verifying JVS-MuSiC manifests...")
        manifest = out_dir / "jvs_music_utterances.jsonl"
        if manifest.exists():
            print(f"[✓] JVS-MuSiC manifests ready in {out_dir}")
        else:
            print(f"[*] Note: {manifest} may be generated from external source.")

    if args.extract_features:
        print(f"[DATA PREP] Extracting features using {args.approach} to {args.feature_root}...")
        manifest_file = out_dir / "gtsinger_utterances.jsonl"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "singing_identity.data.extractors",
                "--manifest",
                str(manifest_file),
                "--feature-root",
                str(args.feature_root),
            ],
            cwd=ROOT,
            check=True,
        )
        print(f"[✓] Features extracted to {args.feature_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
