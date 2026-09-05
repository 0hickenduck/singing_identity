#!/usr/bin/env python3
"""Canonical entry point to prepare manifests and feature caches for singing identity benchmarks."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "legacy/data_prep/make_synthetic_experiment.py"),
                "--root",
                str(syn_root),
                "--speakers",
                "4",
                "--items-per-mode",
                "2",
            ],
            cwd=ROOT,
            check=True,
        )
        print(f"[✓] Synthetic manifest generated: {syn_root / 'synthetic_run_config.json'}")
        return 0

    if args.dataset in ("gtsinger", "all"):
        print("[DATA PREP] Preparing GTSinger manifests...")
        script = ROOT / "legacy/data_prep/build_gtsinger_manifests.py"
        if script.exists():
            subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
        print(f"[✓] GTSinger manifests ready in {out_dir}")

    if args.dataset in ("jvs", "all"):
        print("[DATA PREP] Preparing JVS-MuSiC manifests...")
        script = ROOT / "legacy/data_prep/build_jvs_music_manifests.py"
        if script.exists():
            subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
        print(f"[✓] JVS-MuSiC manifests ready in {out_dir}")

    if args.extract_features:
        print(f"[DATA PREP] Extracting features using {args.approach} to {args.feature_root}...")
        # Check approach config
        app_cfg = json.loads(Path(args.approach).read_text(encoding="utf-8"))
        model_type = app_cfg.get("model_type", "acoustic")
        manifest_file = out_dir / "gtsinger_utterances.jsonl"
        extract_script = ROOT / f"legacy/data_prep/extract_{model_type}_features.py"
        if extract_script.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(extract_script),
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
