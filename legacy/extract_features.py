#!/usr/bin/env python3
"""Unified CLI entrypoint to generate and reconstruct benchmark feature caches."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from singing_identity.utils.provenance import resolve_feature_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract and reconstruct audio feature caches with provenance.")
    parser.add_argument("--approach", type=Path, required=True, help="Path to approach config (e.g. configs/approaches/wavlm_base_plus.json)")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/gtsinger_utterances.jsonl"), help="Input manifest JSONL")
    parser.add_argument("--feature-root", type=Path, default=None, help="Root directory for feature caches")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--limit", type=int, default=None, help="Limit number of utterances for smoke/test runs")
    parser.add_argument("--skip-existing", action="store_true", help="Skip utterances with existing feature files")
    args = parser.parse_args()

    feat_root = resolve_feature_root(args.feature_root)
    feat_root.mkdir(parents=True, exist_ok=True)

    if not args.approach.exists():
        print(f"error: approach config not found: {args.approach}", file=sys.stderr)
        return 1

    app_cfg = json.loads(args.approach.read_text(encoding="utf-8"))
    name = app_cfg.get("name", args.approach.stem)
    print(f"Reconstructing feature cache for approach '{name}' at '{feat_root}'...")

    cmd: list[str] = []
    if "wavlm" in name.lower():
        cmd = [
            sys.executable,
            str(ROOT / "scripts/data_prep/extract_wavlm_features.py"),
            "--manifest", str(args.manifest),
            "--feature-root", str(feat_root),
            "--device", args.device,
        ]
        if "layers" in app_cfg:
            cmd.extend(["--layers"] + [str(l) for l in app_cfg["layers"]])
        if "model_name" in app_cfg or "hf_model_id" in app_cfg:
            cmd.extend(["--model-name", app_cfg.get("model_name") or app_cfg.get("hf_model_id")])
    elif "mert" in name.lower() or "hubert" in name.lower():
        model_key = "mert" if "mert" in name.lower() else "hubert"
        cmd = [
            sys.executable,
            str(ROOT / "scripts/data_prep/extract_hf_audio_features.py"),
            "--manifest", str(args.manifest),
            "--feature-root", str(feat_root),
            "--model", model_key,
            "--device", args.device,
        ]
        if "layers" in app_cfg:
            cmd.extend(["--layers"] + [str(l) for l in app_cfg["layers"]])
    elif "acoustic" in name.lower():
        cmd = [
            sys.executable,
            str(ROOT / "scripts/data_prep/extract_acoustic_features.py"),
            "--manifest", str(args.manifest),
            "--feature-root", str(feat_root),
        ]
    elif "ecapa" in name.lower():
        cmd = [
            sys.executable,
            str(ROOT / "scripts/data_prep/extract_ecapa_features.py"),
            "--manifest", str(args.manifest),
            "--feature-root", str(feat_root),
            "--device", args.device,
        ]
    else:
        print(f"error: unknown approach feature extractor: {name}", file=sys.stderr)
        return 1

    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    if args.skip_existing:
        cmd.append("--skip-existing")

    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=ROOT)
    return res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
