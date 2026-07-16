#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_utils import ExperimentError, current_git_commit, read_table, stable_json_hash, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract WavLM hidden-state features to the project NPZ schema.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--model-name", default="microsoft/wavlm-base-plus")
    parser.add_argument("--checkpoint-hash", default="microsoft_wavlm_base_plus")
    parser.add_argument("--layers", nargs="+", default=["3", "6", "9", "12"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        import torchaudio
        from transformers import AutoFeatureExtractor, WavLMModel
    except ImportError as exc:
        print(f"error: missing WavLM dependencies: {exc}", file=sys.stderr)
        return 2

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"device: {device}")

    rows = read_table(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    layers = [int(layer) for layer in args.layers]
    try:
        processor = AutoFeatureExtractor.from_pretrained(args.model_name)
        model = WavLMModel.from_pretrained(args.model_name, output_hidden_states=True).to(device)
        model.eval()
    except Exception as exc:  # noqa: BLE001
        print(f"error: failed to load {args.model_name}: {exc}", file=sys.stderr)
        return 2

    written = {str(layer): 0 for layer in layers}
    skipped = {str(layer): 0 for layer in layers}
    try:
        for idx, row in enumerate(rows, 1):
            needed_layers = [
                layer
                for layer in layers
                if not (
                    args.skip_existing
                    and (args.feature_root / "wavlm_base_plus" / args.checkpoint_hash / str(layer) / f"{row['utt_id']}.npz").exists()
                )
            ]
            if not needed_layers:
                for layer in layers:
                    skipped[str(layer)] += 1
                continue
            wav, sr = torchaudio.load(row["wav_path"])
            wav = wav.mean(dim=0)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            inputs = processor(wav.numpy(), sampling_rate=16000, return_tensors="pt")
            input_values = inputs.input_values.to(device)
            with torch.inference_mode():
                outputs = model(input_values)
            hidden_states = outputs.hidden_states
            for layer in needed_layers:
                if layer >= len(hidden_states):
                    raise ExperimentError(f"layer {layer} unavailable; model returned {len(hidden_states)} hidden states")
                x = hidden_states[layer].squeeze(0).detach().cpu().numpy().astype(np.float32)
                duration_sec = float(wav.shape[0] / 16000)
                frame_step = duration_sec / max(1, x.shape[0])
                times = (np.arange(x.shape[0]) * frame_step).astype(np.float32)
                payload = {
                    "x": x,
                    "times_sec": times,
                    "voiced_mask": np.ones(x.shape[0], dtype=bool),
                    "phone_id": np.zeros(x.shape[0], dtype=np.int32),
                    "f0_hz": np.zeros(x.shape[0], dtype=np.float32),
                    "energy": np.zeros(x.shape[0], dtype=np.float32),
                }
                out = args.feature_root / "wavlm_base_plus" / args.checkpoint_hash / str(layer) / f"{row['utt_id']}.npz"
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez(out, **payload)
                written[str(layer)] += 1
            for layer in set(layers) - set(needed_layers):
                skipped[str(layer)] += 1
            if idx <= 3 or idx % 10 == 0:
                print(f"{idx}/{len(rows)} {row['utt_id']} written={written} skipped={skipped} x shape: {x.shape}")
    except (ExperimentError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for layer in layers:
        metadata = {
            "extractor_name": "wavlm_base_plus",
            "checkpoint_name": args.model_name,
            "checkpoint_hash": args.checkpoint_hash,
            "git_commit": current_git_commit(),
            "layer": layer,
            "sample_rate": 16000,
            "hop_sec": 0.02,
            "feature_dim": 768,
            "dtype": "float32",
            "normalization": "processor_default",
            "input_manifest_hash": stable_json_hash(rows),
            "command": "scripts/data_prep/extract_wavlm_features.py",
            "num_files": written[str(layer)],
            "num_files_skipped": skipped[str(layer)],
        }
        write_json(metadata, args.feature_root / "wavlm_base_plus" / args.checkpoint_hash / str(layer) / "metadata.json")
    print(json.dumps({"written": written, "device": device}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
