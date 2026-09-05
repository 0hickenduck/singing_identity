#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_utils import current_git_commit, read_table, stable_json_hash, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract generic Hugging Face audio model hidden states.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--extractor", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--layers", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        import torchaudio
        from transformers import AutoFeatureExtractor, AutoModel
    except ImportError as exc:
        print(f"error: missing dependencies: {exc}", file=sys.stderr)
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
        processor = AutoFeatureExtractor.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
        model = AutoModel.from_pretrained(
            args.model_name,
            trust_remote_code=args.trust_remote_code,
            output_hidden_states=True,
        ).to(device)
        model.eval()
    except Exception as exc:  # noqa: BLE001
        print(f"error: failed to load {args.model_name}: {exc}", file=sys.stderr)
        return 2

    target_sr = int(getattr(processor, "sampling_rate", 16000) or 16000)
    written = {str(layer): 0 for layer in layers}
    skipped = {str(layer): 0 for layer in layers}
    feature_dim = None
    try:
        for idx, row in enumerate(rows, 1):
            needed_layers = [
                layer
                for layer in layers
                if not (
                    args.skip_existing
                    and (args.feature_root / args.extractor / args.checkpoint_hash / str(layer) / f"{row['utt_id']}.npz").exists()
                )
            ]
            if not needed_layers:
                for layer in layers:
                    skipped[str(layer)] += 1
                continue
            wav, sr = torchaudio.load(row["wav_path"])
            wav = wav.mean(dim=0)
            if sr != target_sr:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
            inputs = processor(wav.numpy(), sampling_rate=target_sr, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = model(**inputs)
            hidden_states = outputs.hidden_states
            for layer in needed_layers:
                x = hidden_states[layer].squeeze(0).detach().cpu().numpy().astype(np.float32)
                feature_dim = int(x.shape[1])
                duration_sec = float(wav.shape[0] / target_sr)
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
                out = args.feature_root / args.extractor / args.checkpoint_hash / str(layer) / f"{row['utt_id']}.npz"
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez(out, **payload)
                written[str(layer)] += 1
            for layer in set(layers) - set(needed_layers):
                skipped[str(layer)] += 1
            if idx <= 3 or idx % 10 == 0:
                print(f"{idx}/{len(rows)} {row['utt_id']} written={written} skipped={skipped} x shape: {x.shape}")
    except Exception as exc:  # noqa: BLE001
        print(f"error: extraction failed: {exc}", file=sys.stderr)
        return 2

    for layer in layers:
        metadata = {
            "extractor_name": args.extractor,
            "checkpoint_name": args.model_name,
            "checkpoint_hash": args.checkpoint_hash,
            "git_commit": current_git_commit(),
            "layer": layer,
            "sample_rate": target_sr,
            "hop_sec": 0.02,
            "feature_dim": feature_dim,
            "dtype": "float32",
            "normalization": "processor_default",
            "input_manifest_hash": stable_json_hash(rows),
            "command": "scripts/data_prep/extract_hf_audio_features.py",
            "num_files": written[str(layer)],
            "num_files_skipped": skipped[str(layer)],
        }
        write_json(metadata, args.feature_root / args.extractor / args.checkpoint_hash / str(layer) / "metadata.json")
    print(json.dumps({"written": written, "device": device, "sample_rate": target_sr}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
