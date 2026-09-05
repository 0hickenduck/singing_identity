#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_utils import current_git_commit, read_table, stable_json_hash, validate_utterances, write_json  # noqa: E402


def load_encoder_classifier() -> Any:
    try:
        from speechbrain.inference.speaker import EncoderClassifier

        return EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier

        return EncoderClassifier


def output_path(feature_root: Path, extractor: str, checkpoint_hash: str, layer: str, utt_id: str) -> Path:
    return feature_root / extractor / checkpoint_hash / layer / f"{utt_id}.npz"


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract SpeechBrain ECAPA-TDNN speaker embeddings.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--extractor", default="ecapa_tdnn")
    parser.add_argument("--model-name", default="speechbrain/spkrec-ecapa-voxceleb")
    parser.add_argument("--checkpoint-hash", default="speechbrain_spkrec_ecapa_voxceleb")
    parser.add_argument("--layer", default="embedding")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--savedir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summary-out", type=Path)
    args = parser.parse_args()

    try:
        import torch
        import torchaudio
    except ImportError as exc:
        print(f"error: missing dependencies: {exc}", file=sys.stderr)
        return 2

    try:
        EncoderClassifier = load_encoder_classifier()
    except ImportError as exc:
        print(f"error: missing speechbrain dependency: {exc}", file=sys.stderr)
        return 2

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"device: {device}")

    rows = read_table(args.manifest)
    validate_utterances(rows)
    if args.limit:
        rows = rows[: args.limit]

    savedir = args.savedir or args.feature_root / args.extractor / args.checkpoint_hash / "_speechbrain_model"
    try:
        classifier = EncoderClassifier.from_hparams(
            source=args.model_name,
            savedir=str(savedir),
            run_opts={"device": device},
        )
        classifier.eval()
    except Exception as exc:  # noqa: BLE001
        print(f"error: failed to load {args.model_name}: {exc}", file=sys.stderr)
        return 2

    written = 0
    skipped = 0
    failed = 0
    feature_dim = None
    sample_rate = 16000
    failures: list[dict[str, str]] = []

    for idx, row in enumerate(rows, start=1):
        out = output_path(args.feature_root, args.extractor, args.checkpoint_hash, args.layer, str(row["utt_id"]))
        if args.skip_existing and out.exists():
            skipped += 1
            continue
        try:
            wav_path = Path(row["wav_path"])
            signal, sr = torchaudio.load(wav_path)
            signal = signal.mean(dim=0)
            if sr != sample_rate:
                signal = torchaudio.functional.resample(signal, sr, sample_rate)
            signal = signal.to(device).unsqueeze(0)
            with torch.inference_mode():
                emb = classifier.encode_batch(signal)
            x = emb.squeeze().detach().cpu().numpy().astype(np.float32)
            if x.ndim != 1:
                x = x.reshape(-1).astype(np.float32)
            feature_dim = int(x.shape[0])
            duration_sec = float(signal.shape[-1] / sample_rate)
            rms = float(row.get("rms_db", 0.0) or 0.0)
            f0 = float(row.get("f0_mean_hz", 0.0) or 0.0)
            payload = {
                "x": x[None, :],
                "times_sec": np.asarray([0.5 * duration_sec], dtype=np.float32),
                "voiced_mask": np.asarray([True], dtype=bool),
                "phone_id": np.asarray([0], dtype=np.int32),
                "f0_hz": np.asarray([f0], dtype=np.float32),
                "energy": np.asarray([rms], dtype=np.float32),
            }
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, **payload)
            written += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if len(failures) < 20:
                failures.append({"utt_id": str(row.get("utt_id", "")), "error": str(exc)})
        if idx <= 3 or idx % 50 == 0:
            print(f"{idx}/{len(rows)} written={written} skipped={skipped} failed={failed}")

    metadata = {
        "extractor_name": args.extractor,
        "checkpoint_name": args.model_name,
        "checkpoint_hash": args.checkpoint_hash,
        "git_commit": current_git_commit(),
        "layer": args.layer,
        "sample_rate": sample_rate,
        "feature_dim": feature_dim,
        "dtype": "float32",
        "normalization": "speechbrain_default",
        "input_manifest_hash": stable_json_hash(rows),
        "command": "scripts/data_prep/extract_ecapa_features.py",
        "num_manifest_rows": len(rows),
        "num_files_written": written,
        "num_files_skipped": skipped,
        "num_failures": failed,
        "failures": failures,
    }
    metadata_path = args.summary_out or args.feature_root / args.extractor / args.checkpoint_hash / args.layer / "metadata.json"
    write_json(metadata, metadata_path)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
