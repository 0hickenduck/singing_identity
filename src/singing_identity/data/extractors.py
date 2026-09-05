#!/usr/bin/env python3
"""Acoustic feature extraction for Stage 1 experiments."""
from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

from singing_identity.utils.research_utils import (
    ExperimentError,
    current_git_commit,
    read_table,
    stable_json_hash,
    write_json,
)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        sr = handle.getframerate()
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    if width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        signed = (
            bytes_[:, 0].astype(np.int32)
            | (bytes_[:, 1].astype(np.int32) << 8)
            | (bytes_[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed | ~0xFFFFFF, signed)
        audio = signed.astype(np.float32) / 8388608.0
    elif width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ExperimentError(f"unsupported sample width {width}: {path}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sr


def frame_audio(audio: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    if len(audio) < frame_len:
        padded = np.zeros(frame_len, dtype=np.float32)
        padded[: len(audio)] = audio
        return padded[None, :]
    n = 1 + (len(audio) - frame_len) // hop
    shape = (n, frame_len)
    strides = (audio.strides[0] * hop, audio.strides[0])
    return np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides).copy()


def spectral_features(frames: np.ndarray, sr: int) -> np.ndarray:
    window = np.hanning(frames.shape[1]).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * window[None, :], axis=1)).astype(np.float64)
    freqs = np.fft.rfftfreq(frames.shape[1], 1.0 / sr)
    power_sum = np.maximum(spec.sum(axis=1), 1e-10)
    centroid = (spec @ freqs) / power_sum
    bandwidth = np.sqrt(((freqs[None, :] - centroid[:, None]) ** 2 * spec).sum(axis=1) / power_sum)
    cumsum = np.cumsum(spec, axis=1)
    rolloff_idx = (cumsum >= 0.85 * power_sum[:, None]).argmax(axis=1)
    rolloff = freqs[rolloff_idx]
    flatness = np.exp(np.mean(np.log(np.maximum(spec, 1e-10)), axis=1)) / np.maximum(spec.mean(axis=1), 1e-10)
    return np.column_stack([centroid, bandwidth, rolloff, flatness])


def extract_acoustic(audio: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 20.0) -> dict[str, np.ndarray]:
    frame_len = max(16, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    frames = frame_audio(audio, frame_len, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    abs_mean = np.mean(np.abs(frames), axis=1)
    peak = np.max(np.abs(frames), axis=1)
    zcr = np.mean(np.diff(np.signbit(frames), axis=1), axis=1).astype(np.float64)
    spec = spectral_features(frames, sr)
    log_rms = 20.0 * np.log10(np.maximum(rms, 1e-8))
    x = np.column_stack([rms, log_rms, abs_mean, peak, zcr, spec]).astype(np.float32)
    times = (np.arange(len(frames)) * hop / sr).astype(np.float32)
    threshold = max(float(np.percentile(rms, 35)), 1e-5)
    voiced = rms > threshold
    return {
        "x": x,
        "times_sec": times,
        "voiced_mask": voiced.astype(bool),
        "phone_id": np.zeros(len(frames), dtype=np.int32),
        "f0_hz": np.zeros(len(frames), dtype=np.float32),
        "energy": rms.astype(np.float32),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract lightweight acoustic baseline features to NPZ.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--extractor", default="acoustic_baseline")
    parser.add_argument("--checkpoint-hash", default="local_wave_v1")
    parser.add_argument("--layer", default="frame25ms_hop20ms")
    parser.add_argument("--metadata-out", type=Path)
    parser.add_argument("--frame-ms", type=float, default=25.0)
    parser.add_argument("--hop-ms", type=float, default=20.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    rows = read_table(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    written = 0
    skipped = 0
    dims = set()
    try:
        for row in rows:
            out = args.feature_root / args.extractor / args.checkpoint_hash / args.layer / f"{row['utt_id']}.npz"
            if args.skip_existing and out.exists():
                skipped += 1
                continue
            path = Path(row["wav_path"])
            audio, sr = read_wav(path)
            payload = extract_acoustic(audio, sr, args.frame_ms, args.hop_ms)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, **payload)
            written += 1
            dims.add(int(payload["x"].shape[1]))
    except (ExperimentError, OSError, wave.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    metadata = {
        "extractor_name": args.extractor,
        "checkpoint_name": args.checkpoint_hash,
        "checkpoint_hash": args.checkpoint_hash,
        "git_commit": current_git_commit(),
        "layer": args.layer,
        "sample_rate": "native",
        "hop_sec": args.hop_ms / 1000.0,
        "feature_dim": sorted(dims),
        "dtype": "float32",
        "normalization": "none",
        "input_manifest_hash": stable_json_hash(rows),
        "command": "singing_identity.data.extractors",
        "num_files": written,
        "num_files_skipped": skipped,
    }
    metadata_path = args.metadata_out or args.feature_root / args.extractor / args.checkpoint_hash / "metadata.json"
    write_json(metadata, metadata_path)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
