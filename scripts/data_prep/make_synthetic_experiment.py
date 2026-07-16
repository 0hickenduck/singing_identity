#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_utils import (  # noqa: E402
    current_git_commit,
    stable_json_hash,
    write_json,
    write_table,
)


def make_feature(
    rng: np.random.Generator,
    identity: np.ndarray,
    mode_shift: np.ndarray,
    technique_shift: np.ndarray,
    dim: int,
    frames: int,
) -> dict[str, np.ndarray]:
    f0 = rng.normal(180.0 + 25.0 * mode_shift[0], 8.0, size=frames).astype(np.float32)
    energy = rng.normal(0.25 + 0.03 * mode_shift[1], 0.03, size=frames).astype(np.float32)
    x = identity + mode_shift + technique_shift + rng.normal(0.0, 0.08, size=(frames, dim))
    return {
        "x": x.astype(np.float32),
        "times_sec": (np.arange(frames) * 0.02).astype(np.float32),
        "voiced_mask": rng.random(frames) > 0.08,
        "phone_id": rng.integers(1, 32, size=frames, dtype=np.int32),
        "f0_hz": f0,
        "energy": energy,
    }


def write_feature(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create tiny deterministic manifests and feature caches for smoke tests.")
    parser.add_argument("--root", type=Path, default=Path("experiments/synthetic"))
    parser.add_argument("--speakers", type=int, default=12)
    parser.add_argument("--items-per-mode", type=int, default=3)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--extractor", default="synthetic_ssl")
    parser.add_argument("--checkpoint-hash", default="synthetic_v1")
    parser.add_argument("--layer", default="layer6")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    feature_root = args.root / "features"
    track1_manifest = args.root / "track1_timbre" / "manifests" / "utterances.jsonl"
    track1_pairs = args.root / "track1_timbre" / "manifests" / "pairs.jsonl"
    track2_manifest = args.root / "track2_technique" / "manifests" / "utterances.jsonl"
    track2_pairs = args.root / "track2_technique" / "manifests" / "phoneme_pairs.jsonl"

    utterances = []
    pairs = []
    technique_pairs = []
    technique_utterances = []
    mode_shift = np.zeros(args.dim)
    mode_shift[:4] = [0.55, -0.35, 0.25, -0.20]
    vibrato_shift = np.zeros(args.dim)
    vibrato_shift[4:8] = [0.45, 0.30, -0.25, 0.20]

    for s in range(args.speakers):
        speaker_id = f"s{s:03d}"
        identity = rng.normal(0.0, 0.7, size=args.dim)
        language = "ja" if s % 2 == 0 else "en"
        for item in range(args.items_per_mode):
            speech_id = f"{speaker_id}_speech_{item:02d}"
            sing_id = f"{speaker_id}_sing_{item:02d}"
            song_id = f"song_{item:02d}"
            phrase_id = f"phrase_{item:02d}"
            base = {
                "speaker_id": speaker_id,
                "language": language,
                "vocal_range": "unknown",
                "song_id": song_id,
                "phrase_id": phrase_id,
                "take_id": "take1",
                "wav_path": "",
                "start_sec": 0.0,
                "end_sec": args.frames * 0.02,
                "duration_sec": args.frames * 0.02,
                "sample_rate": 16000,
                "num_samples": int(args.frames * 0.02 * 16000),
                "text": "synthetic",
                "phone_seq": "a i u",
                "alignment_path": "",
                "control_utt_id": "",
                "split_group_key": f"{speaker_id}_{song_id}_{phrase_id}",
                "snr_db": 30.0,
                "rms_db": -22.0,
                "f0_min_hz": 120.0,
                "f0_max_hz": 260.0,
                "alignment_quality_flag": "ok",
            }
            for utt_id, mode, technique, shift, paired in [
                (speech_id, "speech", "none", np.zeros(args.dim), sing_id),
                (sing_id, "singing", "control", mode_shift, speech_id),
            ]:
                payload = make_feature(rng, identity, shift, np.zeros(args.dim), args.dim, args.frames)
                row = {
                    **base,
                    "utt_id": utt_id,
                    "mode": mode,
                    "technique": technique,
                    "paired_utt_id": paired,
                    "f0_mean_hz": float(payload["f0_hz"].mean()),
                    "f0_std_hz": float(payload["f0_hz"].std()),
                    "f0_voiced_pct": float(payload["voiced_mask"].mean()),
                    "energy_mean": float(payload["energy"].mean()),
                    "energy_std": float(payload["energy"].std()),
                }
                utterances.append(row)
                write_feature(
                    feature_root / args.extractor / args.checkpoint_hash / args.layer / f"{utt_id}.npz",
                    payload,
                )
            pairs.append(
                {
                    "pair_id": f"{speaker_id}_{item:02d}",
                    "speech_utt_id": speech_id,
                    "singing_utt_id": sing_id,
                    "speaker_id": speaker_id,
                    "language": language,
                    "song_id": song_id,
                    "phrase_id": phrase_id,
                    "technique": "control",
                    "pair_type": "same_singer_same_phrase",
                    "same_text_flag": "true",
                    "same_song_flag": "true",
                }
            )

            normal_id = f"{speaker_id}_normal_{item:02d}"
            vibrato_id = f"{speaker_id}_vibrato_{item:02d}"
            for utt_id, technique, shift in [
                (normal_id, "control", mode_shift),
                (vibrato_id, "vibrato", mode_shift + vibrato_shift),
            ]:
                payload = make_feature(rng, identity, shift, np.zeros(args.dim), args.dim, args.frames)
                row = {
                    **base,
                    "utt_id": utt_id,
                    "mode": "singing_technique" if technique != "control" else "singing_control",
                    "technique": technique,
                    "paired_utt_id": "",
                    "f0_mean_hz": float(payload["f0_hz"].mean()),
                    "f0_std_hz": float(payload["f0_hz"].std()),
                    "f0_voiced_pct": float(payload["voiced_mask"].mean()),
                    "energy_mean": float(payload["energy"].mean()),
                    "energy_std": float(payload["energy"].std()),
                }
                technique_utterances.append(row)
                write_feature(
                    feature_root / args.extractor / args.checkpoint_hash / args.layer / f"{utt_id}.npz",
                    payload,
                )
            technique_pairs.append(
                {
                    "pair_id": f"{speaker_id}_vibrato_{item:02d}",
                    "base_phone_ex_id": f"{normal_id}__ph0000",
                    "technique_phone_ex_id": f"{vibrato_id}__ph0000",
                    "base_utt_id": normal_id,
                    "technique_utt_id": vibrato_id,
                    "speaker_id": speaker_id,
                    "language": language,
                    "phone": "a",
                    "base_technique": "control",
                    "target_technique": "vibrato",
                    "song_id": song_id,
                    "phrase_id": phrase_id,
                    "base_phone_start_sec": 0.0,
                    "base_phone_end_sec": min(0.5, args.frames * 0.02),
                    "technique_phone_start_sec": 0.0,
                    "technique_phone_end_sec": min(0.5, args.frames * 0.02),
                }
            )

    write_table(utterances, track1_manifest)
    write_table(pairs, track1_pairs)
    write_table(technique_utterances, track2_manifest)
    write_table(technique_pairs, track2_pairs)
    metadata = {
        "extractor_name": args.extractor,
        "checkpoint_name": args.checkpoint_hash,
        "checkpoint_hash": args.checkpoint_hash,
        "git_commit": current_git_commit(),
        "layer": args.layer,
        "sample_rate": 16000,
        "hop_sec": 0.02,
        "feature_dim": args.dim,
        "dtype": "float32",
        "normalization": "none",
        "input_manifest_hash": stable_json_hash(utterances + technique_utterances),
        "command": "scripts/data_prep/make_synthetic_experiment.py",
    }
    metadata_path = feature_root / args.extractor / args.checkpoint_hash / "metadata.json"
    write_json(metadata, metadata_path)
    manifest = {
        "track1_manifest": str(track1_manifest),
        "track1_pairs": str(track1_pairs),
        "track2_manifest": str(track2_manifest),
        "track2_pairs": str(track2_pairs),
        "feature_root": str(feature_root),
        "extractor": args.extractor,
        "checkpoint_hash": args.checkpoint_hash,
        "layer": args.layer,
    }
    write_json(manifest, args.root / "synthetic_run_config.json")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
