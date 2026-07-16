#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_utils import ExperimentError, read_table, validate_pairs, validate_utterances, write_json, write_table  # noqa: E402


def split_pairs_like_mapper(pairs: list[dict[str, Any]], pair_split: str) -> list[dict[str, Any]]:
    if pair_split == "all":
        return pairs
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for row in pairs:
        by_speaker.setdefault(str(row["speaker_id"]), []).append(row)
    speakers = sorted(by_speaker)
    heldout = set(speakers[-max(1, len(speakers) // 5):])
    return [
        row
        for row in pairs
        if (str(row["speaker_id"]) in heldout) == (pair_split == "test")
    ]


def select_pairs_round_robin(pairs: list[dict[str, Any]], max_pairs: int) -> list[dict[str, Any]]:
    if max_pairs <= 0:
        return pairs
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for row in pairs:
        by_speaker.setdefault(str(row["speaker_id"]), []).append(row)
    speakers = sorted(by_speaker)
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < max_pairs:
        added = False
        for speaker in speakers:
            rows = by_speaker[speaker]
            if cursor < len(rows):
                selected.append(rows[cursor])
                added = True
                if len(selected) >= max_pairs:
                    break
        if not added:
            break
        cursor += 1
    return selected


def np_stats(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(x)) if x.size else float("nan"),
        "std": float(np.std(x)) if x.size else float("nan"),
        "min": float(np.min(x)) if x.size else float("nan"),
        "max": float(np.max(x)) if x.size else float("nan"),
    }


def pooled_vector(latents: dict[str, np.ndarray]) -> np.ndarray:
    parts = [
        latents["style"].reshape(-1),
        latents["semantic_mean"].reshape(-1),
        latents["semantic_std"].reshape(-1),
        latents["prompt_mean"].reshape(-1),
        latents["prompt_std"].reshape(-1),
        latents["mel_mean"].reshape(-1),
        latents["mel_std"].reshape(-1),
        latents["f0_summary"].reshape(-1),
        latents["audio_summary"].reshape(-1),
    ]
    return np.concatenate(parts).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract Seed-VC native prompt latents for paired speech/singing references.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--seedvc-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=80)
    parser.add_argument("--pair-split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save-sequences", action="store_true")
    args = parser.parse_args()

    try:
        if not (args.seedvc_root / "inference.py").exists():
            raise ExperimentError(f"Seed-VC inference.py not found in {args.seedvc_root}")

        utterances = read_table(args.manifest)
        validate_utterances(utterances, require_audio_exists=True)
        pairs = read_table(args.pairs)
        validate_pairs(pairs, utterances)
        utterance_by_id = {str(row["utt_id"]): row for row in utterances}
        selected_pairs = select_pairs_round_robin(split_pairs_like_mapper(pairs, args.pair_split), args.max_pairs)
        if not selected_pairs:
            raise ExperimentError(f"No pairs selected for pair_split={args.pair_split}")

        cwd = Path.cwd()
        sys.path.insert(0, str(args.seedvc_root))
        os.chdir(args.seedvc_root)
        try:
            import torch
            import torchaudio
            import librosa
            from modules.commons import recursive_munch  # noqa: F401
            import inference as seedvc_inference

            model_args = argparse.Namespace(
                f0_condition=True,
                checkpoint=None,
                config=None,
                fp16=args.fp16,
            )
            model, semantic_fn, f0_fn, _vocoder_fn, campplus_model, mel_fn, mel_fn_args = seedvc_inference.load_models(model_args)
            device = seedvc_inference.device
            sr = int(mel_fn_args["sampling_rate"])

            def extract_one(utt: dict[str, Any], condition: str, pair_id: str) -> dict[str, Any]:
                wav_path = Path(str(utt["wav_path"]))
                ref_audio = librosa.load(str(wav_path), sr=sr)[0]
                ref_audio = torch.tensor(ref_audio[: sr * 25]).unsqueeze(0).float().to(device)
                ref_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
                semantic = semantic_fn(ref_16k)
                mel = mel_fn(ref_audio.float())
                target_lengths = torch.LongTensor([mel.size(2)]).to(mel.device)
                feat = torchaudio.compliance.kaldi.fbank(
                    ref_16k,
                    num_mel_bins=80,
                    dither=0,
                    sample_frequency=16000,
                )
                feat = feat - feat.mean(dim=0, keepdim=True)
                style = campplus_model(feat.unsqueeze(0))
                f0 = f0_fn(ref_16k[0], thred=0.03)
                f0_tensor = torch.from_numpy(f0).to(device)[None]
                prompt_condition, *_ = model.length_regulator(
                    semantic,
                    ylens=target_lengths,
                    n_quantizers=3,
                    f0=f0_tensor,
                )

                semantic_np = semantic.squeeze(0).detach().cpu().numpy().astype(np.float32)
                prompt_np = prompt_condition.squeeze(0).detach().cpu().numpy().astype(np.float32)
                mel_np = mel.squeeze(0).detach().cpu().numpy().astype(np.float32)
                style_np = style.squeeze(0).detach().cpu().numpy().astype(np.float32)
                f0_np = np.asarray(f0, dtype=np.float32)
                voiced = f0_np[f0_np > 1.0]
                ref_np = ref_audio.squeeze(0).detach().cpu().numpy().astype(np.float32)
                latents = {
                    "style": style_np,
                    "semantic_mean": semantic_np.mean(axis=0),
                    "semantic_std": semantic_np.std(axis=0),
                    "prompt_mean": prompt_np.mean(axis=0),
                    "prompt_std": prompt_np.std(axis=0),
                    "mel_mean": mel_np.mean(axis=1),
                    "mel_std": mel_np.std(axis=1),
                    "f0_summary": np.asarray(
                        [
                            float(np.mean(voiced)) if voiced.size else 0.0,
                            float(np.std(voiced)) if voiced.size else 0.0,
                            float(np.min(voiced)) if voiced.size else 0.0,
                            float(np.max(voiced)) if voiced.size else 0.0,
                            float(voiced.size / max(1, f0_np.size)),
                        ],
                        dtype=np.float32,
                    ),
                    "audio_summary": np.asarray(
                        [
                            float(len(ref_np) / sr),
                            float(np.sqrt(np.mean(np.square(ref_np)))) if ref_np.size else 0.0,
                            float(np.max(np.abs(ref_np))) if ref_np.size else 0.0,
                        ],
                        dtype=np.float32,
                    ),
                }
                vector = pooled_vector(latents)
                out_path = args.out_root / "latents" / pair_id / f"{condition}.npz"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    **latents,
                    "pooled_vector": vector,
                    "utt_id": np.asarray([str(utt["utt_id"])]),
                    "condition": np.asarray([condition]),
                    "pair_id": np.asarray([pair_id]),
                    "semantic_shape": np.asarray(semantic_np.shape, dtype=np.int32),
                    "prompt_shape": np.asarray(prompt_np.shape, dtype=np.int32),
                    "mel_shape": np.asarray(mel_np.shape, dtype=np.int32),
                }
                if args.save_sequences:
                    payload.update(
                        {
                            "semantic": semantic_np,
                            "prompt_condition": prompt_np,
                            "mel": mel_np,
                            "f0": f0_np,
                        }
                    )
                np.savez(out_path, **payload)
                return {
                    "condition": condition,
                    "utt_id": str(utt["utt_id"]),
                    "speaker_id": str(utt["speaker_id"]),
                    "mode": str(utt["mode"]),
                    "technique": str(utt["technique"]),
                    "wav_path": str(wav_path),
                    "latent_npz": str(out_path),
                    "pooled_dim": int(vector.shape[0]),
                    "style_dim": int(style_np.shape[0]),
                    "semantic_frames": int(semantic_np.shape[0]),
                    "semantic_dim": int(semantic_np.shape[1]),
                    "prompt_frames": int(prompt_np.shape[0]),
                    "prompt_dim": int(prompt_np.shape[1]),
                    "mel_frames": int(mel_np.shape[1]),
                    "mel_bins": int(mel_np.shape[0]),
                    "f0_voiced_pct": float(latents["f0_summary"][4]),
                    "duration_sec": float(latents["audio_summary"][0]),
                    "rms": float(latents["audio_summary"][1]),
                }

            rows = []
            for idx, pair in enumerate(selected_pairs, 1):
                speech = utterance_by_id[str(pair["speech_utt_id"])]
                singing = utterance_by_id[str(pair["singing_utt_id"])]
                for condition, utt in (("speech_prompt", speech), ("singing_prompt", singing)):
                    rows.append(
                        {
                            "pair_id": pair["pair_id"],
                            "pair_index": idx,
                            "target_speaker_id": pair["speaker_id"],
                            "language": pair["language"],
                            "song_id": pair["song_id"],
                            "phrase_id": pair["phrase_id"],
                            "pair_technique": pair["technique"],
                            **extract_one(utt, condition, str(pair["pair_id"])),
                        }
                    )
                if idx <= 3 or idx % 10 == 0:
                    print(f"extracted {idx}/{len(selected_pairs)} pairs")
        finally:
            os.chdir(cwd)

        metrics = {
            "stage": "seedvc_native_prompt_latent_extraction",
            "seedvc_root": str(args.seedvc_root),
            "pair_split": args.pair_split,
            "pairs": len(selected_pairs),
            "conditions": len(rows),
            "fp16": args.fp16,
            "save_sequences": args.save_sequences,
            "out_root": str(args.out_root),
        }
    except (ExperimentError, OSError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    write_table(rows, args.manifest_out)
    write_json(metrics, args.metrics_out)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
