#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_utils import (  # noqa: E402
    ExperimentError,
    manifest_summary,
    validate_pairs,
    validate_utterances,
    write_json,
    write_table,
)


def read_whitespace_table(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {}
    header = lines[0].split()
    out: dict[str, dict[str, str]] = {}
    for line in lines[1:]:
        parts = line.split(maxsplit=len(header) - 1)
        if len(parts) < len(header):
            parts.extend([""] * (len(header) - len(parts)))
        row = dict(zip(header, parts))
        speaker = row.get("speaker") or row.get("singer")
        if speaker:
            out[speaker] = row
    return out


def read_transcripts(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, text = line.split(":", 1)
        out[key.strip()] = text.strip()
    return out


def wav_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ExperimentError(f"missing wav: {path}")
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
        audio = np.zeros(frames * channels, dtype=np.float32)
    if channels > 1 and audio.size:
        audio = audio.reshape(-1, channels).mean(axis=1)
    rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
    abs_audio = np.abs(audio)
    return {
        "sample_rate": sr,
        "num_samples": frames,
        "duration_sec": frames / sr if sr else 0.0,
        "rms_db": 20.0 * math.log10(max(rms, 1e-8)),
        "energy_mean": float(abs_audio.mean()) if abs_audio.size else 0.0,
        "energy_std": float(abs_audio.std()) if abs_audio.size else 0.0,
    }


def wav_path_stats(path: Path, read_wav_headers: bool = False, trust_paths: bool = False) -> dict[str, Any]:
    if not trust_paths and not path.exists():
        raise ExperimentError(f"missing wav: {path}")
    if not read_wav_headers:
        return {
            "sample_rate": 0,
            "num_samples": 0,
            "duration_sec": 0.0,
            "rms_db": float("nan"),
            "energy_mean": float("nan"),
            "energy_std": float("nan"),
        }
    with wave.open(str(path), "rb") as handle:
        sr = handle.getframerate()
        frames = handle.getnframes()
    return {
        "sample_rate": sr,
        "num_samples": frames,
        "duration_sec": frames / sr if sr else 0.0,
        "rms_db": float("nan"),
        "energy_mean": float("nan"),
        "energy_std": float("nan"),
    }


def base_utterance(
    *,
    utt_id: str,
    speaker_id: str,
    mode: str,
    wav_path: Path,
    text: str,
    song_id: str,
    phrase_id: str,
    take_id: str,
    vocal_range: str,
    gender: str,
    min_f0: str,
    max_f0: str,
    paired_utt_id: str = "",
    compute_audio_stats: bool = False,
    read_wav_headers: bool = False,
    trust_paths: bool = False,
) -> dict[str, Any]:
    stats = wav_stats(wav_path) if compute_audio_stats else wav_path_stats(wav_path, read_wav_headers, trust_paths)
    try:
        f0_min = float(min_f0)
        f0_max = float(max_f0)
        f0_mean = 0.5 * (f0_min + f0_max)
        f0_std = 0.25 * (f0_max - f0_min)
    except ValueError:
        f0_min = float("nan")
        f0_max = float("nan")
        f0_mean = float("nan")
        f0_std = float("nan")
    return {
        "utt_id": utt_id,
        "speaker_id": speaker_id,
        "language": "Japanese",
        "vocal_range": vocal_range,
        "mode": mode,
        "technique": "none",
        "song_id": song_id,
        "phrase_id": phrase_id,
        "take_id": take_id,
        "wav_path": str(wav_path.resolve()),
        "start_sec": 0.0,
        "end_sec": stats["duration_sec"],
        "duration_sec": stats["duration_sec"],
        "sample_rate": stats["sample_rate"],
        "num_samples": stats["num_samples"],
        "text": text,
        "phone_seq": "",
        "alignment_path": "",
        "paired_utt_id": paired_utt_id,
        "control_utt_id": "",
        "split_group_key": speaker_id,
        "snr_db": float("nan"),
        "rms_db": stats["rms_db"],
        "f0_mean_hz": f0_mean,
        "f0_std_hz": f0_std,
        "f0_min_hz": f0_min,
        "f0_max_hz": f0_max,
        "f0_voiced_pct": float("nan"),
        "energy_mean": stats["energy_mean"],
        "energy_std": stats["energy_std"],
        "alignment_quality_flag": f"metadata_only_gender_{gender or 'unknown'}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build JVS + JVS-MuSiC manifests for Track 1 residual prediction.")
    parser.add_argument("--speech-root", type=Path, default=Path("/work/smcintosh/data/jvs_ver1"))
    parser.add_argument("--music-root", type=Path, default=Path("/localdisk/bowen/singing_identity/data/jvs_music_ver1"))
    parser.add_argument("--utterances-out", type=Path, required=True)
    parser.add_argument("--pairs-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--speech-set", default="parallel100", choices=["parallel100", "nonpara30"])
    parser.add_argument("--max-speech-per-speaker", type=int, default=100)
    parser.add_argument("--singing-rel", default="song_common/wav/raw.wav")
    parser.add_argument("--compute-audio-stats", action="store_true")
    parser.add_argument("--read-wav-headers", action="store_true")
    parser.add_argument("--trust-paths", action="store_true")
    args = parser.parse_args()

    try:
        if not args.speech_root.exists():
            raise ExperimentError(f"speech root not found: {args.speech_root}")
        if not args.music_root.exists():
            raise ExperimentError(f"music root not found: {args.music_root}")
        f0_meta = read_whitespace_table(args.speech_root / "gender_f0range.txt")
        singer_info = read_whitespace_table(args.music_root / "singer_info.txt")
        speech_speakers = sorted(path.name for path in args.speech_root.glob("jvs[0-9][0-9][0-9]") if path.is_dir())
        music_speakers = sorted(path.name for path in args.music_root.glob("jvs[0-9][0-9][0-9]") if path.is_dir())
        common = sorted(set(speech_speakers) & set(music_speakers))
        if len(common) < 3:
            raise ExperimentError("need at least three matched JVS/JVS-MuSiC speakers")

        utterances: list[dict[str, Any]] = []
        pairs: list[dict[str, Any]] = []
        for speaker in common:
            meta = f0_meta.get(speaker, {})
            music_meta = singer_info.get(speaker, {})
            gender = meta.get("Male_or_Female", music_meta.get("gender", ""))
            vocal_range = music_meta.get("key_group", "")
            min_f0 = meta.get("minf0[Hz]", "")
            max_f0 = meta.get("maxf0[Hz]", "")
            transcripts = read_transcripts(args.speech_root / speaker / args.speech_set / "transcripts_utf8.txt")
            speech_wavs = sorted((args.speech_root / speaker / args.speech_set / "wav24kHz16bit").glob("*.wav"))
            if args.max_speech_per_speaker > 0:
                speech_wavs = speech_wavs[: args.max_speech_per_speaker]
            singing_path = args.music_root / speaker / args.singing_rel
            singing_utt_id = f"{speaker}__singing__{args.singing_rel.replace('/', '_').replace('.wav', '')}"
            singing_row = base_utterance(
                utt_id=singing_utt_id,
                speaker_id=speaker,
                mode="singing",
                wav_path=singing_path,
                text="katatsumuri" if "song_common" in args.singing_rel else music_meta.get("unique_song_name", ""),
                song_id="song_common" if "song_common" in args.singing_rel else "song_unique",
                phrase_id=args.singing_rel,
                take_id="raw",
                vocal_range=vocal_range,
                gender=gender,
                min_f0=min_f0,
                max_f0=max_f0,
                compute_audio_stats=args.compute_audio_stats,
                read_wav_headers=args.read_wav_headers,
                trust_paths=args.trust_paths,
            )
            utterances.append(singing_row)
            if not speech_wavs:
                raise ExperimentError(f"no speech wavs for {speaker} {args.speech_set}")
            first_speech_utt_id = ""
            for wav in speech_wavs:
                stem = wav.stem
                utt_id = f"{speaker}__speech__{args.speech_set}__{stem}"
                if not first_speech_utt_id:
                    first_speech_utt_id = utt_id
                utterances.append(
                    base_utterance(
                        utt_id=utt_id,
                        speaker_id=speaker,
                        mode="speech",
                        wav_path=wav,
                        text=transcripts.get(stem, ""),
                        song_id=args.speech_set,
                        phrase_id=stem,
                        take_id=stem,
                        vocal_range=vocal_range,
                        gender=gender,
                        min_f0=min_f0,
                        max_f0=max_f0,
                        paired_utt_id=singing_utt_id,
                        compute_audio_stats=args.compute_audio_stats,
                        read_wav_headers=args.read_wav_headers,
                        trust_paths=args.trust_paths,
                    )
                )
            singing_row["paired_utt_id"] = first_speech_utt_id
            pairs.append(
                {
                    "pair_id": f"{speaker}__speech_centroid__singing_{args.singing_rel.replace('/', '_').replace('.wav', '')}",
                    "speech_utt_id": first_speech_utt_id,
                    "singing_utt_id": singing_utt_id,
                    "speaker_id": speaker,
                    "language": "Japanese",
                    "song_id": "song_common" if "song_common" in args.singing_rel else "song_unique",
                    "phrase_id": args.singing_rel,
                    "technique": "none",
                    "pair_type": "same_singer_speech_centroid_to_singing",
                    "same_text_flag": "false",
                    "same_song_flag": "false",
                }
            )

        utterance_summary = validate_utterances(utterances, require_audio_exists=not args.trust_paths)
        pair_summary = validate_pairs(pairs, utterances)
        report = {
            "speech_root": str(args.speech_root),
            "music_root": str(args.music_root),
            "speech_set": args.speech_set,
            "max_speech_per_speaker": args.max_speech_per_speaker,
            "singing_rel": args.singing_rel,
            "compute_audio_stats": bool(args.compute_audio_stats),
            "read_wav_headers": bool(args.read_wav_headers),
            "trust_paths": bool(args.trust_paths),
            "speech_speakers": len(speech_speakers),
            "music_speakers": len(music_speakers),
            "matched_speakers": len(common),
            "utterance_summary": utterance_summary,
            "pair_summary": pair_summary,
            "manifest_summary": manifest_summary(utterances),
        }
    except (ExperimentError, OSError, wave.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    write_table(utterances, args.utterances_out)
    write_table(pairs, args.pairs_out)
    write_json(report, args.report_out)
    print(f"utterances: {len(utterances)}")
    print(f"pairs: {len(pairs)}")
    print(f"matched speakers: {len(common)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
