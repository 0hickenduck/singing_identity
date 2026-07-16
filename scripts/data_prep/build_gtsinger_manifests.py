#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_utils import (  # noqa: E402
    PHONE_EXAMPLE_COLUMNS,
    ExperimentError,
    manifest_summary,
    stable_json_hash,
    validate_pairs,
    validate_utterances,
    write_json,
    write_table,
)


TECHNIQUE_GROUPS = {
    "Control_Group": "control",
    "Mixed_Voice_Group": "mixed_voice",
    "Falsetto_Group": "falsetto",
    "Breathy_Group": "breathy",
    "Pharyngeal_Group": "pharyngeal",
    "Vibrato_Group": "vibrato",
    "Glissando_Group": "glissando",
    "Paired_Speech_Group": "none",
}

TECHNIQUE_COLUMNS = {
    "mixed_voice": "mix_tech",
    "falsetto": "falsetto_tech",
    "breathy": "breathy_tech",
    "pharyngeal": "pharyngeal_tech",
    "vibrato": "vibrato_tech",
    "glissando": "glissando_tech",
}


def parse_item_name(item_name: str) -> dict[str, str]:
    parts = item_name.split("#")
    if len(parts) < 6:
        return {
            "language": "",
            "speaker_id": "",
            "technique_family": "",
            "song_id": "",
            "group": "",
            "take_id": "",
        }
    return {
        "language": parts[0],
        "speaker_id": parts[1],
        "technique_family": parts[2],
        "song_id": parts[3],
        "group": parts[4],
        "take_id": parts[5],
    }


def wav_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ExperimentError(f"missing wav: {path}")
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    if sample_width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        signed = (
            bytes_[:, 0].astype(np.int32)
            | (bytes_[:, 1].astype(np.int32) << 8)
            | (bytes_[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed | ~0xFFFFFF, signed)
        audio = signed.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        audio = np.zeros(frames * channels, dtype=np.float32)
    if channels > 1 and audio.size:
        audio = audio.reshape(-1, channels).mean(axis=1)
    rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
    rms_db = 20.0 * math.log10(max(rms, 1e-8))
    energy = np.abs(audio)
    return {
        "sample_rate": sample_rate,
        "num_samples": frames,
        "duration_sec": frames / sample_rate if sample_rate else 0.0,
        "rms_db": rms_db,
        "energy_mean": float(energy.mean()) if energy.size else 0.0,
        "energy_std": float(energy.std()) if energy.size else 0.0,
    }


def metadata_duration(row: dict[str, Any]) -> float:
    durations = row.get("ph_durs", [])
    try:
        return float(sum(float(value) for value in durations))
    except (TypeError, ValueError):
        return 0.0


def fast_wav_stats(row: dict[str, Any]) -> dict[str, Any]:
    duration = metadata_duration(row)
    return {
        "sample_rate": 0,
        "num_samples": 0,
        "duration_sec": duration,
        "rms_db": -80.0,
        "energy_mean": 0.0,
        "energy_std": 0.0,
    }


def pitch_stats(row: dict[str, Any]) -> dict[str, float]:
    pitches = np.asarray([float(p) for p in row.get("ep_pitches", []) if float(p) > 0.0], dtype=np.float64)
    if pitches.size == 0:
        return {
            "f0_mean_hz": float("nan"),
            "f0_std_hz": float("nan"),
            "f0_min_hz": float("nan"),
            "f0_max_hz": float("nan"),
            "f0_voiced_pct": 0.0,
        }
    hz = 440.0 * (2.0 ** ((pitches - 69.0) / 12.0))
    total = max(1, len(row.get("ep_pitches", [])))
    return {
        "f0_mean_hz": float(hz.mean()),
        "f0_std_hz": float(hz.std()),
        "f0_min_hz": float(hz.min()),
        "f0_max_hz": float(hz.max()),
        "f0_voiced_pct": float(len(pitches) / total),
    }


def text_from_row(row: dict[str, Any]) -> str:
    return " ".join(str(token) for token in row.get("txt", []) if str(token) != "<SP>")


def technique_for_group(group: str) -> str:
    return TECHNIQUE_GROUPS.get(group, group.replace("_Group", "").lower())


def mode_for_technique(technique: str) -> str:
    if technique == "none":
        return "speech"
    if technique == "control":
        return "singing_control"
    return "singing_technique"


def utterance_id(parts: dict[str, str], domain: str) -> str:
    raw = "__".join(
        [
            parts["language"],
            parts["speaker_id"],
            parts["technique_family"],
            parts["song_id"],
            parts["group"] if domain == "singing" else "Paired_Speech_Group",
            parts["take_id"],
        ]
    )
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)


def make_utterance(
    root: Path,
    row: dict[str, Any],
    wav_rel: str,
    utt_id: str,
    paired_utt_id: str,
    mode: str,
    technique: str,
    parts: dict[str, str],
    fast_no_wav_stats: bool = False,
) -> dict[str, Any]:
    wav_path = (root / wav_rel).resolve()
    stats = fast_wav_stats(row) if fast_no_wav_stats else wav_stats(wav_path)
    f0 = pitch_stats(row)
    duration = stats["duration_sec"]
    return {
        "utt_id": utt_id,
        "speaker_id": str(row.get("singer") or parts["speaker_id"]),
        "language": str(row.get("language") or parts["language"]),
        "vocal_range": str(row.get("range", "")),
        "mode": mode,
        "technique": technique,
        "song_id": parts["song_id"],
        "phrase_id": str(row.get("item_name", "")),
        "take_id": parts["take_id"],
        "wav_path": str(wav_path),
        "start_sec": 0.0,
        "end_sec": duration,
        "duration_sec": duration,
        "sample_rate": stats["sample_rate"],
        "num_samples": stats["num_samples"],
        "text": text_from_row(row),
        "phone_seq": " ".join(str(phone) for phone in row.get("ph", [])),
        "alignment_path": "",
        "paired_utt_id": paired_utt_id,
        "control_utt_id": "",
        "split_group_key": f"{parts['speaker_id']}::{parts['song_id']}::{parts['take_id']}",
        "snr_db": float("nan"),
        "rms_db": stats["rms_db"],
        "f0_mean_hz": f0["f0_mean_hz"],
        "f0_std_hz": f0["f0_std_hz"],
        "f0_min_hz": f0["f0_min_hz"],
        "f0_max_hz": f0["f0_max_hz"],
        "f0_voiced_pct": f0["f0_voiced_pct"],
        "energy_mean": stats["energy_mean"],
        "energy_std": stats["energy_std"],
        "alignment_quality_flag": "ok" if row.get("ph") and row.get("ph_durs") else "missing",
    }


def phone_examples(row: dict[str, Any], utt_id: str, base: dict[str, Any]) -> list[dict[str, Any]]:
    phones = [str(phone) for phone in row.get("ph", [])]
    durations = [float(dur) for dur in row.get("ph_durs", [])]
    if len(phones) != len(durations):
        return []
    out = []
    cursor = 0.0
    for idx, (phone, duration) in enumerate(zip(phones, durations)):
        start = cursor
        end = cursor + duration
        cursor = end
        if phone == "<SP>" or duration <= 0:
            continue
        out.append(
            {
                "phone_ex_id": f"{utt_id}__ph{idx:04d}",
                "utt_id": utt_id,
                "speaker_id": base["speaker_id"],
                "language": base["language"],
                "mode": base["mode"],
                "technique": base["technique"],
                "song_id": base["song_id"],
                "phrase_id": base["phrase_id"],
                "phone": phone,
                "phone_start_sec": start,
                "phone_end_sec": end,
                "phone_duration_sec": duration,
            }
        )
    return out


def has_any_technique(row: dict[str, Any], technique: str) -> bool:
    col = TECHNIQUE_COLUMNS.get(technique)
    values = row.get(col, []) if col else []
    return any(int(value) == 1 for value in values)


def select_rows(rows: list[dict[str, Any]], max_rows_per_singer: int) -> list[dict[str, Any]]:
    by_singer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("speech_fn") and row.get("wav_fn") and row.get("singer"):
            by_singer[str(row["singer"])].append(row)
    selected = []
    for _, singer_rows in sorted(by_singer.items()):
        selected.extend(singer_rows[:max_rows_per_singer])
    return selected


def build(args: argparse.Namespace) -> dict[str, Any]:
    utterances_by_id: dict[str, dict[str, Any]] = {}
    pairs_by_id: dict[str, dict[str, Any]] = {}
    phone_rows: list[dict[str, Any]] = []
    metadata_counts = Counter()
    skipped = Counter()

    for metadata_path in sorted((args.root / "processed").glob("*/metadata.json")):
        language_rows = json.loads(metadata_path.read_text(encoding="utf-8"))
        if args.languages and metadata_path.parent.name not in args.languages:
            continue
        if not isinstance(language_rows, list):
            raise ExperimentError(f"{metadata_path} must contain a list")
        rows = select_rows(language_rows, args.max_rows_per_singer)
        metadata_counts[metadata_path.parent.name] += len(rows)
        for row in rows:
            parts = parse_item_name(str(row.get("item_name", "")))
            if not parts["speaker_id"]:
                continue
            technique = technique_for_group(parts["group"])
            singing_id = utterance_id(parts, "singing")
            speech_id = utterance_id(parts, "speech")
            singing_path = args.root / str(row["wav_fn"])
            speech_path = args.root / str(row["speech_fn"])
            if not singing_path.exists() or not speech_path.exists():
                skipped["missing_wav"] += 1
                continue
            singing = make_utterance(
                args.root,
                row,
                str(row["wav_fn"]),
                singing_id,
                speech_id,
                mode_for_technique(technique),
                technique,
                parts,
                args.fast_no_wav_stats,
            )
            speech = make_utterance(
                args.root,
                row,
                str(row["speech_fn"]),
                speech_id,
                singing_id,
                "speech",
                "none",
                parts,
                args.fast_no_wav_stats,
            )
            utterances_by_id.setdefault(speech_id, speech)
            utterances_by_id.setdefault(singing_id, singing)
            pair_id = f"{speech_id}__{singing_id}"
            pairs_by_id[pair_id] = {
                "pair_id": pair_id,
                "speech_utt_id": speech_id,
                "singing_utt_id": singing_id,
                "speaker_id": speech["speaker_id"],
                "language": speech["language"],
                "song_id": speech["song_id"],
                "phrase_id": speech["phrase_id"],
                "technique": technique,
                "pair_type": "same_singer_same_phrase",
                "same_text_flag": "true",
                "same_song_flag": "true",
            }
            phone_rows.extend(phone_examples(row, singing_id, singing))

    utterances = list(utterances_by_id.values())
    pairs = list(pairs_by_id.values())
    validate_utterances(utterances, require_audio_exists=True)
    validate_pairs(pairs, utterances)

    technique_pairs = build_technique_pairs(phone_rows, args.max_phone_pairs_per_group)

    write_table(utterances, args.utterances_out)
    write_table(pairs, args.pairs_out)
    write_table(phone_rows, args.phone_examples_out)
    write_table(technique_pairs, args.technique_pairs_out)
    report = {
        "root": str(args.root.resolve()),
        "metadata_rows_selected": dict(metadata_counts),
        "skipped": dict(skipped),
        "fast_no_wav_stats": bool(args.fast_no_wav_stats),
        "utterances": manifest_summary(utterances),
        "pairs": {
            "num_pairs": len(pairs),
            "techniques": dict(Counter(row["technique"] for row in pairs)),
            "hash": stable_json_hash(pairs),
        },
        "phone_examples": {
            "num_examples": len(phone_rows),
            "phones": len({row["phone"] for row in phone_rows}),
            "techniques": dict(Counter(row["technique"] for row in phone_rows)),
            "hash": stable_json_hash(phone_rows),
        },
        "technique_pairs": {
            "num_pairs": len(technique_pairs),
            "target_techniques": dict(Counter(row["target_technique"] for row in technique_pairs)),
            "hash": stable_json_hash(technique_pairs),
        },
    }
    write_json(report, args.report_out)
    return report


def build_technique_pairs(phone_rows: list[dict[str, Any]], max_pairs_per_group: int) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in phone_rows:
        if row["technique"] == "none":
            continue
        key = (row["speaker_id"], row["language"], row["phone"])
        by_key[key][row["technique"]].append(row)

    out = []
    counts: Counter[tuple[str, str]] = Counter()
    for (speaker_id, language, phone), tech_map in sorted(by_key.items()):
        controls = tech_map.get("control", [])
        if not controls:
            continue
        for target_technique, targets in sorted(tech_map.items()):
            if target_technique in {"control", "none"}:
                continue
            limit_key = (phone, target_technique)
            for control, target in zip(controls, targets):
                if counts[limit_key] >= max_pairs_per_group:
                    break
                pair_id = f"{control['phone_ex_id']}__{target['phone_ex_id']}"
                out.append(
                    {
                        "pair_id": pair_id,
                        "base_phone_ex_id": control["phone_ex_id"],
                        "technique_phone_ex_id": target["phone_ex_id"],
                        "base_utt_id": control["utt_id"],
                        "technique_utt_id": target["utt_id"],
                        "speaker_id": speaker_id,
                        "language": language,
                        "phone": phone,
                        "base_technique": "control",
                        "target_technique": target_technique,
                        "song_id": target["song_id"],
                        "phrase_id": target["phrase_id"],
                        "base_phone_start_sec": control["phone_start_sec"],
                        "base_phone_end_sec": control["phone_end_sec"],
                        "technique_phone_start_sec": target["phone_start_sec"],
                        "technique_phone_end_sec": target["phone_end_sec"],
                    }
                )
                counts[limit_key] += 1
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Track 1/2 manifests from a local GTSinger processed tree.")
    parser.add_argument("--root", type=Path, required=True, help="GTSinger local root containing processed/*/metadata.json.")
    parser.add_argument("--utterances-out", type=Path, required=True)
    parser.add_argument("--pairs-out", type=Path, required=True)
    parser.add_argument("--phone-examples-out", type=Path, required=True)
    parser.add_argument("--technique-pairs-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--languages", nargs="*", help="Optional language folder names, e.g. English Japanese.")
    parser.add_argument("--max-rows-per-singer", type=int, default=12)
    parser.add_argument("--max-phone-pairs-per-group", type=int, default=500)
    parser.add_argument(
        "--fast-no-wav-stats",
        action="store_true",
        help="Do not open wav payloads while building manifests; use metadata durations and placeholder energy stats.",
    )
    args = parser.parse_args()
    try:
        report = build(args)
    except (ExperimentError, OSError, wave.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
