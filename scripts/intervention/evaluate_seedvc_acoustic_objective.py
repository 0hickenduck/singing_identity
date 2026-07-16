#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_utils import ExperimentError, read_table, write_json, write_table  # noqa: E402


FEATURE_NAMES = [
    "duration_sec",
    "rms_db",
    "peak",
    "f0_mean_hz",
    "f0_std_hz",
    "f0_range_hz",
    "voiced_pct",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_rolloff85_hz",
    "spectral_flatness",
    "zcr",
    "low_high_ratio_db",
    "high_band_ratio",
]


def finite(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"could not read JSON {path}: {exc}") from exc


def mean_or_nan(values: list[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return float(np.mean(clean)) if clean else float("nan")


def median_or_nan(values: list[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return float(np.median(clean)) if clean else float("nan")


def sign_counts(values: list[float]) -> dict[str, int]:
    clean = [v for v in values if math.isfinite(v) and abs(v) > 1e-12]
    return {
        "positive": int(sum(v > 0 for v in clean)),
        "negative": int(sum(v < 0 for v in clean)),
        "nonzero": int(len(clean)),
    }


def corr_pair(x: list[float], y: list[float]) -> dict[str, Any]:
    pairs = [(a, b) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 3:
        return {"n": len(pairs), "pearson": None}
    xv = np.asarray([p[0] for p in pairs], dtype=np.float64)
    yv = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if float(np.std(xv)) < 1e-12 or float(np.std(yv)) < 1e-12:
        return {"n": len(pairs), "pearson": None}
    return {"n": len(pairs), "pearson": float(np.corrcoef(xv, yv)[0, 1])}


def load_audio(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    try:
        import librosa
    except ImportError as exc:
        raise ExperimentError("librosa is required for acoustic objective evaluation") from exc
    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        raise ExperimentError(f"empty audio: {path}")
    return y, int(sr)


def band_power(power: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return np.zeros(power.shape[1], dtype=np.float64)
    return power[mask].sum(axis=0)


def audio_features(path: Path, target_sr: int) -> dict[str, Any]:
    try:
        import librosa
    except ImportError as exc:
        raise ExperimentError("librosa is required for acoustic objective evaluation") from exc

    y, sr = load_audio(path, target_sr)
    hop = 512
    frame = 2048
    duration = float(len(y) / sr)
    rms_signal = float(np.sqrt(np.mean(np.square(y))))
    peak = float(np.max(np.abs(y)))
    rms_frames = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    voiced_energy = rms_frames > max(np.percentile(rms_frames, 25), 1e-5)

    try:
        f0 = librosa.yin(
            y,
            fmin=50.0,
            fmax=min(1000.0, sr / 2.0 - 100.0),
            sr=sr,
            frame_length=frame,
            hop_length=hop,
        )
    except Exception:
        f0 = np.full_like(rms_frames, np.nan, dtype=np.float64)
    f0 = np.asarray(f0, dtype=np.float64)
    n = min(len(f0), len(voiced_energy))
    f0 = f0[:n]
    voiced_energy = voiced_energy[:n]
    voiced = np.isfinite(f0) & voiced_energy & (f0 > 1.0)
    voiced_f0 = f0[voiced]
    if len(voiced_f0) == 0:
        voiced_f0 = np.asarray([float("nan")], dtype=np.float64)

    stft = np.abs(librosa.stft(y, n_fft=frame, hop_length=hop))
    power = np.square(stft)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame)
    total_power = np.maximum(power.sum(axis=0), 1e-12)
    low = band_power(power, freqs, 80.0, 1000.0)
    high = band_power(power, freqs, 1000.0, min(6000.0, sr / 2.0))
    high4 = band_power(power, freqs, 4000.0, min(9000.0, sr / 2.0))
    low_high_ratio_db = 10.0 * np.log10(np.maximum(low.mean(), 1e-12) / np.maximum(high.mean(), 1e-12))
    high_band_ratio = float(np.mean(high4 / total_power))

    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(S=stft)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame, hop_length=hop)[0]

    return {
        "audio_wav": str(path),
        "duration_sec": duration,
        "rms_db": float(20.0 * math.log10(max(rms_signal, 1e-12))),
        "peak": peak,
        "f0_mean_hz": float(np.nanmean(voiced_f0)),
        "f0_std_hz": float(np.nanstd(voiced_f0)),
        "f0_range_hz": float(np.nanmax(voiced_f0) - np.nanmin(voiced_f0)),
        "voiced_pct": float(np.mean(voiced)) if len(voiced) else float("nan"),
        "spectral_centroid_hz": float(np.nanmean(centroid)),
        "spectral_bandwidth_hz": float(np.nanmean(bandwidth)),
        "spectral_rolloff85_hz": float(np.nanmean(rolloff)),
        "spectral_flatness": float(np.nanmean(flatness)),
        "zcr": float(np.nanmean(zcr)),
        "low_high_ratio_db": float(low_high_ratio_db),
        "high_band_ratio": high_band_ratio,
    }


def infer_refs(group: list[dict[str, Any]]) -> dict[str, str]:
    first = group[0]
    if "target_speech_wav" in first and "target_singing_wav" in first:
        return {
            "source": str(first["source_wav"]),
            "target_speech": str(first["target_speech_wav"]),
            "target_singing": str(first["target_singing_wav"]),
        }
    by_condition = {str(row["condition"]): row for row in group}
    if "target_speech_prompt" not in by_condition or "target_singing_prompt" not in by_condition:
        raise ExperimentError(f"{first.get('target_pair_id')}: cannot infer target refs")
    return {
        "source": str(first["source_wav"]),
        "target_speech": str(by_condition["target_speech_prompt"]["target_wav"]),
        "target_singing": str(by_condition["target_singing_prompt"]["target_wav"]),
    }


def feature_matrix(features: dict[str, dict[str, Any]]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    keys = sorted(features)
    matrix = np.asarray([[finite(features[key].get(name), 0.0) for name in FEATURE_NAMES] for key in keys], dtype=np.float64)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    z = (matrix - mean) / std
    return {key: z[i] for i, key in enumerate(keys)}, mean.reshape(-1), std.reshape(-1)


def z_distance(z_by_path: dict[str, np.ndarray], a: str, b: str) -> float:
    return float(np.linalg.norm(z_by_path[a] - z_by_path[b]))


def condition_rows(
    manifest_rows: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    z_by_path: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        grouped[str(row["target_pair_id"])].append(row)
    refs_by_pair = {pair_id: infer_refs(group) for pair_id, group in grouped.items()}
    out = []
    for pair_id, group in sorted(grouped.items()):
        refs = refs_by_pair[pair_id]
        for row in group:
            audio = str(row["audio_wav"])
            feat = features[audio]
            out.append(
                {
                    "target_pair_id": pair_id,
                    "condition": str(row["condition"]),
                    "target_speaker_id": row.get("target_speaker_id"),
                    "source_speaker_id": row.get("source_speaker_id"),
                    "audio_wav": audio,
                    **{name: feat[name] for name in FEATURE_NAMES},
                    "dist_to_source": z_distance(z_by_path, audio, refs["source"]),
                    "dist_to_target_speech": z_distance(z_by_path, audio, refs["target_speech"]),
                    "dist_to_target_singing": z_distance(z_by_path, audio, refs["target_singing"]),
                    "source_wav": refs["source"],
                    "target_speech_wav": refs["target_speech"],
                    "target_singing_wav": refs["target_singing"],
                }
            )
    return out, refs_by_pair


def delta_rows(rows: list[dict[str, Any]], baseline_condition: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["target_pair_id"])].append(row)
    out = []
    for pair_id, group in sorted(grouped.items()):
        by_condition = {str(row["condition"]): row for row in group}
        if baseline_condition not in by_condition:
            raise ExperimentError(f"{pair_id}: missing baseline condition {baseline_condition}")
        baseline = by_condition[baseline_condition]
        for condition, row in sorted(by_condition.items()):
            item = {
                "target_pair_id": pair_id,
                "condition": condition,
                "target_speaker_id": row.get("target_speaker_id"),
                "source_speaker_id": row.get("source_speaker_id"),
            }
            for name in FEATURE_NAMES:
                item[f"delta_{name}"] = finite(row[name]) - finite(baseline[name])
            for name in ("dist_to_source", "dist_to_target_speech", "dist_to_target_singing"):
                item[f"delta_{name}"] = finite(row[name]) - finite(baseline[name])
            out.append(item)
    return out


def summarize_by_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for condition in sorted({str(row["condition"]) for row in rows}):
        group = [row for row in rows if str(row["condition"]) == condition]
        item: dict[str, Any] = {"pairs": len(group)}
        for name in (
            "delta_dist_to_target_singing",
            "delta_dist_to_target_speech",
            "delta_dist_to_source",
            "delta_rms_db",
            "delta_f0_mean_hz",
            "delta_f0_std_hz",
            "delta_voiced_pct",
            "delta_spectral_centroid_hz",
            "delta_spectral_flatness",
            "delta_low_high_ratio_db",
            "delta_high_band_ratio",
        ):
            values = [finite(row.get(name)) for row in group]
            item[f"{name}_mean"] = mean_or_nan(values)
            item[f"{name}_median"] = median_or_nan(values)
            item[f"{name}_sign"] = sign_counts(values)
        out[condition] = item
    return out


def write_report(payload: dict[str, Any], report_out: Path) -> None:
    condition_summary = payload["condition_summary"]
    lines = [
        "# SeedVC Objective Acoustic Evaluation",
        "",
        f"Manifest: `{payload['manifest']}`",
        f"Baseline condition: `{payload['baseline_condition']}`",
        f"Pairs: {payload['pairs']}; conditions: {payload['conditions']}",
        "",
        "Negative distance delta means the condition is acoustically closer to that reference than baseline.",
        "",
        "| condition | d dist target singing | d dist target speech | d dist source | d RMS dB | d F0 mean | d F0 std | d voiced pct | d centroid | d tilt low/high | d high-band |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, item in condition_summary.items():
        lines.append(
            f"| {condition} | "
            f"{item['delta_dist_to_target_singing_mean']:.4f} | "
            f"{item['delta_dist_to_target_speech_mean']:.4f} | "
            f"{item['delta_dist_to_source_mean']:.4f} | "
            f"{item['delta_rms_db_mean']:.4f} | "
            f"{item['delta_f0_mean_hz_mean']:.2f} | "
            f"{item['delta_f0_std_hz_mean']:.2f} | "
            f"{item['delta_voiced_pct_mean']:.4f} | "
            f"{item['delta_spectral_centroid_hz_mean']:.2f} | "
            f"{item['delta_low_high_ratio_db_mean']:.4f} | "
            f"{item['delta_high_band_ratio_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation guardrail:",
            "",
            "- These are acoustic proxies, not identity labels.",
            "- F0/voicing/spectral tilt/high-band movement can support a phonation/content-delivery interpretation.",
            "- If distance-to-target-singing improves while distance-to-target-speech worsens, the effect may still be domain/phonation rather than identity.",
        ]
    )
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_table(args.manifest)
    if not rows:
        raise ExperimentError("manifest is empty")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["target_pair_id"])].append(row)
    refs_by_pair = {pair_id: infer_refs(group) for pair_id, group in grouped.items()}
    paths = set()
    for row in rows:
        paths.add(str(row["audio_wav"]))
    for refs in refs_by_pair.values():
        paths.update(refs.values())

    features = {}
    for idx, path in enumerate(sorted(paths), 1):
        if idx <= 3 or idx % 25 == 0:
            print(f"extracting acoustic features {idx}/{len(paths)}: {path}")
        features[path] = audio_features(Path(path), args.sample_rate)
    z_by_path, z_mean, z_std = feature_matrix(features)
    cond_rows, refs_by_pair = condition_rows(rows, features, z_by_path)
    deltas = delta_rows(cond_rows, args.baseline_condition)
    summary = {
        "stage": "seedvc_objective_acoustic_eval",
        "manifest": str(args.manifest),
        "baseline_condition": args.baseline_condition,
        "pairs": len(grouped),
        "conditions": len(rows),
        "unique_audio_paths": len(paths),
        "sample_rate": args.sample_rate,
        "feature_names": FEATURE_NAMES,
        "zscore_mean": {name: float(z_mean[i]) for i, name in enumerate(FEATURE_NAMES)},
        "zscore_std": {name: float(z_std[i]) for i, name in enumerate(FEATURE_NAMES)},
        "condition_counts": dict(Counter(str(row["condition"]) for row in rows)),
        "condition_summary": summarize_by_condition(deltas),
        "pair_refs": refs_by_pair,
    }
    return {
        "summary": summary,
        "audio_features": list(features.values()),
        "condition_rows": cond_rows,
        "delta_rows": deltas,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Objective acoustic proxy evaluation for SeedVC prompt/component outputs.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-condition", required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--features-out", type=Path, required=True)
    parser.add_argument("--condition-scores-out", type=Path, required=True)
    parser.add_argument("--deltas-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    try:
        payload = run(args)
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_json(payload["summary"], args.summary_out)
    write_table(payload["audio_features"], args.features_out)
    write_table(payload["condition_rows"], args.condition_scores_out)
    write_table(payload["delta_rows"], args.deltas_out)
    if args.report_out:
        write_report(payload["summary"], args.report_out)
    print(json.dumps({k: payload["summary"][k] for k in ("stage", "pairs", "conditions", "unique_audio_paths")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
