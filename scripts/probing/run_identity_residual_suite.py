#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_utils import ExperimentError, binary_auc, current_git_commit, read_table  # noqa: E402


MODEL_SPECS = {
    "acoustic": ("acoustic_baseline", "local_wave_v1", "frame25ms_hop20ms"),
    "ecapa": ("ecapa_tdnn", "speechbrain_spkrec_ecapa_voxceleb", "embedding"),
    "hubert_l3": ("hubert_base", "facebook_hubert_base_ls960", "3"),
    "hubert_l6": ("hubert_base", "facebook_hubert_base_ls960", "6"),
    "hubert_l9": ("hubert_base", "facebook_hubert_base_ls960", "9"),
    "hubert_l12": ("hubert_base", "facebook_hubert_base_ls960", "12"),
    "mert_l6": ("mert_v1_95m", "m_a_p_mert_v1_95m", "6"),
    "mert_l9": ("mert_v1_95m", "m_a_p_mert_v1_95m", "9"),
    "mert_l12": ("mert_v1_95m", "m_a_p_mert_v1_95m", "12"),
    "wavlm_l3": ("wavlm_base_plus", "microsoft_wavlm_base_plus", "3"),
    "wavlm_l6": ("wavlm_base_plus", "microsoft_wavlm_base_plus", "6"),
    "wavlm_l9": ("wavlm_base_plus", "microsoft_wavlm_base_plus", "9"),
    "wavlm_l12": ("wavlm_base_plus", "microsoft_wavlm_base_plus", "12"),
    "mert_l3": ("mert_v1_95m", "m_a_p_mert_v1_95m", "3"),
}

BASE_NUISANCE_COLUMNS = [
    "logf0_mean",
    "logf0_std",
    "logf0_p05",
    "logf0_p50",
    "logf0_p95",
    "logf0_range",
    "rms_mean",
    "rms_std",
    "duration_s",
    "voiced_rate",
    "energy_mean",
    "energy_std",
]

NUISANCE_GROUPS = {
    "f0": [
        "logf0_mean",
        "logf0_std",
        "logf0_p05",
        "logf0_p50",
        "logf0_p95",
        "logf0_range",
    ],
    "energy": ["rms_mean", "rms_std", "energy_mean", "energy_std"],
    "duration": ["duration_s"],
    "voicing": ["voiced_rate"],
}


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def write_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_table(path)


def feature_path(feature_root: Path, extractor: str, checkpoint: str, layer: str, utt_id: str) -> Path:
    return feature_root / extractor / checkpoint / str(layer) / f"{utt_id}.npz"


def canonical_mode(row: dict[str, Any]) -> str:
    mode = str(row.get("mode", ""))
    if mode == "speech":
        return "speech"
    if mode == "singing" or mode.startswith("singing_"):
        return "singing"
    return mode


def feature_vector_from_npz(path: Path, half: str | None = None) -> np.ndarray:
    if not path.exists():
        raise ExperimentError(f"missing feature file: {path}")
    with np.load(path) as data:
        x = np.asarray(data["x"], dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ExperimentError(f"bad feature shape in {path}: {x.shape}")
    if half == "a" and x.shape[0] > 1:
        x = x[: max(1, x.shape[0] // 2)]
    elif half == "b" and x.shape[0] > 1:
        x = x[max(1, x.shape[0] // 2) :]
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    return np.concatenate([mean, std]).astype(np.float64)


def load_feature_cache(
    rows: list[dict[str, Any]],
    feature_root: Path,
    spec_name: str,
    cache_dir: Path,
    max_speech_per_speaker: int | None,
    duration_balanced: bool,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, str]]:
    extractor, checkpoint, layer = MODEL_SPECS[spec_name]
    selected: list[dict[str, Any]] = []
    speech_seen: Counter[str] = Counter()
    singing_duration_by_speaker: dict[str, float] = defaultdict(float)
    speech_duration_seen: Counter[str] = Counter()
    if duration_balanced:
        for row in rows:
            if canonical_mode(row) == "singing":
                singing_duration_by_speaker[str(row["speaker_id"])] += max(0.0, finite_float(row.get("duration_sec"), 0.0))
    for row in rows:
        mode = canonical_mode(row)
        if mode not in {"speech", "singing"}:
            continue
        row = {**row, "original_mode": str(row.get("mode", "")), "mode": mode}
        speaker = str(row["speaker_id"])
        if mode == "speech" and max_speech_per_speaker is not None:
            if speech_seen[speaker] >= max_speech_per_speaker:
                continue
            speech_seen[speaker] += 1
        if mode == "speech" and duration_balanced:
            target = singing_duration_by_speaker.get(speaker, 0.0)
            if target > 0 and speech_duration_seen[speaker] >= target:
                continue
            speech_duration_seen[speaker] += max(0.0, finite_float(row.get("duration_sec"), 0.0))
        selected.append(row)
    cache_dir.mkdir(parents=True, exist_ok=True)
    balance_label = "duration_balanced" if duration_balanced else "duration_unbalanced"
    cache_key = f"{spec_name}_speech{max_speech_per_speaker or 'all'}_{balance_label}_{len(selected)}"
    x_path = cache_dir / f"{cache_key}.npy"
    meta_path = cache_dir / f"{cache_key}.json"
    if x_path.exists() and meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))["rows"], np.load(x_path), {
            "extractor": extractor,
            "checkpoint_hash": checkpoint,
            "layer": layer,
        }
    vectors = []
    kept = []
    for row in selected:
        path = feature_path(feature_root, extractor, checkpoint, layer, str(row["utt_id"]))
        vectors.append(feature_vector_from_npz(path))
        kept.append(row)
    x = np.vstack(vectors)
    np.save(x_path, x)
    meta_path.write_text(json.dumps({"rows": kept}, allow_nan=True), encoding="utf-8")
    return kept, x, {"extractor": extractor, "checkpoint_hash": checkpoint, "layer": layer}


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def nuisance_table(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    mat = []
    for row in rows:
        f0_mean = finite_float(row.get("f0_mean_hz"))
        f0_std = finite_float(row.get("f0_std_hz"))
        f0_min = finite_float(row.get("f0_min_hz"))
        f0_max = finite_float(row.get("f0_max_hz"))
        rms_db = finite_float(row.get("rms_db"))
        energy_mean = finite_float(row.get("energy_mean"))
        energy_std = finite_float(row.get("energy_std"))
        vals = [
            math.log(max(f0_mean, 1e-6)) if math.isfinite(f0_mean) else float("nan"),
            math.log1p(max(f0_std, 0.0)) if math.isfinite(f0_std) else float("nan"),
            math.log(max(f0_min, 1e-6)) if math.isfinite(f0_min) else float("nan"),
            math.log(max(f0_mean, 1e-6)) if math.isfinite(f0_mean) else float("nan"),
            math.log(max(f0_max, 1e-6)) if math.isfinite(f0_max) else float("nan"),
            math.log(max(f0_max - f0_min, 1e-6)) if math.isfinite(f0_max) and math.isfinite(f0_min) else float("nan"),
            rms_db,
            float("nan"),
            finite_float(row.get("duration_sec")),
            finite_float(row.get("f0_voiced_pct")),
            energy_mean,
            energy_std,
        ]
        mat.append(vals)
    return np.asarray(mat, dtype=np.float64), list(BASE_NUISANCE_COLUMNS)


def mode_dummy(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[1.0 if canonical_mode(row) == "singing" else 0.0] for row in rows], dtype=np.float64)


def impute_and_scale_fit(n: np.ndarray, train_idx: np.ndarray) -> dict[str, np.ndarray]:
    train = n[train_idx]
    means = []
    for col_idx in range(train.shape[1]):
        col = train[:, col_idx]
        finite = col[np.isfinite(col)]
        means.append(float(finite.mean()) if len(finite) else 0.0)
    means = np.asarray(means, dtype=np.float64)
    filled = np.where(np.isfinite(n), n, means.reshape(1, -1))
    std = filled[train_idx].std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": means.reshape(1, -1), "std": std.reshape(1, -1)}


def impute_and_scale_apply(n: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    filled = np.where(np.isfinite(n), n, scaler["mean"])
    return (filled - scaler["mean"]) / scaler["std"]


def ridge_fit_predictors(x_train: np.ndarray, n_train_z: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    z_aug = np.column_stack([np.ones(len(n_train_z)), n_train_z])
    reg = np.eye(z_aug.shape[1]) * alpha
    reg[0, 0] = 0.0
    weights = np.linalg.pinv(z_aug.T @ z_aug + reg) @ z_aug.T @ x_train
    intercept = weights[:1]
    coef = weights[1:]
    return intercept, coef


def apply_residualization(
    x: np.ndarray,
    n: np.ndarray,
    train_idx: np.ndarray,
    formula: str,
    alpha: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if n.shape[1] == 0:
        mean = x[train_idx].mean(axis=0, keepdims=True)
        if formula != "classic":
            return x.copy(), {"variance_removed": 0.0, "rank": 0}
        out = x - mean
        return out, {"variance_removed": variance_removed(x, out), "rank": 0}
    scaler = impute_and_scale_fit(n, train_idx)
    nz = impute_and_scale_apply(n, scaler)
    intercept, coef = ridge_fit_predictors(x[train_idx], nz[train_idx], alpha=alpha)
    if formula == "classic":
        out = x - (intercept + nz @ coef)
    elif formula == "mean_preserving":
        out = x - nz @ coef
    else:
        raise ExperimentError(f"unknown formula: {formula}")
    return out, {
        "variance_removed": variance_removed(x, out),
        "rank": int(np.linalg.matrix_rank(coef)),
        "coef_shape": list(coef.shape),
    }


def variance_removed(before: np.ndarray, after: np.ndarray) -> float:
    denom = float(np.var(before))
    if denom < 1e-12:
        return 0.0
    return float(1.0 - np.var(after) / denom)


def speaker_split(speakers: list[str], seed: int, train_frac: float, dev_frac: float) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    speakers = list(sorted(speakers))
    rng.shuffle(speakers)
    n = len(speakers)
    n_train = int(round(n * train_frac))
    n_dev = int(round(n * dev_frac))
    out = {}
    for i, speaker in enumerate(speakers):
        if i < n_train:
            out[speaker] = "train"
        elif i < n_train + n_dev:
            out[speaker] = "dev"
        else:
            out[speaker] = "test"
    return out


def make_speaker_split(speakers: list[str], seed: int, protocol: str) -> dict[str, str]:
    if protocol == "jvs_60_20_20":
        return speaker_split(speakers, seed, 0.6, 0.2)
    if protocol == "gtsinger_10_0_10":
        if len(speakers) < 20:
            raise ExperimentError(f"gtsinger_10_0_10 requires at least 20 speakers, got {len(speakers)}")
        rng = np.random.default_rng(seed)
        shuffled = list(sorted(speakers))
        rng.shuffle(shuffled)
        out = {}
        for i, speaker in enumerate(shuffled):
            out[speaker] = "train" if i < 10 else "test"
        return out
    raise ExperimentError(f"unknown split protocol: {protocol}")


def split_name(protocol: str) -> str:
    if protocol == "jvs_60_20_20":
        return "60_20_20_speaker_disjoint"
    if protocol == "gtsinger_10_0_10":
        return "10_0_10_speaker_disjoint"
    return protocol


def centroids(rows: list[dict[str, Any]], x: np.ndarray, speakers: set[str] | None = None) -> dict[tuple[str, str], np.ndarray]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        speaker = str(row["speaker_id"])
        if speakers is not None and speaker not in speakers:
            continue
        groups[(speaker, canonical_mode(row))].append(i)
    return {key: x[idx].mean(axis=0) for key, idx in groups.items()}


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def retrieval(query: np.ndarray, gallery: np.ndarray, query_labels: list[str], gallery_labels: list[str], k: int = 1) -> dict[str, Any]:
    sims = l2_normalize(query) @ l2_normalize(gallery).T
    ranks = []
    hit1 = []
    hit5 = []
    hits = 0
    same = []
    impostor = []
    for i, label in enumerate(query_labels):
        order = np.argsort(-sims[i])
        rank = int(np.where(np.asarray(gallery_labels, dtype=object)[order] == label)[0][0]) + 1
        ranks.append(rank)
        hit1.append(int(rank <= 1))
        hit5.append(int(rank <= min(5, len(gallery_labels))))
        hits += int(rank <= k)
        same.append(float(sims[i, gallery_labels.index(label)]))
        impostor_vals = [float(sims[i, j]) for j, g in enumerate(gallery_labels) if g != label]
        impostor.append(float(np.mean(impostor_vals)) if impostor_vals else float("nan"))
    rr = [1.0 / r for r in ranks]
    margins = (np.asarray(same) - np.asarray(impostor)).astype(float)
    return {
        "num": hits,
        "den": len(query_labels),
        "r": hits / max(1, len(query_labels)),
        "r5": float(np.mean(hit5)) if hit5 else float("nan"),
        "mrr": float(np.mean(rr)) if rr else float("nan"),
        "median_rank": float(np.median(ranks)) if ranks else float("nan"),
        "same_cos": float(np.mean(same)) if same else float("nan"),
        "impostor_cos": float(np.mean(impostor)) if impostor else float("nan"),
        "margin": float(np.nanmean(margins)) if same else float("nan"),
        "labels": list(query_labels),
        "ranks": ranks,
        "hit1": hit1,
        "hit5": hit5,
        "rr": rr,
        "margins": margins.tolist(),
    }


def evaluate_retrieval(rows: list[dict[str, Any]], x: np.ndarray, test_speakers: list[str]) -> dict[str, dict[str, Any]]:
    cents = centroids(rows, x, set(test_speakers))
    labels = [s for s in sorted(test_speakers) if (s, "speech") in cents and (s, "singing") in cents]
    speech = np.vstack([cents[(s, "speech")] for s in labels])
    singing = np.vstack([cents[(s, "singing")] for s in labels])
    return {
        "S->G": retrieval(speech, singing, labels, labels),
        "G->S": retrieval(singing, speech, labels, labels),
    }


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if samples <= 0:
        return float("nan"), float("nan")
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_retrieval(metrics: dict[str, Any], seed: int, samples: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    r1_low, r1_high = bootstrap_ci(np.asarray(metrics["hit1"], dtype=np.float64), rng, samples)
    r5_low, r5_high = bootstrap_ci(np.asarray(metrics["hit5"], dtype=np.float64), rng, samples)
    mrr_low, mrr_high = bootstrap_ci(np.asarray(metrics["rr"], dtype=np.float64), rng, samples)
    margin_low, margin_high = bootstrap_ci(np.asarray(metrics["margins"], dtype=np.float64), rng, samples)
    return {
        "R1_boot_CI_low": r1_low,
        "R1_boot_CI_high": r1_high,
        "R5_boot_CI_low": r5_low,
        "R5_boot_CI_high": r5_high,
        "MRR_boot_CI_low": mrr_low,
        "MRR_boot_CI_high": mrr_high,
        "margin_boot_CI_low": margin_low,
        "margin_boot_CI_high": margin_high,
    }


def mode_auc(rows: list[dict[str, Any]], x: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> float:
    y = np.asarray([1 if canonical_mode(row) == "singing" else 0 for row in rows], dtype=int)
    if len(set(y[train_idx])) < 2 or len(set(y[test_idx])) < 2:
        return float("nan")
    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = np.where(x[train_idx].std(axis=0, keepdims=True) < 1e-8, 1.0, x[train_idx].std(axis=0, keepdims=True))
    train_z = (x[train_idx] - mean) / std
    test_z = (x[test_idx] - mean) / std
    direction = train_z[y[train_idx] == 1].mean(axis=0) - train_z[y[train_idx] == 0].mean(axis=0)
    score = test_z @ direction
    return float(binary_auc(y[test_idx], score))


def run_manifest_audit(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.manifest)
    out_dir = args.results_dir / "manifest_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    audit_rows = []
    duplicates = Counter(str(row["utt_id"]) for row in rows)
    for row in rows:
        norm = {
            "dataset": args.dataset,
            "utt_id": row.get("utt_id", ""),
            "speaker_id": row.get("speaker_id", ""),
            "mode": row.get("mode", ""),
            "wav_path": row.get("wav_path", ""),
            "duration_s": row.get("duration_sec", ""),
            "language": row.get("language", ""),
            "gender": row.get("alignment_quality_flag", ""),
            "song_id": row.get("song_id", ""),
            "text_id": row.get("phrase_id", ""),
            "technique": row.get("technique", ""),
            "pair_id": row.get("paired_utt_id", ""),
            "split_group": row.get("split_group_key", ""),
            "has_acoustic_features": all(k in row for k in ["f0_mean_hz", "energy_mean", "duration_sec", "rms_db"]),
            "has_wavlm": feature_path(args.feature_root, *MODEL_SPECS["wavlm_l12"], str(row["utt_id"])).exists(),
            "has_hubert": feature_path(args.feature_root, "hubert_base", "facebook_hubert_base_ls960", "12", str(row["utt_id"])).exists(),
            "has_mert": feature_path(args.feature_root, *MODEL_SPECS["mert_l3"], str(row["utt_id"])).exists(),
            "has_ecapa": feature_path(args.feature_root, "ecapa_tdnn", "speechbrain_spkrec_ecapa_voxceleb", "embedding", str(row["utt_id"])).exists(),
            "has_contentvec": False,
            "notes": "duplicate_utt_id" if duplicates[str(row["utt_id"])] > 1 else "",
        }
        manifest_rows.append(norm)
    write_csv(manifest_rows, out_dir / "manifest.csv")
    by_speaker_mode = Counter((row["speaker_id"], row["mode"]) for row in rows)
    modes_by_speaker: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        modes_by_speaker[str(row["speaker_id"])].add(str(row["mode"]))
    missing_both = [s for s, modes in modes_by_speaker.items() if not {"speech", "singing"} <= modes]
    for (speaker, mode), count in sorted(by_speaker_mode.items()):
        audit_rows.append({"dataset": args.dataset, "speaker_id": speaker, "mode": mode, "utterances": count})
    write_csv(audit_rows, out_dir / "per_speaker_mode_counts.csv")
    nmat, ncols = nuisance_table(rows)
    nuisance_summary = []
    for col_idx, col in enumerate(ncols):
        values = nmat[:, col_idx]
        nuisance_summary.append(
            {
                "column": col,
                "nan_or_inf": int(np.sum(~np.isfinite(values))),
                "mean": float(np.nanmean(values)) if np.isfinite(values).any() else float("nan"),
                "std": float(np.nanstd(values)) if np.isfinite(values).any() else float("nan"),
            }
        )
    write_csv(nuisance_summary, out_dir / "nuisance_summary.csv")
    report = [
        "# Manifest Audit",
        "",
        f"- dataset: `{args.dataset}`",
        f"- manifest: `{args.manifest}`",
        f"- utterances: {len(rows)}",
        f"- speakers: {len(modes_by_speaker)}",
        f"- missing speech or singing speakers: {len(missing_both)}",
        f"- duplicate utt_id count: {sum(v > 1 for v in duplicates.values())}",
        f"- outputs: `{out_dir}`",
    ]
    (out_dir / "audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def exp_a_variants(rows: list[dict[str, Any]], x: np.ndarray, train_idx: np.ndarray, seed: int) -> list[tuple[str, str, np.ndarray, dict[str, Any], list[str]]]:
    n_full, ncols = nuisance_table(rows)
    n_mode = mode_dummy(rows)
    col_index = {col: i for i, col in enumerate(ncols)}
    rng = np.random.default_rng(seed)
    variants: list[tuple[str, str, np.ndarray, dict[str, Any], list[str]]] = [("raw", "raw", x.copy(), {"variance_removed": 0.0, "rank": 0}, [])]
    centered, diag = apply_residualization(x, np.zeros((len(rows), 0)), train_idx, "classic")
    variants.append(("center_only_classic", "classic", centered, diag, []))
    for name, n, cols in [
        ("mode_dummy_only", n_mode, ["mode_dummy"]),
        ("acoustic_full_no_mode", n_full, ncols),
        ("acoustic_full_plus_mode", np.column_stack([n_full, n_mode]), ncols + ["mode_dummy"]),
    ]:
        for formula in ["mean_preserving", "classic"]:
            out, diag = apply_residualization(x, n, train_idx, formula)
            variants.append((name, formula, out, diag, cols))
    for group, cols in NUISANCE_GROUPS.items():
        idx = [col_index[col] for col in cols if col in col_index]
        if not idx:
            continue
        out, diag = apply_residualization(x, n_full[:, idx], train_idx, "mean_preserving")
        variants.append((f"single_group_{group}", "mean_preserving", out, diag, cols))
    for group, cols in NUISANCE_GROUPS.items():
        drop = {col_index[col] for col in cols if col in col_index}
        keep = [i for i in range(n_full.shape[1]) if i not in drop]
        keep_cols = [ncols[i] for i in keep]
        if not keep:
            continue
        out, diag = apply_residualization(x, n_full[:, keep], train_idx, "mean_preserving")
        variants.append((f"full_minus_{group}", "mean_preserving", out, diag, keep_cols))
    shuffled = n_full.copy()
    train_perm = train_idx.copy()
    rng.shuffle(train_perm)
    shuffled[train_idx] = shuffled[train_perm]
    out, diag = apply_residualization(x, shuffled, train_idx, "mean_preserving")
    variants.append(("row_shuffled_nuisance", "mean_preserving", out, diag, ncols))
    random_n = rng.normal(size=n_full.shape)
    out, diag = apply_residualization(x, random_n, train_idx, "mean_preserving")
    variants.append(("random_gaussian_nuisance", "mean_preserving", out, diag, ncols))
    # Mean-preserving random low-rank removal with train-derived orthonormal directions.
    k = min(n_full.shape[1], x.shape[1])
    q, _ = np.linalg.qr(rng.normal(size=(x.shape[1], k)))
    train_mean = x[train_idx].mean(axis=0, keepdims=True)
    centered_x = x - train_mean
    low_rank = x - centered_x @ q @ q.T
    variants.append(("random_low_rank_projection", "mean_preserving", low_rank, {"variance_removed": variance_removed(x, low_rank), "rank": k}, [f"random_dir_{i}" for i in range(k)]))
    return variants


def run_exp_a(args: argparse.Namespace) -> None:
    rows_all = read_jsonl(args.manifest)
    out_dir = args.results_dir / "expA_residualization_audit"
    run_cache = args.run_root / "cache"
    summary_rows: list[dict[str, Any]] = []
    per_split_rows: list[dict[str, Any]] = []
    per_speaker_rows: list[dict[str, Any]] = []
    geometry: dict[str, Any] = {}
    for spec_name in args.models:
        rows, x, model_meta = load_feature_cache(
            rows_all, args.feature_root, spec_name, run_cache, args.max_speech_per_speaker, args.duration_balanced
        )
        speakers = sorted({str(row["speaker_id"]) for row in rows})
        for seed in args.seeds:
            split = make_speaker_split(speakers, seed, args.split_protocol)
            splits = np.asarray([split[str(row["speaker_id"])] for row in rows])
            train_idx = np.flatnonzero(splits == "train")
            test_idx = np.flatnonzero(splits == "test")
            test_speakers = sorted([s for s, sp in split.items() if sp == "test"])
            raw_metrics = None
            raw_ranks_by_direction: dict[str, dict[str, int]] = {}
            mode_metrics_by_variant = {}
            for variant, formula, xr, diag, cols in exp_a_variants(rows, x, train_idx, seed):
                ret = evaluate_retrieval(rows, xr, test_speakers)
                auc = mode_auc(rows, xr, train_idx, test_idx)
                mode_metrics_by_variant[variant, formula] = auc
                for direction, metrics in ret.items():
                    if variant == "raw":
                        raw_metrics = raw_metrics or {}
                        raw_metrics[direction] = metrics["r"]
                        raw_ranks_by_direction[direction] = {
                            label: int(rank) for label, rank in zip(metrics["labels"], metrics["ranks"])
                        }
                    delta = metrics["r"] - (raw_metrics.get(direction, metrics["r"]) if raw_metrics else metrics["r"])
                    boot = bootstrap_retrieval(
                        metrics,
                        seed=(seed * 1000003 + len(per_split_rows) * 9176 + (0 if direction == "S->G" else 1)),
                        samples=args.bootstrap_samples,
                    )
                    row = {
                        "dataset": args.dataset,
                        "model": spec_name,
                        "layer": model_meta["layer"],
                        "split_seed": seed,
                        "split_protocol": split_name(args.split_protocol),
                        "duration_balanced": bool(args.duration_balanced),
                        "residualization_variant": variant,
                        "formula": formula,
                        "nuisance_columns": ";".join(cols),
                        "direction": direction,
                        "test_speakers_N": metrics["den"],
                        "chance_R1": 1.0 / max(1, metrics["den"]),
                        "R1_num": metrics["num"],
                        "R1_den": metrics["den"],
                        "R1": metrics["r"],
                        **boot,
                        "R5": metrics["r5"],
                        "MRR": metrics["mrr"],
                        "median_rank": metrics["median_rank"],
                        "same_minus_impostor_margin": metrics["margin"],
                        "delta_R1_vs_raw": delta,
                        "mode_AUC": auc,
                        "variance_removed": diag.get("variance_removed", float("nan")),
                        "B_rank": diag.get("rank", 0),
                    }
                    per_split_rows.append(row)
                    raw_rank_map = raw_ranks_by_direction.get(direction, {})
                    for label, rank, hit1, hit5, margin in zip(
                        metrics["labels"], metrics["ranks"], metrics["hit1"], metrics["hit5"], metrics["margins"]
                    ):
                        raw_rank = raw_rank_map.get(label, int(rank))
                        per_speaker_rows.append(
                            {
                                "dataset": args.dataset,
                                "model": spec_name,
                                "layer": model_meta["layer"],
                                "split_seed": seed,
                                "speaker_id": label,
                                "duration_balanced": bool(args.duration_balanced),
                                "direction": direction,
                                "residualization_variant": variant,
                                "formula": formula,
                                "rank": int(rank),
                                "raw_rank": int(raw_rank),
                                "rank_improvement_vs_raw": int(raw_rank) - int(rank),
                                "hit1": int(hit1),
                                "hit5": int(hit5),
                                "same_minus_impostor_margin": float(margin),
                            }
                        )
    write_csv(per_split_rows, out_dir / "per_split.csv")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ["dataset", "model", "layer", "residualization_variant", "formula", "direction"]
    for row in per_split_rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    for key, items in grouped.items():
        r1 = np.asarray([float(item["R1"]) for item in items])
        r5 = np.asarray([float(item["R5"]) for item in items])
        mrr = np.asarray([float(item["MRR"]) for item in items])
        delta = np.asarray([float(item["delta_R1_vs_raw"]) for item in items])
        summary_rows.append(
            {
                **{k: v for k, v in zip(keys, key)},
                "splits": len(items),
                "test_speakers_N": items[0]["test_speakers_N"],
                "chance_R1": items[0]["chance_R1"],
                "R1_mean": float(r1.mean()),
                "R1_std": float(r1.std()),
                "R1_CI_low": float(np.percentile(r1, 2.5)),
                "R1_CI_high": float(np.percentile(r1, 97.5)),
                "R5_mean": float(r5.mean()),
                "MRR_mean": float(mrr.mean()),
                "delta_R1_vs_raw_mean": float(delta.mean()),
                "delta_R1_vs_raw_CI_low": float(np.percentile(delta, 2.5)),
                "delta_R1_vs_raw_CI_high": float(np.percentile(delta, 97.5)),
                "mode_AUC_mean": float(np.nanmean([float(item["mode_AUC"]) for item in items])),
                "notes": "expanded_jvs_audit" if len(args.seeds) >= 20 else "minimal_jvs_audit",
            }
        )
    write_csv(summary_rows, out_dir / "summary.csv")
    write_csv(per_speaker_rows, out_dir / "per_speaker.csv")
    write_json(geometry, out_dir / "geometry.json")
    write_yaml(base_card(args, "expA_residualization_audit") | {"split_seeds": list(args.seeds)}, out_dir / "experiment_card.yaml")
    (out_dir / "README_results.md").write_text(
        "# Experiment A Residualization Audit\n\n"
        "Speaker-disjoint JVS/JVS-MuSiC audit. Residualizers, scalers, mode probes, and centroids use train/test speaker separation; headline rows report exact R@1 numerator and denominator in `per_split.csv`. Bootstrap CIs resample held-out test speakers within each split.\n",
        encoding="utf-8",
    )


def pca_explained(x: np.ndarray, ks: list[int]) -> dict[str, float]:
    if len(x) < 2:
        return {f"pc{k}": float("nan") for k in ks}
    xc = x - x.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(xc, full_matrices=False)
    total = float(np.sum(s**2))
    out = {}
    for k in ks:
        out[f"pc{k}"] = float(np.sum(s[: min(k, len(s))] ** 2) / total) if total > 1e-12 else float("nan")
    return out


def run_b0(args: argparse.Namespace) -> None:
    rows_all = read_jsonl(args.manifest)
    out_dir = args.results_dir / "expB_global_and_reliability"
    rows_out = []
    for spec_name in args.models:
        rows, x, model_meta = load_feature_cache(
            rows_all, args.feature_root, spec_name, args.run_root / "cache", args.max_speech_per_speaker, args.duration_balanced
        )
        speakers = sorted({str(row["speaker_id"]) for row in rows})
        for seed in args.seeds:
            split = make_speaker_split(speakers, seed, args.split_protocol)
            train_speakers = {s for s, sp in split.items() if sp == "train"}
            test_speakers = {s for s, sp in split.items() if sp == "test"}
            cents = centroids(rows, x)
            valid = [s for s in speakers if (s, "speech") in cents and (s, "singing") in cents]
            speech = {s: cents[(s, "speech")] for s in valid}
            singing = {s: cents[(s, "singing")] for s in valid}
            delta = {s: singing[s] - speech[s] for s in valid}
            train_delta = np.vstack([delta[s] for s in valid if s in train_speakers])
            test_delta = np.vstack([delta[s] for s in valid if s in test_speakers])
            mu = train_delta.mean(axis=0)
            residual = test_delta - mu
            denom = float(np.mean(np.sum(test_delta**2, axis=1)))
            global_fraction = float(np.sum(mu**2) / denom) if denom > 1e-12 else float("nan")
            cos = (test_delta @ mu) / np.maximum(np.linalg.norm(test_delta, axis=1) * np.linalg.norm(mu), 1e-12)
            pca_delta = pca_explained(train_delta, [1, 3, 5, 10])
            pca_res = pca_explained(train_delta - mu, [1, 3, 5, 10])
            rows_out.append(
                {
                    "dataset": args.dataset,
                    "model": spec_name,
                    "layer": model_meta["layer"],
                    "split_seed": seed,
                    "global_fraction": global_fraction,
                    "mean_cos_delta_mu": float(np.mean(cos)),
                    "delta_PC1_explained": pca_delta["pc1"],
                    "delta_PC3_explained": pca_delta["pc3"],
                    "residual_PC1_explained": pca_res["pc1"],
                    "residual_PC3_explained": pca_res["pc3"],
                    "test_speakers_N": len(test_speakers),
                    "interpretation": "global_high" if global_fraction > 0.5 else "global_not_dominant",
                }
            )
    write_csv(rows_out, out_dir / "b0_global_residual_accounting.csv")


def split_half_vectors(
    rows: list[dict[str, Any]],
    feature_root: Path,
    spec_name: str,
    speakers: list[str],
    max_speech_per_speaker: int | None,
) -> tuple[list[str], dict[str, dict[str, np.ndarray]], str]:
    extractor, checkpoint, layer = MODEL_SPECS[spec_name]
    by: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"speech": [], "singing": []})
    for row in rows:
        mode = canonical_mode(row)
        if mode in {"speech", "singing"}:
            by[str(row["speaker_id"])][mode].append(row)
    out: dict[str, dict[str, np.ndarray]] = {}
    valid = []
    limitation = []
    for speaker in speakers:
        speech_rows = sorted(by[speaker]["speech"], key=lambda r: str(r.get("phrase_id", r["utt_id"])))
        singing_rows = sorted(by[speaker]["singing"], key=lambda r: str(r.get("phrase_id", r["utt_id"])))
        if max_speech_per_speaker is not None:
            speech_rows = speech_rows[:max_speech_per_speaker]
        if len(speech_rows) < 4 or len(singing_rows) < 1:
            continue
        mid = len(speech_rows) // 2
        speech_a = [feature_vector_from_npz(feature_path(feature_root, extractor, checkpoint, layer, str(r["utt_id"]))) for r in speech_rows[:mid]]
        speech_b = [feature_vector_from_npz(feature_path(feature_root, extractor, checkpoint, layer, str(r["utt_id"]))) for r in speech_rows[mid:]]
        if len(singing_rows) >= 2:
            smid = len(singing_rows) // 2
            singing_a = [feature_vector_from_npz(feature_path(feature_root, extractor, checkpoint, layer, str(r["utt_id"]))) for r in singing_rows[:smid]]
            singing_b = [feature_vector_from_npz(feature_path(feature_root, extractor, checkpoint, layer, str(r["utt_id"]))) for r in singing_rows[smid:]]
        else:
            limitation.append("single_singing_utterance_frame_halves")
            path = feature_path(feature_root, extractor, checkpoint, layer, str(singing_rows[0]["utt_id"]))
            singing_a = [feature_vector_from_npz(path, "a")]
            singing_b = [feature_vector_from_npz(path, "b")]
        out[speaker] = {
            "speech_A": np.vstack(speech_a).mean(axis=0),
            "speech_B": np.vstack(speech_b).mean(axis=0),
            "singing_A": np.vstack(singing_a).mean(axis=0),
            "singing_B": np.vstack(singing_b).mean(axis=0),
        }
        valid.append(speaker)
    return valid, out, ";".join(sorted(set(limitation))) or "utterance_disjoint_halves"


def same_diff_cos(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    sims = l2_normalize(a) @ l2_normalize(b).T
    same = np.diag(sims)
    diff = sims[~np.eye(len(sims), dtype=bool)]
    return float(same.mean()), float(diff.mean()), float(same.mean() - diff.mean())


def run_b1(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.manifest)
    out_dir = args.results_dir / "expB_global_and_reliability"
    rows_out = []
    for spec_name in args.models:
        speakers = sorted({str(row["speaker_id"]) for row in rows})
        valid, hv, limitation = split_half_vectors(rows, args.feature_root, spec_name, speakers, args.max_speech_per_speaker)
        for seed in args.seeds:
            split = make_speaker_split(valid, seed, args.split_protocol)
            train = [s for s in valid if split[s] == "train"]
            test = [s for s in valid if split[s] == "test"]
            if not train or not test:
                continue
            mu_a = np.vstack([hv[s]["singing_A"] - hv[s]["speech_A"] for s in train]).mean(axis=0)
            mu_b = np.vstack([hv[s]["singing_B"] - hv[s]["speech_B"] for s in train]).mean(axis=0)
            targets = {
                "speech": (np.vstack([hv[s]["speech_A"] for s in test]), np.vstack([hv[s]["speech_B"] for s in test])),
                "singing": (np.vstack([hv[s]["singing_A"] for s in test]), np.vstack([hv[s]["singing_B"] for s in test])),
                "delta": (
                    np.vstack([hv[s]["singing_A"] - hv[s]["speech_A"] for s in test]),
                    np.vstack([hv[s]["singing_B"] - hv[s]["speech_B"] for s in test]),
                ),
                "r": (
                    np.vstack([hv[s]["singing_A"] - hv[s]["speech_A"] - mu_a for s in test]),
                    np.vstack([hv[s]["singing_B"] - hv[s]["speech_B"] - mu_b for s in test]),
                ),
            }
            for target, (a, b) in targets.items():
                ret = retrieval(a, b, test, test)
                same, diff, margin = same_diff_cos(a, b)
                pass_state = "underpowered_single_singing_file" if "single_singing" in limitation else "pass" if target == "r" and ret["r"] >= 3 / max(1, len(test)) and margin > 0 else "inspect"
                rows_out.append(
                    {
                        "dataset": args.dataset,
                        "model": spec_name,
                        "layer": MODEL_SPECS[spec_name][2],
                        "split_seed": seed,
                        "target": target,
                        "test_speakers_N": len(test),
                        "chance_R1": 1.0 / max(1, len(test)),
                        "R1_num": ret["num"],
                        "R1_den": ret["den"],
                        "R1": ret["r"],
                        "R5": ret["r5"],
                        "MRR": ret["mrr"],
                        "same_cos_mean": same,
                        "diff_cos_mean": diff,
                        "same_minus_diff": margin,
                        "CI": "not_bootstrapped_minimal",
                        "reliability_pass": pass_state,
                        "split_half_method": limitation,
                    }
                )
    write_csv(rows_out, out_dir / "b1_split_half_reliability.csv")
    write_yaml(base_card(args, "expB_global_and_reliability") | {"split_half_limitation": "JVS-MuSiC has one singing file per speaker; B1 uses frame halves for singing."}, out_dir / "experiment_card.yaml")
    (out_dir / "README_results.md").write_text(
        "# Experiment B Global And Reliability\n\n"
        "Contains B0 global residual accounting and B1 split-half reliability. For JVS-MuSiC, singing split-half is limited because the local manifest has one singing utterance per speaker; the script uses non-overlapping frame halves and labels the result accordingly.\n",
        encoding="utf-8",
    )


def base_card(args: argparse.Namespace, name: str) -> dict[str, Any]:
    return {
        "experiment_name": name,
        "dataset": args.dataset,
        "manifest": str(args.manifest),
        "feature_root": str(args.feature_root),
        "run_root": str(args.run_root),
        "hostname": socket.gethostname(),
        "git_commit": current_git_commit(),
        "python": sys.executable,
        "cache_root": os.environ.get("XDG_CACHE_HOME", ""),
        "max_speech_per_speaker": args.max_speech_per_speaker,
        "duration_balanced": bool(args.duration_balanced),
        "split_protocol": args.split_protocol,
        "residualization_formula": "mean_preserving_and_classic_reported",
        "all_preprocessing_fit_population": "train_speakers_only",
        "claim_supported": "audit_only_until_synthetic_and_controls_pass",
        "known_limitations": "minimal suite; bootstrap CIs and full sweeps not yet included",
    }


def write_final_report(args: argparse.Namespace) -> None:
    path = args.results_dir / "final_experiment_report.md"
    sections = [
        "# Final Experiment Report",
        "",
        "This is a compact interim report for the strict residual suite run.",
        "",
        "## Data Actually Used",
        f"- dataset: `{args.dataset}`",
        f"- manifest: `{args.manifest}`",
        f"- feature root: `{args.feature_root}`",
        "",
        "## Speaker Splits Actually Used",
        f"- split seeds: `{','.join(map(str, args.seeds))}`",
        f"- protocol: `{split_name(args.split_protocol)}`",
        f"- max speech per speaker: `{args.max_speech_per_speaker}`",
        f"- duration balanced: `{bool(args.duration_balanced)}`",
        "",
        "## Outputs",
        f"- manifest audit: `{args.results_dir / 'manifest_audit'}`",
        f"- Experiment A: `{args.results_dir / 'expA_residualization_audit'}`",
        f"- Experiment B0/B1: `{args.results_dir / 'expB_global_and_reliability'}`",
        "",
        "## Current Claim",
        "No final scientific claim is supported by this driver alone until synthetic gate and full control criteria are reviewed.",
        "",
        "## Exact Next Experiment Recommended",
        "Review Exp A controls. If random/shuffled controls do not explain true residualization gains, expand to full seeds/layers and bootstrap CIs; otherwise audit geometry and mean-preserving behavior before mapper work.",
    ]
    path.write_text("\n".join(sections) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict residual identity experiment suite driver.")
    parser.add_argument("--manifest", type=Path, default=Path("/localdisk/bowen/singing_identity/runs/jvs_music_retrieval_2026-07-08/manifests/jvs_music_utterances.jsonl"))
    parser.add_argument("--feature-root", type=Path, default=Path("/localdisk/bowen/singing_identity/features/jvs_music_2026-07-08"))
    parser.add_argument("--run-root", type=Path, default=Path("/localdisk/bowen/singing_identity/runs/identity_residual_suite_2026-07-09"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/identity_residual_suite_2026-07-09"))
    parser.add_argument("--dataset", default="JVS_JVSMuSiC")
    parser.add_argument("--models", nargs="+", default=["wavlm_l12", "mert_l3"], choices=sorted(MODEL_SPECS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 17, 19])
    parser.add_argument("--split-protocol", choices=["jvs_60_20_20", "gtsinger_10_0_10"], default="jvs_60_20_20")
    parser.add_argument("--max-speech-per-speaker", type=int, default=20)
    parser.add_argument("--duration-balanced", action="store_true", help="Select speech utterances up to each speaker's total singing duration.")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--steps", nargs="+", default=["manifest", "expA", "b0", "b1", "report"], choices=["manifest", "expA", "b0", "b1", "report"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.results_dir = args.results_dir.resolve()
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    try:
        if "manifest" in args.steps:
            run_manifest_audit(args)
        if "expA" in args.steps:
            run_exp_a(args)
        if "b0" in args.steps:
            run_b0(args)
        if "b1" in args.steps:
            run_b1(args)
        if "report" in args.steps:
            write_final_report(args)
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote strict residual suite outputs -> {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
