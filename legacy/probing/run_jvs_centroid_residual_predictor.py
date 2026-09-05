#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_utils import (  # noqa: E402
    ExperimentError,
    cosine_similarity_matrix,
    deterministic_speaker_split,
    load_feature_vector,
    raw_nuisance_matrix,
    read_table,
    recall_at_k,
    validate_utterances,
    write_json,
    write_table,
)


ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def zscore_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return mean, np.where(std < 1e-8, 1.0, std)


def zscore_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.column_stack([np.ones(x.shape[0]), x])
    reg = np.eye(x_aug.shape[1], dtype=np.float64) * float(alpha)
    reg[0, 0] = 0.0
    lhs = x_aug.T @ x_aug + reg
    rhs = x_aug.T @ y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


def ridge_predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(x.shape[0]), x]) @ weights


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    return np.sum(a * b, axis=1) / denom


def fit_predict_ridge(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    dev_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    mean, std = zscore_fit(x[train_idx])
    xz = zscore_apply(x, mean, std)
    if len(dev_idx) == 0:
        dev_idx = train_idx
    best_alpha = ALPHA_GRID[-1]
    best_dev_mse = float("inf")
    for alpha in ALPHA_GRID:
        weights = ridge_fit(xz[train_idx], y[train_idx], alpha)
        pred = ridge_predict(xz[dev_idx], weights)
        dev_mse = float(np.mean(np.square(y[dev_idx] - pred)))
        if dev_mse < best_dev_mse:
            best_alpha = float(alpha)
            best_dev_mse = dev_mse
    weights = ridge_fit(xz[train_idx], y[train_idx], best_alpha)
    return ridge_predict(xz[test_idx], weights), best_alpha, best_dev_mse


def one_hot(values: list[str]) -> tuple[np.ndarray, list[str]]:
    cats = sorted(set(values))
    index = {cat: i for i, cat in enumerate(cats)}
    mat = np.zeros((len(values), len(cats)), dtype=np.float64)
    for row, value in enumerate(values):
        mat[row, index[value]] = 1.0
    return mat, cats


def speaker_centroids(
    rows: list[dict[str, Any]],
    feature_root: Path,
    extractor: str,
    checkpoint_hash: str,
    layer: str,
    voiced_only: bool,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    by_speaker: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        by_speaker.setdefault(str(row["speaker_id"]), {"speech": [], "singing": []})
        if row["mode"] == "speech":
            by_speaker[str(row["speaker_id"])]["speech"].append(row)
        elif row["mode"] == "singing":
            by_speaker[str(row["speaker_id"])]["singing"].append(row)

    meta_rows: list[dict[str, Any]] = []
    speech_vecs: list[np.ndarray] = []
    sing_vecs: list[np.ndarray] = []
    nuisance_vecs: list[np.ndarray] = []
    for speaker in sorted(by_speaker):
        speech_rows = by_speaker[speaker]["speech"]
        singing_rows = by_speaker[speaker]["singing"]
        if not speech_rows or not singing_rows:
            continue
        speech_features = [
            load_feature_vector(feature_root, extractor, checkpoint_hash, layer, row["utt_id"], voiced_only)
            for row in speech_rows
        ]
        singing_features = [
            load_feature_vector(feature_root, extractor, checkpoint_hash, layer, row["utt_id"], voiced_only)
            for row in singing_rows
        ]
        speech_vec = np.vstack(speech_features).mean(axis=0)
        singing_vec = np.vstack(singing_features).mean(axis=0)
        nuisance = raw_nuisance_matrix(speech_rows).mean(axis=0)
        first_speech = speech_rows[0]
        first_singing = singing_rows[0]
        meta_rows.append(
            {
                "speaker_id": speaker,
                "language": first_speech["language"],
                "vocal_range": first_speech["vocal_range"],
                "speech_utterances": len(speech_rows),
                "singing_utterances": len(singing_rows),
                "speech_duration_sec": float(sum(float(row["duration_sec"]) for row in speech_rows)),
                "singing_duration_sec": float(sum(float(row["duration_sec"]) for row in singing_rows)),
                "speech_example": first_speech["utt_id"],
                "singing_example": first_singing["utt_id"],
            }
        )
        speech_vecs.append(speech_vec)
        sing_vecs.append(singing_vec)
        nuisance_vecs.append(nuisance)
    if len(meta_rows) < 3:
        raise ExperimentError("need at least three speakers with both speech and singing features")
    return meta_rows, np.vstack(speech_vecs), np.vstack(sing_vecs), np.vstack(nuisance_vecs)


def prediction_metrics(
    name: str,
    pred_delta: np.ndarray,
    true_delta: np.ndarray,
    z_speech: np.ndarray,
    z_sing: np.ndarray,
    labels: list[str],
    m0_mse: float,
) -> dict[str, Any]:
    pred_sing = z_speech + pred_delta
    delta_mse = float(np.mean(np.square(true_delta - pred_delta)))
    return {
        "model": name,
        "test_speakers": int(len(labels)),
        "delta_cosine_mean": float(np.mean(row_cosine(pred_delta, true_delta))),
        "delta_mse": delta_mse,
        "delta_mse_reduction_vs_speech_only": float(1.0 - delta_mse / max(m0_mse, 1e-12)),
        "singing_cosine_mean": float(np.mean(row_cosine(pred_sing, z_sing))),
        "speech_only_singing_cosine_mean": float(np.mean(row_cosine(z_speech, z_sing))),
        "retrieval_r1": float(recall_at_k(pred_sing, z_sing, labels, labels, 1)),
        "retrieval_r5": float(recall_at_k(pred_sing, z_sing, labels, labels, min(5, len(labels)))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Pro-style speaker-centroid residual predictors on JVS/JVS-MuSiC.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--extractor", required=True)
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--voiced-only", action="store_true")
    args = parser.parse_args()

    try:
        rows = read_table(args.manifest)
        validate_utterances(rows, require_audio_exists=True)
        meta, z_speech, z_sing, nuisance = speaker_centroids(
            rows,
            args.feature_root,
            args.extractor,
            args.checkpoint_hash,
            args.layer,
            args.voiced_only,
        )
        delta = z_sing - z_speech
        split_map = deterministic_speaker_split(meta, seed=args.seed, train_frac=0.7, dev_frac=0.15)
        splits = np.asarray([split_map[row["speaker_id"]] for row in meta])
        train_idx = np.flatnonzero(splits == "train")
        dev_idx = np.flatnonzero(splits == "dev")
        test_idx = np.flatnonzero(splits == "test")
        if len(train_idx) < 2 or len(test_idx) < 1:
            raise ExperimentError("invalid speaker split")

        test_labels = [meta[i]["speaker_id"] for i in test_idx]
        zero_delta = np.zeros_like(delta[test_idx])
        m0_mse = float(np.mean(np.square(delta[test_idx] - zero_delta)))
        global_delta = delta[train_idx].mean(axis=0, keepdims=True)
        pred_global = np.repeat(global_delta, len(test_idx), axis=0)

        language, language_cats = one_hot([row["language"] for row in meta])
        vocal_range, range_cats = one_hot([row["vocal_range"] for row in meta])
        duration = np.asarray(
            [[row["speech_duration_sec"], row["singing_duration_sec"], row["speech_utterances"], row["singing_utterances"]] for row in meta],
            dtype=np.float64,
        )
        acoustic_x = np.hstack([nuisance, duration, language, vocal_range])
        speech_x = z_speech
        speech_acoustic_x = np.hstack([z_speech, acoustic_x])

        pred_acoustic, alpha_acoustic, dev_acoustic = fit_predict_ridge(acoustic_x, delta, train_idx, dev_idx, test_idx)
        pred_speech, alpha_speech, dev_speech = fit_predict_ridge(speech_x, delta, train_idx, dev_idx, test_idx)
        pred_speech_acoustic, alpha_speech_acoustic, dev_speech_acoustic = fit_predict_ridge(
            speech_acoustic_x, delta, train_idx, dev_idx, test_idx
        )

        metrics = [
            prediction_metrics("speech_only_no_residual", zero_delta, delta[test_idx], z_speech[test_idx], z_sing[test_idx], test_labels, m0_mse),
            prediction_metrics("global_mean_residual", pred_global, delta[test_idx], z_speech[test_idx], z_sing[test_idx], test_labels, m0_mse),
            prediction_metrics("acoustic_metadata_ridge", pred_acoustic, delta[test_idx], z_speech[test_idx], z_sing[test_idx], test_labels, m0_mse),
            prediction_metrics("speech_embedding_ridge", pred_speech, delta[test_idx], z_speech[test_idx], z_sing[test_idx], test_labels, m0_mse),
            prediction_metrics(
                "speech_embedding_plus_acoustic_ridge",
                pred_speech_acoustic,
                delta[test_idx],
                z_speech[test_idx],
                z_sing[test_idx],
                test_labels,
                m0_mse,
            ),
        ]
        alpha_by_model = {
            "acoustic_metadata_ridge": {"alpha": alpha_acoustic, "dev_mse": dev_acoustic},
            "speech_embedding_ridge": {"alpha": alpha_speech, "dev_mse": dev_speech},
            "speech_embedding_plus_acoustic_ridge": {"alpha": alpha_speech_acoustic, "dev_mse": dev_speech_acoustic},
        }
        row_by_speaker = {row["speaker_id"]: row for row in meta}
        prediction_rows: list[dict[str, Any]] = []
        predictions = {
            "speech_only_no_residual": zero_delta,
            "global_mean_residual": pred_global,
            "acoustic_metadata_ridge": pred_acoustic,
            "speech_embedding_ridge": pred_speech,
            "speech_embedding_plus_acoustic_ridge": pred_speech_acoustic,
        }
        for model_name, pred in predictions.items():
            pred_sing = z_speech[test_idx] + pred
            for local_i, speaker in enumerate(test_labels):
                prediction_rows.append(
                    {
                        "speaker_id": speaker,
                        "split": "test",
                        "model": model_name,
                        "delta_cosine": float(row_cosine(pred[local_i : local_i + 1], delta[test_idx][local_i : local_i + 1])[0]),
                        "delta_mse": float(np.mean(np.square(delta[test_idx][local_i] - pred[local_i]))),
                        "singing_cosine": float(row_cosine(pred_sing[local_i : local_i + 1], z_sing[test_idx][local_i : local_i + 1])[0]),
                        "speech_only_singing_cosine": float(
                            row_cosine(z_speech[test_idx][local_i : local_i + 1], z_sing[test_idx][local_i : local_i + 1])[0]
                        ),
                        "speech_utterances": row_by_speaker[speaker]["speech_utterances"],
                        "speech_duration_sec": row_by_speaker[speaker]["speech_duration_sec"],
                        "singing_duration_sec": row_by_speaker[speaker]["singing_duration_sec"],
                    }
                )

        summary = {
            "stage": "track1_jvs_centroid_residual_predictor",
            "manifest": str(args.manifest),
            "feature_root": str(args.feature_root),
            "extractor": args.extractor,
            "checkpoint_hash": args.checkpoint_hash,
            "layer": args.layer,
            "voiced_only": bool(args.voiced_only),
            "seed": args.seed,
            "speakers": len(meta),
            "feature_dim": int(z_speech.shape[1]),
            "split_counts": {split: int(np.sum(splits == split)) for split in sorted(set(splits))},
            "language_categories": language_cats,
            "vocal_range_categories": range_cats,
            "alpha_by_model": alpha_by_model,
            "metrics": metrics,
        }
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(summary, args.out_dir / "metrics.json")
    write_table(prediction_rows, args.out_dir / "predictions.jsonl")
    write_table([{**row, "split": split_map[row["speaker_id"]]} for row in meta], args.out_dir / "speaker_manifest.jsonl")
    print("metrics:")
    for item in metrics:
        print(
            f"{item['model']}: delta_mse={item['delta_mse']:.6f}, "
            f"red={item['delta_mse_reduction_vs_speech_only']:.4f}, "
            f"sing_cos={item['singing_cosine_mean']:.4f}, r1={item['retrieval_r1']:.4f}"
        )
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
