#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_utils import (  # noqa: E402
    ExperimentError,
    cosine_similarity_matrix,
    group_rows,
    load_feature_vector,
    read_table,
    validate_pairs,
    validate_utterances,
    write_json,
    write_table,
)


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.column_stack([np.ones(x.shape[0]), x])
    reg = np.eye(x_aug.shape[1]) * alpha
    reg[0, 0] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)


def ridge_predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.column_stack([np.ones(x.shape[0]), x])
    return x_aug @ weights


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a lightweight speech-to-singing residual mapper over cached representation statistics."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--extractor", required=True)
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--predictions-out", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--run-id", default="micro_mapper_ridge")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--voiced-only", action="store_true")
    args = parser.parse_args()

    try:
        utterances = read_table(args.manifest)
        validate_utterances(utterances)
        pairs = read_table(args.pairs)
        validate_pairs(pairs, utterances)
        by_utt = {row["utt_id"]: row for row in utterances}
        by_speaker = group_rows(pairs, "speaker_id")
        speakers = sorted(by_speaker)
        if len(speakers) < 3:
            raise ExperimentError("micro mapper needs at least three speakers for a train/test split")
        heldout = set(speakers[-max(1, len(speakers) // 5):])
        train_pairs = [row for row in pairs if row["speaker_id"] not in heldout]
        test_pairs = [row for row in pairs if row["speaker_id"] in heldout]

        def pair_arrays(pair_rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
            speech_vecs = []
            deltas = []
            kept = []
            for pair in pair_rows:
                speech = load_feature_vector(
                    args.feature_root,
                    args.extractor,
                    args.checkpoint_hash,
                    args.layer,
                    pair["speech_utt_id"],
                    args.voiced_only,
                )
                singing = load_feature_vector(
                    args.feature_root,
                    args.extractor,
                    args.checkpoint_hash,
                    args.layer,
                    pair["singing_utt_id"],
                    args.voiced_only,
                )
                speech_vecs.append(speech)
                deltas.append(singing - speech)
                kept.append(pair)
            return np.vstack(speech_vecs), np.vstack(deltas), kept

        x_train, y_train, _ = pair_arrays(train_pairs)
        x_test, y_test, kept_test = pair_arrays(test_pairs)
        print(f"train speech shape: {x_train.shape}")
        print(f"train delta shape: {y_train.shape}")
        weights = ridge_fit(x_train, y_train, args.alpha)
        pred_delta = ridge_predict(x_test, weights)
        cos = np.diag(cosine_similarity_matrix(pred_delta, y_test))
        mse = ((pred_delta - y_test) ** 2).mean(axis=1)
        global_delta = y_train.mean(axis=0, keepdims=True)
        global_pred = np.repeat(global_delta, len(y_test), axis=0)
        global_cos = np.diag(cosine_similarity_matrix(global_pred, y_test))
        prediction_rows = []
        for pair, cos_value, mse_value, baseline_cos in zip(kept_test, cos, mse, global_cos):
            prediction_rows.append(
                {
                    "run_id": args.run_id,
                    "pair_id": pair["pair_id"],
                    "speaker_id": pair["speaker_id"],
                    "language": pair["language"],
                    "mode": "speech_to_singing",
                    "technique": pair["technique"],
                    "song_id": pair["song_id"],
                    "split": "test",
                    "score": float(cos_value),
                    "mse": float(mse_value),
                    "baseline_name": "ridge_speech_to_delta",
                    "global_mean_residual_cosine": float(baseline_cos),
                    "system_name": "micro_mapper_ridge",
                    "oracle_flag": False,
                    "deployable_flag": True,
                }
            )
        metrics = {
            "run_id": args.run_id,
            "stage": "stage_b",
            "extractor": args.extractor,
            "checkpoint_hash": args.checkpoint_hash,
            "layer_or_stream": args.layer,
            "model": "ridge_speech_to_delta",
            "alpha": args.alpha,
            "train_speakers": len(set(row["speaker_id"] for row in train_pairs)),
            "test_speakers": len(set(row["speaker_id"] for row in test_pairs)),
            "train_pairs": len(train_pairs),
            "test_pairs": len(test_pairs),
            "cosine_mean": float(np.mean(cos)),
            "cosine_std": float(np.std(cos)),
            "mse_mean": float(np.mean(mse)),
            "global_mean_residual_cosine_mean": float(np.mean(global_cos)),
        }
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.model_out, weights=weights, alpha=np.asarray([args.alpha], dtype=np.float32))
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    write_json(metrics, args.metrics_out)
    write_table(prediction_rows, args.predictions_out)
    print(f"cosine_mean: {metrics['cosine_mean']:.4f}")
    print(f"wrote model -> {args.model_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
