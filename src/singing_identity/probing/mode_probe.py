#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from singing_identity.utils.research_utils import (  # noqa: E402
    ExperimentError,
    apply_nuisance_residualizer,
    assert_no_speaker_leakage,
    auc_summary,
    build_matrix,
    fit_nuisance_residualizer,
    fit_logistic_regression,
    kfold_speaker_splits,
    manifest_summary,
    predict_logistic,
    raw_nuisance_matrix,
    read_table,
    validate_utterances,
    write_json,
    write_table,
)


def run_probe(args: argparse.Namespace) -> dict:
    rows = read_table(args.manifest)
    validate_utterances(rows)
    rows = [row for row in rows if row["mode"] in {"speech", "singing", "singing_control", "singing_technique"}]
    if args.binary_singing:
        rows = [
            {**row, "mode_binary": "speech" if row["mode"] == "speech" else "singing"}
            for row in rows
        ]
    else:
        rows = [row for row in rows if row["mode"] in {"speech", "singing"}]
        rows = [{**row, "mode_binary": row["mode"]} for row in rows]
    if len({row["mode_binary"] for row in rows}) != 2:
        raise ExperimentError("mode probe requires both speech and singing rows")

    x_raw, utt_ids = build_matrix(
        rows, args.feature_root, args.extractor, args.checkpoint_hash, args.layer, args.voiced_only
    )
    y = np.asarray([1 if row["mode_binary"] == "singing" else 0 for row in rows], dtype=int)
    if args.shuffle_labels:
        rng = np.random.default_rng(args.seed)
        y = rng.permutation(y)
    nuisance_raw = raw_nuisance_matrix(rows)
    fold_splits = kfold_speaker_splits(rows, args.folds, args.seed)

    prediction_rows = []
    fold_metrics = []
    print(f"x shape: {x_raw.shape}")
    for fold_idx, speaker_split in enumerate(fold_splits):
        splits = np.asarray([speaker_split[row["speaker_id"]] for row in rows])
        assert_no_speaker_leakage(rows, splits)
        train_idx = np.where(splits == "train")[0]
        test_idx = np.where(splits == "test")[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        if len(np.unique(y[train_idx])) != 2 or len(np.unique(y[test_idx])) != 2:
            continue
        x = x_raw.copy()
        if args.residualize_nuisance:
            residualizer = fit_nuisance_residualizer(x[train_idx], nuisance_raw[train_idx])
            x[train_idx] = apply_nuisance_residualizer(x[train_idx], nuisance_raw[train_idx], residualizer)
            x[test_idx] = apply_nuisance_residualizer(x[test_idx], nuisance_raw[test_idx], residualizer)

        print(f"fold {fold_idx}: train/test rows: {len(train_idx)}/{len(test_idx)}")
        w, b = fit_logistic_regression(x[train_idx], y[train_idx], seed=args.seed + fold_idx, steps=args.steps, lr=args.lr)
        scores = predict_logistic(x[train_idx], x[test_idx], w, b)
        auc_info = auc_summary(y[test_idx], scores)
        preds = (scores >= 0.5).astype(int)
        accuracy = float((preds == y[test_idx]).mean())
        fold_metric = {
            "fold": fold_idx,
            "signed_auc": float(auc_info["signed_auc"]),
            "separability_auc": float(auc_info["separability_auc"]),
            "orientation": str(auc_info["orientation"]),
            "accuracy": accuracy,
            "train_speakers": len({rows[i]["speaker_id"] for i in train_idx}),
            "test_speakers": len({rows[i]["speaker_id"] for i in test_idx}),
            "train_items": int(len(train_idx)),
            "test_items": int(len(test_idx)),
        }
        fold_metrics.append(fold_metric)
        for idx, score, pred in zip(test_idx, scores, preds):
            row = rows[int(idx)]
            prediction_rows.append(
                {
                    "run_id": args.run_id,
                    "fold": fold_idx,
                    "utt_id": row["utt_id"],
                    "speaker_id": row["speaker_id"],
                    "language": row["language"],
                    "mode": row["mode_binary"],
                    "split": "test",
                    "y_true": int(y[int(idx)]),
                    "y_pred": int(pred),
                    "score": float(score),
                    "baseline_name": (
                        "shuffled_labels"
                        if args.shuffle_labels
                        else "nuisance_residualized"
                        if args.residualize_nuisance
                        else "raw"
                    ),
                    "system_name": "mode_probe",
                    "oracle_flag": False,
                    "deployable_flag": True,
                }
            )
    if not fold_metrics:
        raise ExperimentError("no valid folds had both classes in train and test")
    signed_values = np.asarray([fold["signed_auc"] for fold in fold_metrics], dtype=np.float64)
    separability_values = np.asarray([fold["separability_auc"] for fold in fold_metrics], dtype=np.float64)
    accuracy_values = np.asarray([fold["accuracy"] for fold in fold_metrics], dtype=np.float64)
    orientation_counts = {}
    for fold in fold_metrics:
        orientation_counts[fold["orientation"]] = orientation_counts.get(fold["orientation"], 0) + 1
    metrics = {
        "run_id": args.run_id,
        "stage": "stage_a",
        "manifest_hash": manifest_summary(rows)["manifest_hash"],
        "split_name": "speaker_disjoint",
        "extractor": args.extractor,
        "checkpoint_hash": args.checkpoint_hash,
        "layer_or_stream": args.layer,
        "representation": "mean_std",
        "target": "mode_binary",
        "controls": "f0_energy_duration_voiced_rms" if args.residualize_nuisance else "none",
        "probe_type": "numpy_logistic_regression",
        "optimization_steps": args.steps,
        "learning_rate": args.lr,
        "folds_requested": args.folds,
        "folds_completed": len(fold_metrics),
        "shuffle_labels": bool(args.shuffle_labels),
        "train_only_residualization": bool(args.residualize_nuisance),
        "orientation_counts": orientation_counts,
        "fold_metrics": fold_metrics,
        "metrics": {
            "signed_auc_mean": float(np.mean(signed_values)),
            "signed_auc_std": float(np.std(signed_values)),
            "separability_auc_mean": float(np.mean(separability_values)),
            "separability_auc_std": float(np.std(separability_values)),
            "accuracy_mean": float(np.mean(accuracy_values)),
            "accuracy_std": float(np.std(accuracy_values)),
            # Backward-compatible field: signed AUC for old readers.
            "auc": float(np.mean(signed_values)),
            "accuracy": float(np.mean(accuracy_values)),
        },
        "seed": args.seed,
    }
    return {"metrics": metrics, "predictions": prediction_rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run speech-vs-singing mode probe on cached features.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--extractor", required=True)
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--run-id", default="mode_probe")
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--predictions-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--voiced-only", action="store_true")
    parser.add_argument("--residualize-nuisance", action="store_true")
    parser.add_argument("--shuffle-labels", action="store_true")
    parser.add_argument("--binary-singing", action="store_true", help="Map singing_control/singing_technique to singing.")
    args = parser.parse_args()
    if args.shuffle_labels and args.steps == parser.get_default("steps"):
        args.steps = 400
    try:
        result = run_probe(args)
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_json(result["metrics"], args.metrics_out)
    write_table(result["predictions"], args.predictions_out)
    print(
        "signed_auc_mean: "
        f"{result['metrics']['metrics']['signed_auc_mean']:.4f}; "
        "separability_auc_mean: "
        f"{result['metrics']['metrics']['separability_auc_mean']:.4f}"
    )
    print(f"wrote metrics -> {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
