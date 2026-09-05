#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from singing_identity.utils.research_utils import (  # noqa: E402
    ExperimentError,
    apply_nuisance_residualizer,
    assert_no_speaker_leakage,
    auc_summary,
    feature_path,
    fit_logistic_regression,
    fit_nuisance_residualizer,
    kfold_speaker_splits,
    predict_logistic,
    read_table,
    validate_feature_npz,
    write_json,
    write_table,
)


SPECIAL_PHONES = {"<AP>", "<SP>", "<SIL>", "<sil>", "sil", "sp", ""}


def interval_arrays(
    feature_root: Path,
    extractor: str,
    checkpoint_hash: str,
    layer: str,
    utt_id: str,
    start_sec: float,
    end_sec: float,
    voiced_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    path = feature_path(feature_root, extractor, checkpoint_hash, layer, utt_id)
    if not path.exists():
        raise ExperimentError(f"Missing feature file: {path}")
    validate_feature_npz(path)
    with np.load(path) as data:
        x = np.asarray(data["x"], dtype=np.float64)
        times = np.asarray(data["times_sec"], dtype=np.float64)
        voiced = np.asarray(data["voiced_mask"], dtype=bool)
        f0 = np.asarray(data["f0_hz"], dtype=np.float64)
        energy = np.asarray(data["energy"], dtype=np.float64)
        mask = (times >= float(start_sec)) & (times <= float(end_sec))
        if voiced_only:
            mask = mask & voiced
        if not mask.any():
            center = 0.5 * (float(start_sec) + float(end_sec))
            nearest = int(np.argmin(np.abs(times - center)))
            mask = np.zeros_like(times, dtype=bool)
            mask[nearest] = True
        x_i = x[mask]
        f0_i = f0[mask]
        energy_i = energy[mask]
        voiced_i = voiced[mask]
        finite_f0 = f0_i[np.isfinite(f0_i) & (f0_i > 0)]
        if len(finite_f0) == 0:
            finite_f0 = np.asarray([0.0], dtype=np.float64)
        nuisance = np.asarray(
            [
                max(0.0, float(end_sec) - float(start_sec)),
                float(finite_f0.mean()),
                float(finite_f0.std()),
                float(voiced_i.mean()) if len(voiced_i) else 0.0,
                float(np.nanmean(energy_i)) if len(energy_i) else 0.0,
                float(np.nanstd(energy_i)) if len(energy_i) else 0.0,
            ],
            dtype=np.float64,
        )
        vec = np.concatenate([x_i.mean(axis=0), x_i.std(axis=0)]).astype(np.float64)
        return vec, nuisance


def build_examples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    rows = read_table(args.pairs)
    examples: list[dict[str, Any]] = []
    x_rows = []
    nuisance_rows = []
    cache: dict[tuple[str, float, float, bool], tuple[np.ndarray, np.ndarray]] = {}

    def cached(utt_id: str, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
        key = (utt_id, float(start), float(end), bool(args.voiced_only))
        if key not in cache:
            cache[key] = interval_arrays(
                args.feature_root,
                args.extractor,
                args.checkpoint_hash,
                args.layer,
                utt_id,
                start,
                end,
                args.voiced_only,
            )
        return cache[key]

    for row in rows:
        if str(row.get("target_technique")) != args.technique:
            continue
        phone = str(row.get("phone", ""))
        if phone in SPECIAL_PHONES or phone.startswith("<"):
            continue
        base, base_nuis = cached(
            str(row["base_utt_id"]),
            float(row["base_phone_start_sec"]),
            float(row["base_phone_end_sec"]),
        )
        target, target_nuis = cached(
            str(row["technique_utt_id"]),
            float(row["technique_phone_start_sec"]),
            float(row["technique_phone_end_sec"]),
        )
        base_record = {
            "example_id": str(row["base_phone_ex_id"]),
            "pair_id": str(row["pair_id"]),
            "speaker_id": str(row["speaker_id"]),
            "language": str(row["language"]),
            "phone": phone,
            "label": 0,
            "condition": "control",
        }
        target_record = {
            "example_id": str(row["technique_phone_ex_id"]),
            "pair_id": str(row["pair_id"]),
            "speaker_id": str(row["speaker_id"]),
            "language": str(row["language"]),
            "phone": phone,
            "label": 1,
            "condition": args.technique,
        }
        examples.extend([base_record, target_record])
        x_rows.extend([base, target])
        nuisance_rows.extend([base_nuis, target_nuis])
        if args.max_pairs and len(examples) >= 2 * args.max_pairs:
            break
    if len(examples) < 20:
        raise ExperimentError(f"too few examples: {len(examples)}")
    return examples, np.vstack(x_rows), np.vstack(nuisance_rows), np.asarray([row["label"] for row in examples], dtype=int)


def run_one_condition(
    rows: list[dict[str, Any]],
    x_raw: np.ndarray,
    nuisance: np.ndarray,
    y_raw: np.ndarray,
    args: argparse.Namespace,
    condition: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y = y_raw.copy()
    if condition == "shuffled_labels":
        rng = np.random.default_rng(args.seed)
        y = rng.permutation(y)
    splits_list = kfold_speaker_splits(rows, args.folds, args.seed)
    prediction_rows = []
    fold_metrics = []
    for fold_idx, split_map in enumerate(splits_list):
        splits = np.asarray([split_map[str(row["speaker_id"])] for row in rows])
        assert_no_speaker_leakage(rows, splits)
        train_idx = np.where(splits == "train")[0]
        test_idx = np.where(splits == "test")[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        if len(np.unique(y[train_idx])) != 2 or len(np.unique(y[test_idx])) != 2:
            continue
        x = x_raw.copy()
        if condition == "residualized_nuisance":
            residualizer = fit_nuisance_residualizer(x[train_idx], nuisance[train_idx])
            x[train_idx] = apply_nuisance_residualizer(x[train_idx], nuisance[train_idx], residualizer)
            x[test_idx] = apply_nuisance_residualizer(x[test_idx], nuisance[test_idx], residualizer)
        w, b = fit_logistic_regression(x[train_idx], y[train_idx], seed=args.seed + fold_idx)
        scores = predict_logistic(x[train_idx], x[test_idx], w, b)
        auc_info = auc_summary(y[test_idx], scores)
        preds = (scores >= 0.5).astype(int)
        fold_metric = {
            "fold": int(fold_idx),
            "signed_auc": float(auc_info["signed_auc"]),
            "separability_auc": float(auc_info["separability_auc"]),
            "orientation": str(auc_info["orientation"]),
            "accuracy": float((preds == y[test_idx]).mean()),
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
                    "condition": condition,
                    "fold": int(fold_idx),
                    "example_id": row["example_id"],
                    "pair_id": row["pair_id"],
                    "speaker_id": row["speaker_id"],
                    "language": row["language"],
                    "phone": row["phone"],
                    "target": row["condition"],
                    "y_true": int(y[int(idx)]),
                    "y_pred": int(pred),
                    "score": float(score),
                }
            )
    if not fold_metrics:
        raise ExperimentError(f"no valid folds for condition {condition}")
    signed = np.asarray([row["signed_auc"] for row in fold_metrics], dtype=np.float64)
    sep = np.asarray([row["separability_auc"] for row in fold_metrics], dtype=np.float64)
    acc = np.asarray([row["accuracy"] for row in fold_metrics], dtype=np.float64)
    return (
        {
            "condition": condition,
            "folds_completed": len(fold_metrics),
            "fold_metrics": fold_metrics,
            "signed_auc_mean": float(signed.mean()),
            "signed_auc_std": float(signed.std()),
            "separability_auc_mean": float(sep.mean()),
            "separability_auc_std": float(sep.std()),
            "accuracy_mean": float(acc.mean()),
            "accuracy_std": float(acc.std()),
        },
        prediction_rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Speaker-disjoint breathy detection probe from interval features.")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--extractor", required=True)
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--predictions-out", type=Path, required=True)
    parser.add_argument("--run-id", default="breathy_detection_probe")
    parser.add_argument("--technique", default="breathy")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--voiced-only", action="store_true")
    args = parser.parse_args()

    try:
        rows, x, nuisance, y = build_examples(args)
        metrics = {
            "run_id": args.run_id,
            "stage": "breathy_detection_probe",
            "extractor": args.extractor,
            "checkpoint_hash": args.checkpoint_hash,
            "layer_or_stream": args.layer,
            "technique": args.technique,
            "examples": len(rows),
            "pairs": len(rows) // 2,
            "speakers": len({row["speaker_id"] for row in rows}),
            "phones": len({row["phone"] for row in rows}),
            "feature_dim": int(x.shape[1]),
            "nuisance_dim": int(nuisance.shape[1]),
            "conditions": {},
        }
        all_predictions = []
        for condition in ["raw", "residualized_nuisance", "shuffled_labels"]:
            condition_metrics, predictions = run_one_condition(rows, x, nuisance, y, args, condition)
            metrics["conditions"][condition] = condition_metrics
            all_predictions.extend(predictions)
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_json(metrics, args.metrics_out)
    write_table(all_predictions, args.predictions_out)
    for name, item in metrics["conditions"].items():
        print(f"{name}: sep_auc={item['separability_auc_mean']:.4f} acc={item['accuracy_mean']:.4f}")
    print(f"wrote metrics -> {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
