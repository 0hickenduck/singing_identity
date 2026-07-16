#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_utils import (  # noqa: E402
    ExperimentError,
    assert_no_speaker_leakage,
    deterministic_speaker_split,
    kfold_speaker_splits,
    recall_at_k,
    write_json,
    write_table,
)
from probing.run_prompt_mismatch_accounting import (  # noqa: E402
    build_dataset,
    corr_summary,
    one_hot,
    row_cosine,
)


ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0)
SPECIAL_PHONES = {"<AP>", "<SP>", "<SIL>", "<sil>", "sil", "sp", ""}
VOWEL_CHARS = set("aeiouɑɐɒæɛɜəɘɚɝɞɨɪɯɵøœɔʊʌyʏɤɶ")
SONORANT_PREFIXES = ("m", "n", "ŋ", "ɲ", "ɳ", "ɴ", "l", "ɭ", "ʎ", "r", "ɾ", "ɹ", "ɻ", "ʁ", "j", "w", "N", "M", "L", "R", "Y", "W")
ARPABET_VOWELS = ("AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW")


def zscore_from_train(x: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True)
    return (x - mean) / np.where(std < 1e-8, 1.0, std)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.column_stack([np.ones(x.shape[0]), x])
    penalty = np.eye(x_aug.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    lhs = x_aug.T @ x_aug + penalty
    rhs = x_aug.T @ y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


def predict_ridge(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(x.shape[0]), x]) @ beta


def base_phone(phone: str) -> str:
    return phone.split("_", 1)[0]


def phone_class_counts(tokens: list[str]) -> list[float]:
    clean = [token for token in tokens if token not in SPECIAL_PHONES and not token.startswith("<")]
    if not clean:
        return [0.0, 0.0, 0.0, 0.0]
    vowels = 0
    sonorants = 0
    consonants = 0
    for token in clean:
        phone = base_phone(token)
        is_vowel = any(phone.startswith(vowel) for vowel in ARPABET_VOWELS) or any(ch in phone for ch in VOWEL_CHARS)
        is_sonorant = phone.startswith(SONORANT_PREFIXES)
        vowels += int(is_vowel)
        sonorants += int((not is_vowel) and is_sonorant)
        consonants += int((not is_vowel) and (not is_sonorant))
    total = max(1, len(clean))
    return [len(clean), vowels / total, sonorants / total, consonants / total]


def lowdim_designs(
    rows: list[dict[str, Any]],
    acoustic_delta: np.ndarray,
    use_acoustic: bool,
    seed: int,
) -> dict[str, dict[str, Any]]:
    nuisance = np.asarray(
        [
            [
                row["delta_f0_mean"],
                row["delta_f0_std"],
                row["delta_f0_range"],
                row["delta_f0_voiced_pct"],
                row["duration_ratio"],
                row["delta_duration_sec"],
                row["singing_phone_rate"] - row["speech_phone_rate"],
                row["delta_energy_mean"],
                row["delta_energy_std"],
                row["delta_rms_db"],
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    lang, lang_names = one_hot([str(row.get("language", "")) for row in rows], sorted({str(row.get("language", "")) for row in rows}))
    vocal_range, range_names = one_hot(
        [str(row.get("vocal_range", "")) for row in rows], sorted({str(row.get("vocal_range", "")) for row in rows})
    )
    technique, technique_names = one_hot(
        [str(row.get("technique", "")) for row in rows], sorted({str(row.get("technique", "")) for row in rows})
    )
    content = np.asarray(
        [
            [
                float(row["text_len_chars"]),
                float(row["phone_count"]),
                *phone_class_counts(row["phone_tokens"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    metadata = np.hstack([lang, vocal_range])
    m1 = nuisance
    m2 = np.hstack([m1, metadata])
    m3 = np.hstack([m2, content])
    m4 = np.hstack([m3, technique])
    rng = np.random.default_rng(seed)
    shuffled_nuisance = nuisance[rng.permutation(len(rows))]
    shuffled_technique = technique[rng.permutation(len(rows))]
    out = {
        "L1_nuisance": {"x": m1, "description": "F0/prosody + timing/energy only"},
        "L2_nuisance_metadata": {"x": m2, "description": "L1 + language/vocal range"},
        "L3_low_content": {"x": m3, "description": "L2 + low-dimensional content/phone-class summaries"},
        "L4_technique": {"x": m4, "description": "L3 + technique label"},
        "C_metadata_only": {"x": metadata, "description": "negative control: metadata only"},
        "C_content_only": {"x": content, "description": "negative control: low-dimensional content only"},
        "C_shuffled_nuisance": {"x": shuffled_nuisance, "description": "negative control: shuffled nuisance"},
        "C_l4_shuffle_technique": {
            "x": np.hstack([m3, shuffled_technique]),
            "description": "negative control: L4 with shuffled technique labels",
        },
        "_meta": {
            "language_categories": lang_names,
            "vocal_range_categories": range_names,
            "technique_categories": technique_names,
            "use_acoustic_covariates": bool(use_acoustic),
        },
    }
    if use_acoustic and acoustic_delta.shape[1] > 0:
        out["L5_acoustic_proxy"] = {
            "x": np.hstack([m4, acoustic_delta]),
            "description": "L4 + acoustic-baseline delta proxy",
        }
        out["C_acoustic_only"] = {
            "x": acoustic_delta,
            "description": "negative/control: acoustic-baseline delta proxy only",
        }
        out["_meta"]["acoustic_covariate_dim"] = int(acoustic_delta.shape[1])
    else:
        out["_meta"]["acoustic_covariate_dim"] = 0
    return out


def pca_basis(delta_train: np.ndarray, components: int) -> tuple[np.ndarray, np.ndarray]:
    mean = delta_train.mean(axis=0, keepdims=True)
    centered = delta_train - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(int(components), vt.shape[0])
    return mean, vt[:k]


def fit_predict_lowrank(
    x: np.ndarray,
    delta: np.ndarray,
    train_idx: np.ndarray,
    dev_idx: np.ndarray,
    test_idx: np.ndarray,
    components: int,
) -> tuple[np.ndarray, float, float]:
    xz = zscore_from_train(x, train_idx)
    mean, basis = pca_basis(delta[train_idx], components)
    y_train = (delta[train_idx] - mean) @ basis.T
    best_alpha = ALPHA_GRID[-1]
    best_dev_mse = float("inf")
    if len(dev_idx) == 0:
        dev_idx = train_idx
    for alpha in ALPHA_GRID:
        beta = fit_ridge(xz[train_idx], y_train, alpha)
        pred_coef = predict_ridge(xz[dev_idx], beta)
        pred_delta = mean + pred_coef @ basis
        dev_mse = float(np.mean(np.square(delta[dev_idx] - pred_delta)))
        if dev_mse < best_dev_mse:
            best_alpha = alpha
            best_dev_mse = dev_mse
    beta = fit_ridge(xz[train_idx], y_train, best_alpha)
    pred_coef = predict_ridge(xz[test_idx], beta)
    return mean + pred_coef @ basis, float(best_alpha), float(best_dev_mse)


def metrics_for_prediction(
    model: str,
    true_delta: np.ndarray,
    pred_delta: np.ndarray,
    z_speech: np.ndarray,
    z_sing: np.ndarray,
    labels: list[str],
    m0_mse: float,
) -> dict[str, Any]:
    residual = true_delta - pred_delta
    pred_sing = z_speech + pred_delta
    return {
        "model": model,
        "test_items": int(len(true_delta)),
        "delta_cosine_mean": float(np.mean(row_cosine(pred_delta, true_delta))),
        "delta_mse": float(np.mean(np.square(residual))),
        "delta_mse_reduction_vs_m0": float(1.0 - np.mean(np.square(residual)) / max(m0_mse, 1e-12)),
        "residual_norm_mean": float(np.linalg.norm(residual, axis=1).mean()),
        "cos_speech_plus_pred_to_singing_mean": float(np.mean(row_cosine(pred_sing, z_sing))),
        "before_retrieval_r1": recall_at_k(z_speech, z_sing, labels, labels, 1),
        "after_retrieval_r1": recall_at_k(pred_sing, z_sing, labels, labels, 1),
        "after_retrieval_r5": recall_at_k(pred_sing, z_sing, labels, labels, 5),
    }


def downstream_correlations(rows: list[dict[str, Any]], delta: np.ndarray, pred_by_model: dict[str, dict[int, np.ndarray]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model, preds in pred_by_model.items():
        x_resid = []
        x_pred = []
        y = []
        for idx, pred in preds.items():
            gap = float(rows[idx].get("prompt_gap", float("nan")))
            if not math.isfinite(gap):
                continue
            x_resid.append(float(np.linalg.norm(delta[idx] - pred)))
            x_pred.append(float(np.linalg.norm(pred)))
            y.append(gap)
        out[model] = {
            "residual_norm_vs_prompt_gap": corr_summary(x_resid, y),
            "predicted_norm_vs_prompt_gap": corr_summary(x_pred, y),
        }
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.extractor == "acoustic_baseline":
        args.use_acoustic_covariates = False
    dataset = build_dataset(args)
    rows = dataset["rows"]
    z_speech = dataset["z_speech"]
    z_sing = dataset["z_sing"]
    delta = dataset["delta"]
    designs = lowdim_designs(rows, dataset["acoustic_delta"], args.use_acoustic_covariates, args.seed)
    model_names = [name for name in designs if not name.startswith("_")]
    split_maps = (
        [deterministic_speaker_split(rows, args.seed)]
        if args.folds <= 1
        else kfold_speaker_splits(rows, args.folds, args.seed)
    )
    metrics_by_k: dict[str, Any] = {}
    all_prediction_rows = []
    for components in args.target_components:
        pred_by_model: dict[str, dict[int, np.ndarray]] = {"M0_mean_delta": {}}
        true_by_model: dict[str, dict[int, np.ndarray]] = {"M0_mean_delta": {}}
        alpha_rows = []
        for name in model_names:
            pred_by_model[name] = {}
            true_by_model[name] = {}
        for fold_idx, split_map in enumerate(split_maps):
            splits = np.asarray([split_map[str(row["speaker_id"])] for row in rows])
            assert_no_speaker_leakage(rows, splits)
            train_idx = np.where(splits == "train")[0]
            dev_idx = np.where(splits == "dev")[0]
            test_idx = np.where(splits == "test")[0]
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            mean_delta = delta[train_idx].mean(axis=0, keepdims=True)
            m0_pred = np.repeat(mean_delta, len(test_idx), axis=0)
            for local_i, global_i in enumerate(test_idx):
                pred_by_model["M0_mean_delta"][int(global_i)] = m0_pred[local_i]
                true_by_model["M0_mean_delta"][int(global_i)] = delta[int(global_i)]
            print(
                f"k={components} fold {fold_idx}: train/dev/test "
                f"{len(train_idx)}/{len(dev_idx)}/{len(test_idx)}"
            )
            for name in model_names:
                pred, alpha, dev_mse = fit_predict_lowrank(
                    designs[name]["x"], delta, train_idx, dev_idx, test_idx, components
                )
                alpha_rows.append(
                    {
                        "target_components": int(components),
                        "fold": int(fold_idx),
                        "model": name,
                        "selected_alpha": alpha,
                        "dev_mse": dev_mse,
                    }
                )
                for local_i, global_i in enumerate(test_idx):
                    pred_by_model[name][int(global_i)] = pred[local_i]
                    true_by_model[name][int(global_i)] = delta[int(global_i)]
                    row = rows[int(global_i)]
                    all_prediction_rows.append(
                        {
                            "run_id": args.run_id,
                            "target_components": int(components),
                            "fold": int(fold_idx),
                            "model": name,
                            "pair_id": row["pair_id"],
                            "speaker_id": row["speaker_id"],
                            "language": row["language"],
                            "technique": row["technique"],
                            "prompt_gap": float(row["prompt_gap"]),
                            "pred_delta_norm": float(np.linalg.norm(pred[local_i])),
                            "true_delta_norm": float(np.linalg.norm(delta[int(global_i)])),
                            "residual_norm": float(np.linalg.norm(delta[int(global_i)] - pred[local_i])),
                            "delta_cosine": float(row_cosine(pred[local_i][None, :], delta[int(global_i)][None, :])[0]),
                        }
                    )
        m0_idx = sorted(pred_by_model["M0_mean_delta"])
        m0_true = np.vstack([true_by_model["M0_mean_delta"][idx] for idx in m0_idx])
        m0_pred = np.vstack([pred_by_model["M0_mean_delta"][idx] for idx in m0_idx])
        m0_mse = float(np.mean(np.square(m0_true - m0_pred)))
        model_metrics = {}
        for name, preds in pred_by_model.items():
            idx = sorted(preds)
            if not idx:
                continue
            true = np.vstack([true_by_model[name][i] for i in idx])
            pred = np.vstack([pred_by_model[name][i] for i in idx])
            labels = [str(rows[i]["speaker_id"]) for i in idx]
            item = metrics_for_prediction(name, true, pred, z_speech[idx], z_sing[idx], labels, m0_mse)
            item["num_covariates"] = 0 if name == "M0_mean_delta" else int(designs[name]["x"].shape[1])
            item["description"] = "mean Delta_z baseline" if name == "M0_mean_delta" else designs[name]["description"]
            model_metrics[name] = item
        metrics_by_k[str(components)] = {
            "target_components": int(components),
            "model_metrics": model_metrics,
            "downstream_correlations": downstream_correlations(rows, delta, pred_by_model),
            "fold_fit": alpha_rows,
        }
    return {
        "run_id": args.run_id,
        "stage": "track1_prompt_mismatch_accounting_v2_lowdim_lowrank",
        "extractor": args.extractor,
        "checkpoint_hash": args.checkpoint_hash,
        "layer_or_stream": args.layer,
        "pairs": len(rows),
        "speakers": len({str(row["speaker_id"]) for row in rows}),
        "techniques": dict(Counter(str(row["technique"]) for row in rows)),
        "design": designs["_meta"],
        "target_components": [int(k) for k in args.target_components],
        "by_target_components": metrics_by_k,
        "predictions": all_prediction_rows,
    }


def write_summary(metrics: dict[str, Any], path: Path) -> None:
    lines = [
        "# Track 1 Accounting v2: Low-Dim Input, Low-Rank Target",
        "",
        f"Run: `{metrics['run_id']}`",
        f"Representation: `{metrics['extractor']}` / `{metrics['checkpoint_hash']}` / `{metrics['layer_or_stream']}`",
        f"Pairs: {metrics['pairs']}; speakers: {metrics['speakers']}",
        "",
    ]
    for k, block in metrics["by_target_components"].items():
        lines.extend(
            [
                f"## Target PCA K={k}",
                "",
                "| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, item in block["model_metrics"].items():
            lines.append(
                f"| {name} | {item['num_covariates']} | {item['delta_cosine_mean']:.4f} | "
                f"{item['delta_mse_reduction_vs_m0']:.4f} | {item['before_retrieval_r1']:.4f} | "
                f"{item['after_retrieval_r1']:.4f} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Track 1 prompt-mismatch accounting v2.")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--utterances", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--extractor", required=True)
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--run-id", default="track1_prompt_mismatch_accounting_v2")
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--predictions-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--seedvc-summary", type=Path)
    parser.add_argument("--target-components", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--voiced-only", action="store_true")
    parser.add_argument("--no-acoustic-covariates", dest="use_acoustic_covariates", action="store_false")
    parser.set_defaults(use_acoustic_covariates=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    predictions = result.pop("predictions")
    write_json(result, args.metrics_out)
    write_table(predictions, args.predictions_out)
    write_summary(result, args.summary_out)
    for k, block in result["by_target_components"].items():
        best = max(block["model_metrics"].items(), key=lambda item: item[1]["delta_mse_reduction_vs_m0"])
        print(f"k={k} best={best[0]} mse_red={best[1]['delta_mse_reduction_vs_m0']:.4f} r1={best[1]['after_retrieval_r1']:.4f}")
    print(f"wrote metrics -> {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
