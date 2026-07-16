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

from probing.run_identity_residual_suite import (  # noqa: E402
    MODEL_SPECS,
    canonical_mode,
    centroids,
    finite_float,
    l2_normalize,
    load_feature_cache,
    make_speaker_split,
    nuisance_table,
    read_jsonl,
    retrieval,
    split_name,
    write_csv,
    write_json,
    write_yaml,
)
from research_utils import ExperimentError, current_git_commit  # noqa: E402


ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
PCA_KS = (2, 5, 10, 20, 50)


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)


def zscore_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = np.where(x.std(axis=0, keepdims=True) < 1e-8, 1.0, x.std(axis=0, keepdims=True))
    return mean, std


def zscore_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def ridge_weights(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    if x.shape[1] > x.shape[0]:
        # Dual ridge is much faster for high-dimensional embeddings with few train speakers.
        penalty = np.eye(x.shape[0]) * float(alpha)
        gram = x @ x.T + penalty
        dual = np.linalg.solve(gram, y)
        return x.T @ dual
    reg = np.eye(x.shape[1]) * float(alpha)
    return np.linalg.solve(x.T @ x + reg, x.T @ y)


def ridge_predict(x: np.ndarray, weights: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return intercept + x @ weights


def fit_ridge_model(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    dev_idx: np.ndarray,
    fixed_alpha: float = 1.0,
) -> tuple[np.ndarray, float, float]:
    mean, std = zscore_fit(x[train_idx])
    xz = zscore_apply(x, mean, std)
    y_intercept = y[train_idx].mean(axis=0, keepdims=True)
    y_centered = y - y_intercept
    if len(dev_idx) == 0:
        alpha = fixed_alpha
        weights = ridge_weights(xz[train_idx], y_centered[train_idx], alpha)
        return ridge_predict(xz, weights, y_intercept), alpha, float("nan")
    best_alpha = ALPHAS[0]
    best_mse = float("inf")
    for alpha in ALPHAS:
        weights = ridge_weights(xz[train_idx], y_centered[train_idx], alpha)
        pred = ridge_predict(xz[dev_idx], weights, y_intercept)
        mse = float(np.mean(np.square(y[dev_idx] - pred)))
        if mse < best_mse:
            best_alpha = alpha
            best_mse = mse
    weights = ridge_weights(xz[train_idx], y_centered[train_idx], best_alpha)
    return ridge_predict(xz, weights, y_intercept), float(best_alpha), best_mse


def fit_pca_ridge_model(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    dev_idx: np.ndarray,
    fixed_alpha: float = 1.0,
    fixed_k: int = 5,
) -> tuple[np.ndarray, float, int, float]:
    y_train = y[train_idx]
    y_mean = y_train.mean(axis=0, keepdims=True)
    yc = y_train - y_mean
    _, _, vt = np.linalg.svd(yc, full_matrices=False)
    max_k = min(len(train_idx) - 1, vt.shape[0], vt.shape[1])
    if max_k < 1:
        return np.repeat(y_mean, len(y), axis=0), fixed_alpha, 0, float("nan")
    candidates = [k for k in PCA_KS if k <= max_k]
    if not candidates:
        candidates = [max_k]
    if len(dev_idx) == 0:
        candidates = [min(fixed_k, max_k)]
        alphas = [fixed_alpha]
    else:
        alphas = list(ALPHAS)
    best = (float("inf"), fixed_alpha, candidates[0], np.repeat(y_mean, len(y), axis=0))
    for k in candidates:
        comps = vt[:k]
        coeff = (y - y_mean) @ comps.T
        for alpha in alphas:
            coeff_pred, _, _ = fit_ridge_model(x, coeff, train_idx, dev_idx, fixed_alpha=alpha)
            y_pred = y_mean + coeff_pred @ comps
            score_idx = dev_idx if len(dev_idx) else train_idx
            mse = float(np.mean(np.square(y[score_idx] - y_pred[score_idx])))
            if mse < best[0]:
                best = (mse, float(alpha), int(k), y_pred)
    return best[3], best[1], best[2], best[0]


def acoustic_centroids(rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    nmat, _ = nuisance_table(rows)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        if canonical_mode(row) == "speech":
            groups[str(row["speaker_id"])].append(i)
    speakers = sorted(groups)
    vecs = []
    for speaker in speakers:
        arr = nmat[groups[speaker]]
        means = []
        for col in range(arr.shape[1]):
            finite = arr[:, col][np.isfinite(arr[:, col])]
            means.append(float(finite.mean()) if len(finite) else 0.0)
        vecs.append(means)
    return speakers, np.asarray(vecs, dtype=np.float64)


def speaker_matrices(rows: list[dict[str, Any]], x: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    cents = centroids(rows, x)
    speakers = sorted({speaker for speaker, mode in cents if (speaker, "speech") in cents and (speaker, "singing") in cents})
    speech = np.vstack([cents[(speaker, "speech")] for speaker in speakers])
    singing = np.vstack([cents[(speaker, "singing")] for speaker in speakers])
    acoustic_speakers, acoustic = acoustic_centroids(rows)
    acoustic_map = {speaker: acoustic[i] for i, speaker in enumerate(acoustic_speakers)}
    acoustic_mat = np.vstack([acoustic_map[speaker] for speaker in speakers])
    return speakers, speech, singing, acoustic_mat


def eval_prediction(
    model_name: str,
    pred_r: np.ndarray,
    true_r: np.ndarray,
    speech: np.ndarray,
    singing: np.ndarray,
    mu: np.ndarray,
    test_idx: np.ndarray,
    labels: list[str],
    global_mse: float,
    alpha: float | str = "",
    pca_k: int | str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pred_g = speech[test_idx] + mu + pred_r[test_idx]
    true_g = singing[test_idx]
    ret = retrieval(pred_g, true_g, labels, labels)
    residual_mse = float(np.mean(np.square(true_r[test_idx] - pred_r[test_idx])))
    residual_cos = float(np.mean(row_cosine(pred_r[test_idx], true_r[test_idx])))
    global_pred_g = speech[test_idx] + mu
    speech_pred_g = speech[test_idx]
    row = {
        "mapper": model_name,
        "ridge_alpha": alpha,
        "target_pca_K": pca_k,
        "test_speakers_N": len(test_idx),
        "chance_R1": 1.0 / max(1, len(test_idx)),
        "predicted_g_R1_num": ret["num"],
        "predicted_g_R1_den": ret["den"],
        "predicted_g_R1": ret["r"],
        "predicted_g_R5": ret["r5"],
        "predicted_g_MRR": ret["mrr"],
        "cosine_to_true_g": float(np.mean(row_cosine(pred_g, true_g))),
        "delta_cos_vs_speech_only": float(np.mean(row_cosine(pred_g, true_g) - row_cosine(speech_pred_g, true_g))),
        "delta_cos_vs_global": float(np.mean(row_cosine(pred_g, true_g) - row_cosine(global_pred_g, true_g))),
        "residual_MSE": residual_mse,
        "residual_MSE_reduction_vs_global": float((global_mse - residual_mse) / max(global_mse, 1e-12)),
        "residual_cosine": residual_cos,
    }
    if extra:
        row.update(extra)
    return row


def run_mapper(args: argparse.Namespace) -> None:
    rows_all = read_jsonl(args.manifest)
    out_dir = args.results_dir / "expB_mapper_eval"
    run_cache = args.run_root / "cache"
    summary_rows: list[dict[str, Any]] = []
    per_split_rows: list[dict[str, Any]] = []
    for spec_name in args.models:
        rows, x, model_meta = load_feature_cache(
            rows_all, args.feature_root, spec_name, run_cache, args.max_speech_per_speaker, args.duration_balanced
        )
        speakers, speech, singing, acoustic = speaker_matrices(rows, x)
        delta = singing - speech
        for seed in args.seeds:
            split = make_speaker_split(speakers, seed, args.split_protocol)
            split_arr = np.asarray([split[speaker] for speaker in speakers])
            train_idx = np.flatnonzero(split_arr == "train")
            dev_idx = np.flatnonzero(split_arr == "dev")
            test_idx = np.flatnonzero(split_arr == "test")
            if len(train_idx) < 3 or len(test_idx) < 1:
                raise ExperimentError("mapper split needs at least 3 train speakers and 1 test speaker")
            labels = [speakers[i] for i in test_idx]
            mu = delta[train_idx].mean(axis=0, keepdims=True)
            true_r = delta - mu
            global_pred_r = np.zeros_like(true_r)
            global_mse = float(np.mean(np.square(true_r[test_idx])))
            speech_only_pred_r = -np.repeat(mu, len(speakers), axis=0)

            common = {
                "dataset": args.dataset,
                "model": spec_name,
                "layer": model_meta["layer"],
                "split_seed": seed,
                "split_protocol": split_name(args.split_protocol),
                "target": "r",
            }
            rows_for_split = [
                eval_prediction("speech_only", speech_only_pred_r, true_r, speech, singing, mu, test_idx, labels, global_mse),
                eval_prediction("global_mean_residual", global_pred_r, true_r, speech, singing, mu, test_idx, labels, global_mse),
            ]

            pred_acoustic, alpha_acoustic, dev_acoustic = fit_ridge_model(acoustic, true_r, train_idx, dev_idx)
            rows_for_split.append(
                eval_prediction(
                    "acoustic_only_ridge",
                    pred_acoustic,
                    true_r,
                    speech,
                    singing,
                    mu,
                    test_idx,
                    labels,
                    global_mse,
                    alpha_acoustic,
                    "",
                    {"dev_residual_MSE": dev_acoustic, "input_features": "speech_acoustic"},
                )
            )
            pred_speech, alpha_speech, dev_speech = fit_ridge_model(speech, true_r, train_idx, dev_idx)
            rows_for_split.append(
                eval_prediction(
                    "speech_embedding_ridge_full",
                    pred_speech,
                    true_r,
                    speech,
                    singing,
                    mu,
                    test_idx,
                    labels,
                    global_mse,
                    alpha_speech,
                    "",
                    {"dev_residual_MSE": dev_speech, "input_features": "speech_embedding"},
                )
            )
            speech_acoustic = np.hstack([speech, acoustic])
            pred_combo, alpha_combo, dev_combo = fit_ridge_model(speech_acoustic, true_r, train_idx, dev_idx)
            rows_for_split.append(
                eval_prediction(
                    "speech_embedding_plus_acoustic_ridge_full",
                    pred_combo,
                    true_r,
                    speech,
                    singing,
                    mu,
                    test_idx,
                    labels,
                    global_mse,
                    alpha_combo,
                    "",
                    {"dev_residual_MSE": dev_combo, "input_features": "speech_embedding_plus_acoustic"},
                )
            )
            pred_pca, alpha_pca, k_pca, dev_pca = fit_pca_ridge_model(speech, true_r, train_idx, dev_idx)
            rows_for_split.append(
                eval_prediction(
                    "speech_embedding_ridge_pca_target",
                    pred_pca,
                    true_r,
                    speech,
                    singing,
                    mu,
                    test_idx,
                    labels,
                    global_mse,
                    alpha_pca,
                    k_pca,
                    {"dev_residual_MSE": dev_pca, "input_features": "speech_embedding"},
                )
            )
            pred_combo_pca, alpha_combo_pca, k_combo_pca, dev_combo_pca = fit_pca_ridge_model(
                speech_acoustic, true_r, train_idx, dev_idx
            )
            rows_for_split.append(
                eval_prediction(
                    "speech_embedding_plus_acoustic_ridge_pca_target",
                    pred_combo_pca,
                    true_r,
                    speech,
                    singing,
                    mu,
                    test_idx,
                    labels,
                    global_mse,
                    alpha_combo_pca,
                    k_combo_pca,
                    {"dev_residual_MSE": dev_combo_pca, "input_features": "speech_embedding_plus_acoustic"},
                )
            )

            rng = np.random.default_rng(seed)
            wrong_metrics = []
            random_metrics = []
            train_r = true_r[train_idx]
            cov_diag = train_r.var(axis=0)
            for draw in range(args.control_draws):
                wrong = np.zeros_like(true_r)
                wrong[test_idx] = train_r[rng.integers(0, len(train_r), size=len(test_idx))]
                wrong_metrics.append(
                    eval_prediction(
                        "wrong_speaker_residual",
                        wrong,
                        true_r,
                        speech,
                        singing,
                        mu,
                        test_idx,
                        labels,
                        global_mse,
                        "",
                        "",
                        {"draw": draw},
                    )
                )
                rand = np.zeros_like(true_r)
                rand[test_idx] = rng.normal(0.0, np.sqrt(np.maximum(cov_diag, 1e-12)), size=(len(test_idx), train_r.shape[1]))
                random_metrics.append(
                    eval_prediction(
                        "random_residual",
                        rand,
                        true_r,
                        speech,
                        singing,
                        mu,
                        test_idx,
                        labels,
                        global_mse,
                        "",
                        "",
                        {"draw": draw},
                    )
                )
            for name, metrics in [("wrong_speaker_residual", wrong_metrics), ("random_residual", random_metrics)]:
                avg = {key: np.mean([float(row[key]) for row in metrics]) for key in metrics[0] if key not in {"mapper", "draw"} and isinstance(metrics[0][key], (int, float, np.floating))}
                rows_for_split.append({"mapper": name, **avg})

            for shuffle in range(args.shuffle_draws):
                perm = train_idx.copy()
                rng.shuffle(perm)
                y_shuffle = true_r.copy()
                y_shuffle[train_idx] = true_r[perm]
                pred_shuffle, alpha_shuffle, dev_shuffle = fit_ridge_model(speech, y_shuffle, train_idx, dev_idx)
                rows_for_split.append(
                    eval_prediction(
                        "shuffle_label_ridge",
                        pred_shuffle,
                        true_r,
                        speech,
                        singing,
                        mu,
                        test_idx,
                        labels,
                        global_mse,
                        alpha_shuffle,
                        "",
                        {"shuffle": shuffle, "dev_residual_MSE": dev_shuffle, "input_features": "speech_embedding"},
                    )
                )

            for row in rows_for_split:
                row = {**common, **row}
                row["beats_global_MSE"] = bool(float(row["residual_MSE_reduction_vs_global"]) > 0)
                row["beats_global_cosine"] = bool(float(row["delta_cos_vs_global"]) > 0)
                per_split_rows.append(row)
    write_csv(per_split_rows, out_dir / "per_split.csv")
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_split_rows:
        grouped[(row["model"], row["layer"], row["mapper"])].append(row)
    for (model, layer, mapper), items in grouped.items():
        summary_rows.append(
            {
                "dataset": args.dataset,
                "model": model,
                "layer": layer,
                "mapper": mapper,
                "splits": len(items),
                "predicted_g_R1_mean": float(np.mean([float(row["predicted_g_R1"]) for row in items])),
                "predicted_g_R1_CI_low": float(np.percentile([float(row["predicted_g_R1"]) for row in items], 2.5)),
                "predicted_g_R1_CI_high": float(np.percentile([float(row["predicted_g_R1"]) for row in items], 97.5)),
                "cosine_to_true_g_mean": float(np.mean([float(row["cosine_to_true_g"]) for row in items])),
                "delta_cos_vs_global_mean": float(np.mean([float(row["delta_cos_vs_global"]) for row in items])),
                "residual_MSE_mean": float(np.mean([float(row["residual_MSE"]) for row in items])),
                "residual_MSE_reduction_vs_global_mean": float(
                    np.mean([float(row["residual_MSE_reduction_vs_global"]) for row in items])
                ),
                "residual_MSE_reduction_vs_global_CI_low": float(
                    np.percentile([float(row["residual_MSE_reduction_vs_global"]) for row in items], 2.5)
                ),
                "residual_MSE_reduction_vs_global_CI_high": float(
                    np.percentile([float(row["residual_MSE_reduction_vs_global"]) for row in items], 97.5)
                ),
            }
        )
    write_csv(summary_rows, out_dir / "summary.csv")
    write_yaml(
        {
            "experiment_name": "expB_mapper_eval",
            "dataset": args.dataset,
            "manifest": str(args.manifest),
            "feature_root": str(args.feature_root),
            "run_root": str(args.run_root),
            "hostname": socket.gethostname(),
            "git_commit": current_git_commit(),
            "split_protocol": args.split_protocol,
            "cache_root": os.environ.get("XDG_CACHE_HOME", ""),
            "target": "global_plus_person_residual",
            "primary_baseline": "global_mean_residual",
            "known_limitations": "GTSinger is exploratory; fixed hyperparameters when no dev speakers exist.",
        },
        out_dir / "experiment_card.yaml",
    )
    (out_dir / "README_results.md").write_text(
        "# Experiment B2 Mapper Evaluation\n\n"
        "Held-out-speaker mapper evaluation against the mandatory global mean residual baseline. Inputs use speech-side information only.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict B2 mapper evaluation for identity residuals.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models", nargs="+", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--split-protocol", choices=["jvs_60_20_20", "gtsinger_10_0_10"], required=True)
    parser.add_argument("--max-speech-per-speaker", type=int, default=20)
    parser.add_argument("--duration-balanced", action="store_true")
    parser.add_argument("--control-draws", type=int, default=50)
    parser.add_argument("--shuffle-draws", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_mapper(args)
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote mapper outputs -> {args.results_dir / 'expB_mapper_eval'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
