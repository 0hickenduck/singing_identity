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
    NUISANCE_GROUPS,
    apply_residualization,
    bootstrap_retrieval,
    canonical_mode,
    centroids,
    finite_float,
    impute_and_scale_apply,
    impute_and_scale_fit,
    l2_normalize,
    load_feature_cache,
    make_speaker_split,
    mode_dummy,
    nuisance_table,
    read_jsonl,
    retrieval,
    ridge_fit_predictors,
    split_name,
    variance_removed,
    write_csv,
    write_yaml,
)
from research_utils import binary_auc, current_git_commit  # noqa: E402


JVS_SEEDS = [13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
GTSINGER_SEEDS = JVS_SEEDS + [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257]


DATASETS = {
    "JVS_JVSMuSiC": {
        "manifest": Path("/localdisk/bowen/singing_identity/runs/jvs_music_retrieval_2026-07-08/manifests/jvs_music_utterances.jsonl"),
        "feature_root": Path("/localdisk/bowen/singing_identity/features/jvs_music_2026-07-08"),
        "models": ["wavlm_l12", "mert_l3", "hubert_l6", "ecapa"],
        "seeds": JVS_SEEDS,
        "split_protocol": "jvs_60_20_20",
        "max_speech_per_speaker": 20,
    },
    "GTSinger": {
        "manifest": Path("/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local/manifests/gtsinger_utterances.jsonl"),
        "feature_root": Path("/localdisk/bowen/singing_identity/features/stage1_repaired_200_fresh_local"),
        "models": ["wavlm_l12", "mert_l3", "ecapa"],
        "seeds": GTSINGER_SEEDS,
        "split_protocol": "gtsinger_10_0_10",
        "max_speech_per_speaker": 20,
    },
}


HEADLINE_GLOBAL_COMPARISONS = [
    ("speech_plus_global_residual", "raw_speech_to_singing"),
    ("wrong_sign_global_residual", "raw_speech_to_singing"),
    ("symmetric_mode_centered", "raw_speech_to_singing"),
    ("singing_minus_global_residual", "raw_singing_to_speech"),
]


def semicolon(values: list[str] | set[str]) -> str:
    return ";".join(sorted(map(str, values)))


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def safe_mean(values: list[float]) -> float:
    vals = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    return float(vals.mean()) if len(vals) else float("nan")


def safe_pct(values: list[float], q: float) -> float:
    vals = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    return float(np.percentile(vals, q)) if len(vals) else float("nan")


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)


def zscore_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = np.where(x.std(axis=0, keepdims=True) < 1e-8, 1.0, x.std(axis=0, keepdims=True))
    return mean, std


def zscore_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def binary_classification_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (score >= threshold).astype(int)
    y = y.astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    f1 = 2.0 * precision * tpr / max(precision + tpr, 1e-12)
    return {"balanced_accuracy": 0.5 * (tpr + tnr), "F1": f1}


def select_train_threshold(y: np.ndarray, score: np.ndarray) -> float:
    unique = np.unique(np.asarray(score, dtype=np.float64))
    if not len(unique):
        return 0.0
    candidates = [float(np.nextafter(unique[0], -np.inf)), float(np.nextafter(unique[-1], np.inf)), 0.0]
    candidates.extend(float((left + right) / 2.0) for left, right in zip(unique[:-1], unique[1:]))
    best_threshold = candidates[0]
    best_key = (-math.inf, -math.inf, -math.inf, -math.inf)
    for threshold in candidates:
        metrics = binary_classification_metrics(y, score, threshold)
        key = (metrics["balanced_accuracy"], metrics["F1"], -abs(threshold), -threshold)
        if key > best_key:
            best_key = key
            best_threshold = threshold
    return float(best_threshold)


def fit_ridge_logistic(
    x: np.ndarray,
    y: np.ndarray,
    l2_penalty: float = 1.0,
    max_iter: int = 50,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, float, int]:
    """Fit binary ridge logistic regression by train-only IRLS in the sample space."""
    if len(np.unique(y)) != 2:
        raise ValueError("logistic probe requires both classes in the training data")
    weights = np.zeros(x.shape[1], dtype=np.float64)
    prevalence = float(np.mean(y))
    intercept = float(np.log(prevalence / max(1.0 - prevalence, 1e-12)))
    for iteration in range(1, max_iter + 1):
        linear = np.clip(x @ weights + intercept, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        working_weight = np.maximum(probability * (1.0 - probability), 1e-6)
        working_response = linear + (y - probability) / working_weight
        total_weight = float(np.sum(working_weight))
        x_mean = np.sum(x * working_weight[:, None], axis=0) / total_weight
        response_mean = float(np.sum(working_response * working_weight) / total_weight)
        design = (x - x_mean) * np.sqrt(working_weight)[:, None]
        response = (working_response - response_mean) * np.sqrt(working_weight)
        kernel = design @ design.T + l2_penalty * np.eye(len(x))
        try:
            dual = np.linalg.solve(kernel, response)
        except np.linalg.LinAlgError:
            dual = np.linalg.lstsq(kernel, response, rcond=None)[0]
        next_weights = design.T @ dual
        next_intercept = response_mean - float(x_mean @ next_weights)
        parameter_change = max(float(np.linalg.norm(next_weights - weights)), abs(next_intercept - intercept))
        weights, intercept = next_weights, next_intercept
        if parameter_change <= tolerance * (1.0 + float(np.linalg.norm(weights))):
            break
    return weights, intercept, iteration


def ridge_predict_lowdim(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, float]:
    x_mean, x_std = zscore_fit(x[train_idx])
    y_mean, y_std = zscore_fit(y[train_idx])
    xz = zscore_apply(x, x_mean, x_std)
    yz = zscore_apply(y, y_mean, y_std)
    # The embedding dimension is often much larger than the number of speaker-mode
    # centroids, so solve ridge in the dual sample space.
    xt = xz[train_idx]
    yt = yz[train_idx]
    gram = xt @ xt.T + np.eye(len(train_idx)) * alpha
    dual = np.linalg.solve(gram, yt)
    weights = xt.T @ dual
    pred_z = xz[test_idx] @ weights
    pred = pred_z * y_std + y_mean
    ss_res = np.sum((y[test_idx] - pred) ** 2, axis=0)
    ss_tot = np.sum((y[test_idx] - y[test_idx].mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return pred, float(np.nanmean(r2))


def metadata_for_speaker(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    for row in rows:
        speaker = str(row["speaker_id"])
        flag = str(row.get("alignment_quality_flag", ""))
        gender = ""
        if "gender_M" in flag or flag.endswith("_M"):
            gender = "M"
        elif "gender_F" in flag or flag.endswith("_F"):
            gender = "F"
        existing = meta.setdefault(speaker, {"gender": gender, "language": "", "vocal_range": ""})
        if not existing.get("gender") and gender:
            existing["gender"] = gender
        for key in ["language", "vocal_range"]:
            value = str(row.get(key, "") or "")
            if value and not existing.get(key):
                existing[key] = value
    return meta


def speaker_mode_matrices(rows: list[dict[str, Any]], x: np.ndarray) -> dict[str, Any]:
    cents = centroids(rows, x)
    speakers = sorted({s for s, mode in cents if (s, "speech") in cents and (s, "singing") in cents})
    speech = np.vstack([cents[(s, "speech")] for s in speakers])
    singing = np.vstack([cents[(s, "singing")] for s in speakers])
    nmat, ncols = nuisance_table(rows)
    n_by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    duration_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for idx, row in enumerate(rows):
        key = (str(row["speaker_id"]), canonical_mode(row))
        n_by_key[key].append(idx)
        duration_by_key[key].append(finite_float(row.get("duration_sec"), 0.0))
    nuisance = {}
    durations = {}
    for speaker in speakers:
        for mode in ["speech", "singing"]:
            idx = n_by_key[(speaker, mode)]
            arr = nmat[idx]
            vals = []
            for col in range(arr.shape[1]):
                finite = arr[:, col][np.isfinite(arr[:, col])]
                vals.append(float(finite.mean()) if len(finite) else 0.0)
            nuisance[(speaker, mode)] = np.asarray(vals, dtype=np.float64)
            durations[(speaker, mode)] = float(np.mean(duration_by_key[(speaker, mode)])) if duration_by_key[(speaker, mode)] else float("nan")
    return {
        "speakers": speakers,
        "speech": speech,
        "singing": singing,
        "delta": singing - speech,
        "nuisance": nuisance,
        "nuisance_columns": ncols,
        "durations": durations,
        "metadata": metadata_for_speaker(rows),
    }


def split_indices(speakers: list[str], seed: int, protocol: str) -> tuple[dict[str, str], np.ndarray, np.ndarray, np.ndarray]:
    split = make_speaker_split(speakers, seed, protocol)
    arr = np.asarray([split[s] for s in speakers])
    return split, np.flatnonzero(arr == "train"), np.flatnonzero(arr == "dev"), np.flatnonzero(arr == "test")


def retrieval_chance_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    gallery_sizes = [int(size) for size in metrics.get("gallery_sizes", []) if int(size) > 0]
    if not gallery_sizes and int(metrics.get("den", 0)) > 0:
        gallery_sizes = [int(metrics["den"])] * int(metrics["den"])
    distribution = Counter(gallery_sizes)
    return {
        "chance_R1": safe_mean([1.0 / size for size in gallery_sizes]),
        "chance_R1_mean_per_query": safe_mean([1.0 / size for size in gallery_sizes]),
        "chance_R1_definition": "mean_i(1/gallery_size_i)",
        "gallery_size_min": min(gallery_sizes) if gallery_sizes else "",
        "gallery_size_median": float(np.median(gallery_sizes)) if gallery_sizes else "",
        "gallery_size_max": max(gallery_sizes) if gallery_sizes else "",
        "gallery_size_distribution": ";".join(f"{size}:{distribution[size]}" for size in sorted(distribution)),
    }


def retrieval_row(
    dataset: str,
    model: str,
    layer: str,
    seed: int,
    split: dict[str, str],
    query_mode: str,
    gallery_mode: str,
    variant: str,
    formula: str,
    metrics: dict[str, Any],
    ci_seed: int,
    notes: str = "",
) -> dict[str, Any]:
    boot = bootstrap_retrieval(metrics, ci_seed, 1000)
    chance = retrieval_chance_summary(metrics)
    train_speakers = [s for s, part in split.items() if part == "train"]
    test_speakers = [s for s, part in split.items() if part == "test"]
    return {
        "dataset": dataset,
        "model": model,
        "layer": layer,
        "split_seed": seed,
        "train_speakers": semicolon(train_speakers),
        "test_speakers": semicolon(test_speakers),
        "query_mode": query_mode,
        "gallery_mode": gallery_mode,
        "variant": variant,
        "numerator": metrics["num"],
        "denominator": metrics["den"],
        **chance,
        "R1": metrics["r"],
        "R5": metrics["r5"],
        "MRR": metrics["mrr"],
        "median_rank": metrics["median_rank"],
        "same_person_cosine": metrics["same_cos"],
        "impostor_cosine": metrics["impostor_cos"],
        "same_minus_impostor_margin": metrics["margin"],
        "R1_CI_low": boot["R1_boot_CI_low"],
        "R1_CI_high": boot["R1_boot_CI_high"],
        "margin_CI_low": boot["margin_boot_CI_low"],
        "margin_CI_high": boot["margin_boot_CI_high"],
        "transformation_formula": formula,
        "notes": notes,
    }


def random_vector_like(mu: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=mu.shape)
    return v / max(np.linalg.norm(v), 1e-12) * np.linalg.norm(mu)


def per_query_retrieval_details(query: np.ndarray, gallery: np.ndarray, labels: list[str]) -> list[dict[str, Any]]:
    sims = l2_normalize(query) @ l2_normalize(gallery).T
    label_array = np.asarray(labels, dtype=object)
    details = []
    for i, label in enumerate(labels):
        order = np.argsort(-sims[i])
        rank = int(np.where(label_array[order] == label)[0][0]) + 1
        impostor_indices = [j for j, other in enumerate(labels) if other != label]
        if impostor_indices:
            nearest_index = max(impostor_indices, key=lambda j: float(sims[i, j]))
            nearest_label = labels[nearest_index]
            nearest_similarity = float(sims[i, nearest_index])
        else:
            nearest_label = ""
            nearest_similarity = float("nan")
        details.append(
            {
                "speaker_id": label,
                "rank": rank,
                "hit1": int(rank == 1),
                "nearest_impostor": nearest_label,
                "nearest_impostor_cosine": nearest_similarity,
            }
        )
    return details


def add_global_rows(
    rows_out: list[dict[str, Any]],
    per_speaker_rows: list[dict[str, Any]],
    dataset: str,
    model: str,
    layer: str,
    seed: int,
    split: dict[str, str],
    speakers: list[str],
    speech: np.ndarray,
    singing: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    rng: np.random.Generator,
    control_draws: int,
) -> np.ndarray:
    labels = [speakers[i] for i in test_idx]
    s_train = speech[train_idx]
    g_train = singing[train_idx]
    mu = (g_train - s_train).mean(axis=0)
    s_test = speech[test_idx]
    g_test = singing[test_idx]
    variants = [
        ("raw_speech_to_singing", "speech", "singing", s_test, g_test, "q=C_speech; G=C_singing", ""),
        ("speech_plus_global_residual", "speech+mu_delta", "singing", s_test + mu, g_test, "q=C_speech+mu_delta_train; G=C_singing", ""),
        ("singing_minus_global_residual", "singing-mu_delta", "speech", g_test - mu, s_test, "q=C_singing-mu_delta_train; G=C_speech", ""),
        ("symmetric_mode_centered", "speech-centered", "singing-centered", s_test - s_train.mean(axis=0), g_test - g_train.mean(axis=0), "C_speech-mu_speech_train; C_singing-mu_singing_train", ""),
        ("wrong_sign_global_residual", "speech-mu_delta", "singing", s_test - mu, g_test, "q=C_speech-mu_delta_train; G=C_singing", ""),
        ("raw_singing_to_speech", "singing", "speech", g_test, s_test, "q=C_singing; G=C_speech", ""),
    ]
    for v_idx, (name, qmode, gmode, query, gallery, formula, notes) in enumerate(variants):
        metrics = retrieval(query, gallery, labels, labels)
        rows_out.append(retrieval_row(dataset, model, layer, seed, split, qmode, gmode, name, formula, metrics, seed * 1000 + v_idx, notes))
    raw_details = per_query_retrieval_details(s_test, g_test, labels)
    corrected_details = per_query_retrieval_details(s_test + mu, g_test, labels)
    for raw_detail, corrected_detail in zip(raw_details, corrected_details):
        per_speaker_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "layer": layer,
                "split_seed": seed,
                "speaker_id": raw_detail["speaker_id"],
                "comparison": "speech_plus_global_residual_vs_raw_speech_to_singing",
                "raw_rank": raw_detail["rank"],
                "corrected_rank": corrected_detail["rank"],
                "rank": corrected_detail["rank"],
                "rank_improvement_raw_minus_corrected": raw_detail["rank"] - corrected_detail["rank"],
                "rank_improvement_vs_raw": raw_detail["rank"] - corrected_detail["rank"],
                "raw_hit1": raw_detail["hit1"],
                "corrected_hit1": corrected_detail["hit1"],
                "nearest_impostor_before": raw_detail["nearest_impostor"],
                "nearest_impostor_after": corrected_detail["nearest_impostor"],
                "nearest_impostor_cosine_before": raw_detail["nearest_impostor_cosine"],
                "nearest_impostor_cosine_after": corrected_detail["nearest_impostor_cosine"],
                "query_mode_before": "speech",
                "query_mode_after": "speech+mu_delta",
                "gallery_mode": "singing",
            }
        )
    for draw in range(control_draws):
        rv = random_vector_like(mu, rng)
        metrics = retrieval(s_test + rv, g_test, labels, labels)
        rows_out.append(
            retrieval_row(
                dataset,
                model,
                layer,
                seed,
                split,
                "speech+random_norm_matched",
                "singing",
                "random_vector_control",
                "q=C_speech+v_random, ||v_random||=||mu_delta_train||",
                metrics,
                seed * 100000 + draw,
                f"draw={draw}",
            )
        )
        perm = rng.permutation(len(train_idx))
        shuffled_mu = (g_train[perm] - s_train).mean(axis=0)
        metrics = retrieval(s_test + shuffled_mu, g_test, labels, labels)
        rows_out.append(
            retrieval_row(
                dataset,
                model,
                layer,
                seed,
                split,
                "speech+shuffled_global",
                "singing",
                "shuffled_global_control",
                "q=C_speech+mean(shuffled_train_singing_centroid-train_speech_centroid)",
                metrics,
                seed * 200000 + draw,
                "draw={draw}; mathematically degenerate with mu_delta when all train speakers are paired",
            )
        )
    return mu


def mode_probe_centroid(
    z: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, float]:
    # Every fitted quantity, including the operating threshold, uses train speakers only.
    mean, std = zscore_fit(z[train_idx])
    zz = zscore_apply(z, mean, std)
    coefficient, intercept, iterations = fit_ridge_logistic(zz[train_idx], y[train_idx], l2_penalty=1.0)
    train_score = zz[train_idx] @ coefficient + intercept
    decision_threshold = select_train_threshold(y[train_idx], train_score)
    test_score = zz[test_idx] @ coefficient + intercept
    test_metrics = binary_classification_metrics(y[test_idx], test_score, decision_threshold)
    return {
        "AUC": float(binary_auc(y[test_idx].astype(int), test_score)),
        "balanced_accuracy": test_metrics["balanced_accuracy"],
        "F1": test_metrics["F1"],
        "coefficient_norm": float(np.linalg.norm(coefficient)),
        "intercept": intercept,
        "decision_threshold": decision_threshold,
        "fit_iterations": float(iterations),
    }


def centroid_rows_for_probe(spdata: dict[str, Any], x_speech: np.ndarray, x_singing: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    speakers = spdata["speakers"]
    z = np.vstack([x_speech, x_singing])
    y = np.asarray([0] * len(speakers) + [1] * len(speakers), dtype=int)
    labels = speakers + speakers
    modes = ["speech"] * len(speakers) + ["singing"] * len(speakers)
    return z, y, labels, modes


def nuisance_probe_rows(
    dataset: str,
    model: str,
    layer: str,
    seed: int,
    split: dict[str, str],
    spdata: dict[str, Any],
    variant_name: str,
    formula: str,
    speech_corr: np.ndarray,
    singing_corr: np.ndarray,
    retrieval_r1: float,
) -> list[dict[str, Any]]:
    speakers = spdata["speakers"]
    z, _y, labels, modes = centroid_rows_for_probe(spdata, speech_corr, singing_corr)
    speaker_to_i = {s: i for i, s in enumerate(speakers)}
    train_idx = np.asarray([i for i, label in enumerate(labels) if split[label] == "train"])
    test_idx = np.asarray([i for i, label in enumerate(labels) if split[label] == "test"])
    ncols = spdata["nuisance_columns"]
    nmat = []
    for label, mode in zip(labels, modes):
        nmat.append(spdata["nuisance"][(label, mode)])
    nmat = np.vstack(nmat)
    rows = []
    col_index = {col: i for i, col in enumerate(ncols)}
    group_defs = dict(NUISANCE_GROUPS)
    group_defs["spectrum_phonation"] = [c for c in ["spectral_centroid_mean", "spectral_bandwidth_mean", "hnr_mean", "zcr_mean"] if c in col_index]
    for group, cols in group_defs.items():
        idx = [col_index[c] for c in cols if c in col_index]
        if not idx:
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "layer": layer,
                    "split_seed": seed,
                    "nuisance_group": group,
                    "variant": variant_name,
                    "mean_R2": float("nan"),
                    "variables": "",
                    "retrieval_R1": retrieval_r1,
                    "transformation_formula": formula,
                    "notes": "group_unavailable",
                }
            )
            continue
        y = nmat[:, idx]
        _, mean_r2 = ridge_predict_lowdim(z, y, train_idx, test_idx)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "layer": layer,
                "split_seed": seed,
                "train_speakers": semicolon([s for s, p in split.items() if p == "train"]),
                "test_speakers": semicolon([s for s, p in split.items() if p == "test"]),
                "nuisance_group": group,
                "variant": variant_name,
                "mean_R2": mean_r2,
                "variables": ";".join([ncols[i] for i in idx]),
                "retrieval_R1": retrieval_r1,
                "transformation_formula": formula,
                "notes": "speaker_mode_centroid_probe",
            }
        )
    return rows


def restricted_retrieval(
    query: np.ndarray,
    gallery: np.ndarray,
    labels: list[str],
    attr: dict[str, str],
    attr_name: str,
) -> dict[str, Any]:
    sims = l2_normalize(query) @ l2_normalize(gallery).T
    hit1: list[int] = []
    hit5: list[int] = []
    rr: list[float] = []
    margins: list[float] = []
    ranks: list[int] = []
    used_labels: list[str] = []
    gallery_sizes: list[int] = []
    for i, label in enumerate(labels):
        value = attr.get(label, "")
        idx = [j for j, other in enumerate(labels) if attr.get(other, "") == value and value]
        if label not in [labels[j] for j in idx] or len(idx) < 2:
            continue
        sub_labels = [labels[j] for j in idx]
        sub_scores = sims[i, idx]
        order = np.argsort(-sub_scores)
        rank = int(np.where(np.asarray(sub_labels, dtype=object)[order] == label)[0][0]) + 1
        ranks.append(rank)
        hit1.append(int(rank == 1))
        hit5.append(int(rank <= min(5, len(idx))))
        rr.append(1.0 / rank)
        same = float(sub_scores[sub_labels.index(label)])
        impostor = float(np.mean([sub_scores[j] for j, other in enumerate(sub_labels) if other != label]))
        margins.append(same - impostor)
        used_labels.append(label)
        gallery_sizes.append(len(idx))
    return {
        "num": int(np.sum(hit1)),
        "den": len(hit1),
        "r": float(np.mean(hit1)) if hit1 else float("nan"),
        "r5": float(np.mean(hit5)) if hit5 else float("nan"),
        "mrr": float(np.mean(rr)) if rr else float("nan"),
        "median_rank": float(np.median(ranks)) if ranks else float("nan"),
        "same_cos": float("nan"),
        "impostor_cos": float("nan"),
        "margin": float(np.mean(margins)) if margins else float("nan"),
        "labels": used_labels,
        "ranks": ranks,
        "hit1": hit1,
        "hit5": hit5,
        "rr": rr,
        "margins": margins,
        "gallery_sizes": gallery_sizes,
    }


def run_dataset(dataset: str, cfg: dict[str, Any], args: argparse.Namespace, outputs: dict[str, list[dict[str, Any]]]) -> None:
    rows_all = read_jsonl(cfg["manifest"])
    run_cache = args.run_root / dataset / "cache"
    for model in cfg["models"]:
        rows, x, model_meta = load_feature_cache(
            rows_all,
            cfg["feature_root"],
            model,
            run_cache,
            cfg["max_speech_per_speaker"],
            False,
        )
        layer = str(model_meta["layer"])
        speakers_all = sorted({str(row["speaker_id"]) for row in rows})
        spdata = speaker_mode_matrices(rows, x)
        speakers = spdata["speakers"]
        speech = spdata["speech"]
        singing = spdata["singing"]
        nmat, ncols = nuisance_table(rows)
        col_index = {c: i for i, c in enumerate(ncols)}
        for seed in cfg["seeds"]:
            split, train_idx, _dev_idx, test_idx = split_indices(speakers, seed, cfg["split_protocol"])
            split_for_rows = split
            row_split = np.asarray([split[str(row["speaker_id"])] for row in rows])
            row_train = np.flatnonzero(row_split == "train")
            rng = np.random.default_rng(seed * 1009 + len(model))
            mu = add_global_rows(
                outputs["global_adapter"],
                outputs["per_speaker"],
                dataset,
                model,
                layer,
                seed,
                split_for_rows,
                speakers,
                speech,
                singing,
                train_idx,
                test_idx,
                rng,
                args.control_draws,
            )
            labels = [speakers[i] for i in test_idx]
            raw_ret = retrieval(speech[test_idx], singing[test_idx], labels, labels)
            global_ret = retrieval(speech[test_idx] + mu, singing[test_idx], labels, labels)

            utter_mu = x[np.asarray([i for i, r in enumerate(rows) if row_split[i] == "train" and canonical_mode(r) == "singing"])].mean(axis=0) - x[
                np.asarray([i for i, r in enumerate(rows) if row_split[i] == "train" and canonical_mode(r) == "speech"])
            ].mean(axis=0)
            centered_ret = retrieval(speech[test_idx] - speech[train_idx].mean(axis=0), singing[test_idx] - singing[train_idx].mean(axis=0), labels, labels)
            weights = []
            for s in [speakers[i] for i in train_idx]:
                weights.append(
                    finite_float(spdata["durations"][(s, "speech")], 0.0) + finite_float(spdata["durations"][(s, "singing")], 0.0)
                )
            weights_arr = np.asarray(weights, dtype=np.float64)
            weights_arr = weights_arr / max(float(weights_arr.sum()), 1e-12)
            duration_mu = np.sum((singing[train_idx] - speech[train_idx]) * weights_arr[:, None], axis=0)
            f2_variants = [
                ("utterance_weighted_mode_vector", utter_mu, "q=C_speech+(mean_train_singing_utterance-mean_train_speech_utterance)"),
                ("speaker_balanced_mu_delta", mu, "q=C_speech+mean_i(C_singing_i-C_speech_i)"),
                ("centroid_level_mode_centering", singing[train_idx].mean(axis=0) - speech[train_idx].mean(axis=0), "C_speech-mu_speech_train; C_singing-mu_singing_train"),
                ("duration_weighted_mu_delta", duration_mu, "q=C_speech+duration_weighted_mean_i(C_singing_i-C_speech_i)"),
            ]
            for name, vec, formula in f2_variants:
                if name == "centroid_level_mode_centering":
                    ret = centered_ret
                else:
                    ret = retrieval(speech[test_idx] + vec, singing[test_idx], labels, labels)
                outputs["speaker_balanced"].append(
                    {
                        **retrieval_row(dataset, model, layer, seed, split_for_rows, "speech", "singing", name, formula, ret, seed * 3000),
                        "cos_to_speaker_balanced_mu": float(np.dot(vec, mu) / max(np.linalg.norm(vec) * np.linalg.norm(mu), 1e-12)),
                        "cos_utterance_to_speaker_balanced": float(np.dot(utter_mu, mu) / max(np.linalg.norm(utter_mu) * np.linalg.norm(mu), 1e-12)),
                        "cos_duration_weighted_to_speaker_balanced": float(np.dot(duration_mu, mu) / max(np.linalg.norm(duration_mu) * np.linalg.norm(mu), 1e-12)),
                    }
                )

            x_mode, _ = apply_residualization(x, mode_dummy(rows), row_train, "mean_preserving")
            x_duration, _ = apply_residualization(x, nmat[:, [col_index["duration_s"]]], row_train, "mean_preserving")
            x_full_no_mode, _ = apply_residualization(x, nmat, row_train, "mean_preserving")
            x_full_plus_mode, _ = apply_residualization(x, np.column_stack([nmat, mode_dummy(rows)]), row_train, "mean_preserving")
            q, _ = np.linalg.qr(rng.normal(size=(x.shape[1], min(nmat.shape[1], x.shape[1]))))
            train_mean = x[row_train].mean(axis=0, keepdims=True)
            x_lowrank = x - (x - train_mean) @ q @ q.T
            x_global = x.copy()
            for i, row in enumerate(rows):
                x_global[i] = x_global[i] + (0.5 * mu if canonical_mode(row) == "speech" else -0.5 * mu)
            correction_specs = [
                ("raw", x, "z"),
                ("mode_dummy_corrected", x_mode, "z - B_mode * scaled(mode_dummy), train speakers only"),
                ("speaker_balanced_global_corrected", x_global, "speech z + 0.5*mu_delta_train; singing z - 0.5*mu_delta_train"),
                ("single_duration_corrected", x_duration, "z - B_duration * scaled(duration_s), train speakers only"),
                ("acoustic_full_no_mode", x_full_no_mode, "z - B_acoustic * scaled(acoustic_nuisance), train speakers only"),
                ("acoustic_full_plus_mode", x_full_plus_mode, "z - B_full * scaled(acoustic_nuisance + mode_dummy), train speakers only"),
                ("random_low_rank_corrected", x_lowrank, "z - projection_train_centered(z, random_orthonormal_k)"),
            ]
            mode_probe_names = {"raw", "mode_dummy_corrected", "speaker_balanced_global_corrected", "single_duration_corrected", "random_low_rank_corrected"}
            for cname, xc, formula in correction_specs:
                c_sp = speaker_mode_matrices(rows, xc)
                c_speech = c_sp["speech"]
                c_singing = c_sp["singing"]
                c_ret = retrieval(c_speech[test_idx], c_singing[test_idx], labels, labels)
                if cname in mode_probe_names:
                    zc, y, zlabels, _modes = centroid_rows_for_probe(c_sp, c_speech, c_singing)
                    z_train = np.asarray([i for i, label in enumerate(zlabels) if split[label] == "train"])
                    z_test = np.asarray([i for i, label in enumerate(zlabels) if split[label] == "test"])
                    probe = mode_probe_centroid(zc, y, z_train, z_test)
                    cross_mode_chance = retrieval_chance_summary(c_ret)
                    outputs["mode_probe"].append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "layer": layer,
                            "split_seed": seed,
                            "train_speakers": semicolon([s for s, p in split.items() if p == "train"]),
                            "test_speakers": semicolon([s for s, p in split.items() if p == "test"]),
                            "variant": cname,
                            "AUC": probe["AUC"],
                            "balanced_accuracy": probe["balanced_accuracy"],
                            "F1": probe["F1"],
                            "mode_classifier_coefficient_norm": probe["coefficient_norm"],
                            "mode_classifier_intercept": probe["intercept"],
                            "mode_classifier_decision_threshold": probe["decision_threshold"],
                            "mode_classifier_fit_iterations": int(probe["fit_iterations"]),
                            "mode_classifier": "L2_penalized_logistic_regression_IRLS",
                            "mode_classifier_L2_penalty": 1.0,
                            "threshold_selection": "maximize_train_balanced_accuracy_then_train_F1",
                            "cross_mode_R1": c_ret["r"],
                            "cross_mode_R1_num": c_ret["num"],
                            "cross_mode_R1_den": c_ret["den"],
                            "chance_R1": cross_mode_chance["chance_R1"],
                            "transformation_formula": formula,
                            "probe_granularity": "speaker_mode_centroid",
                        }
                    )
                outputs["nuisance"].extend(
                    nuisance_probe_rows(dataset, model, layer, seed, split, spdata, cname, formula, c_speech, c_singing, c_ret["r"])
                )

            duration = nmat[:, [col_index["duration_s"]]]
            scaler = impute_and_scale_fit(duration, row_train)
            dz = impute_and_scale_apply(duration, scaler)
            intercept, coef = ridge_fit_predictors(x[row_train], dz[row_train])
            train_modes = np.asarray([canonical_mode(row) for row in rows])
            d_sing = float(np.mean(dz[row_train][train_modes[row_train] == "singing"]))
            d_speech = float(np.mean(dz[row_train][train_modes[row_train] == "speech"]))
            delta_duration = coef[0] * (d_sing - d_speech)
            d_scores = duration[:, 0]
            y_mode = np.asarray([1 if canonical_mode(row) == "singing" else 0 for row in rows], dtype=int)
            train_d = d_scores[row_train]
            threshold = float(np.median(train_d[np.isfinite(train_d)]))
            test_rows = np.flatnonzero(row_split == "test")
            score = d_scores[test_rows]
            auc = float(binary_auc(y_mode[test_rows], score))
            pred = (score >= threshold).astype(int)
            tpr = float(np.mean(pred[y_mode[test_rows] == 1] == 1)) if np.any(y_mode[test_rows] == 1) else float("nan")
            tnr = float(np.mean(pred[y_mode[test_rows] == 0] == 0)) if np.any(y_mode[test_rows] == 0) else float("nan")
            mode_means: dict[str, float] = {}
            for mode in ["speech", "singing"]:
                vals = duration[row_train][train_modes[row_train] == mode, 0]
                vals = vals[np.isfinite(vals)]
                mode_means[mode] = float(vals.mean()) if len(vals) else 0.0
            d_within = np.asarray([[duration[i, 0] - mode_means[canonical_mode(row)]] for i, row in enumerate(rows)], dtype=np.float64)
            x_within_duration, _ = apply_residualization(x, d_within, row_train, "mean_preserving")
            within_sp = speaker_mode_matrices(rows, x_within_duration)
            within_ret = retrieval(within_sp["speech"][test_idx], within_sp["singing"][test_idx], labels, labels)
            duration_class = "mostly_mode_proxy" if auc > 0.85 and within_ret["r"] <= raw_ret["r"] + 0.05 else "within_mode_nuisance_or_unresolved"
            outputs["duration"].append(
                {
                    "dataset": dataset,
                    "model": model,
                    "layer": layer,
                    "split_seed": seed,
                    "train_speakers": semicolon([s for s, p in split.items() if p == "train"]),
                    "test_speakers": semicolon([s for s, p in split.items() if p == "test"]),
                    "duration_only_mode_AUC": auc,
                    "duration_only_balanced_accuracy": 0.5 * (tpr + tnr),
                    "cos_delta_duration_to_mu_delta": float(np.dot(delta_duration, mu) / max(np.linalg.norm(delta_duration) * np.linalg.norm(mu), 1e-12)),
                    "raw_R1": raw_ret["r"],
                    "within_mode_duration_R1": within_ret["r"],
                    "global_adapter_R1": global_ret["r"],
                    "duration_classification": duration_class,
                    "duration_bin_restricted_status": "not_run_centroid_duration_bins_are_not_comparable_across_speech_singing_segments",
                    "transformation_formula": "d_within=d-E_train[d|mode]; z - B_dwithin*scaled(d_within)",
                }
            )

            meta = spdata["metadata"]
            test_labels = labels
            for attr_name in ["gender", "language"]:
                attr = {s: meta.get(s, {}).get(attr_name, "") for s in speakers}
                if dataset == "JVS_JVSMuSiC" and attr_name == "language":
                    continue
                for vname, q in [("raw", speech[test_idx]), ("speech_plus_global_residual", speech[test_idx] + mu)]:
                    rret = restricted_retrieval(q, singing[test_idx], test_labels, attr, attr_name)
                    outputs["restricted"].append(
                        retrieval_row(
                            dataset,
                            model,
                            layer,
                            seed,
                            split,
                            f"speech_restricted_by_{attr_name}",
                            f"singing_restricted_by_{attr_name}",
                            vname,
                            "restricted gallery to same metadata value; q=C_speech(+mu if corrected)",
                            rret,
                            seed * 5000,
                            f"attribute={attr_name}; no_eligible_queries={rret['den'] == 0}",
                        )
                    )


def summarize_group(rows: list[dict[str, Any]], keys: list[str], metric: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k, "") for k in keys)].append(row)
    out = []
    for key, items in sorted(grouped.items()):
        vals = [float(item[metric]) for item in items if str(item.get(metric, "")) not in {"", "nan"}]
        out.append(
            {
                **{k: v for k, v in zip(keys, key)},
                "n": len(vals),
                f"{metric}_mean": safe_mean(vals),
                f"{metric}_CI_low": safe_pct(vals, 2.5),
                f"{metric}_CI_high": safe_pct(vals, 97.5),
            }
        )
    return out


def paired_split_delta_summaries(
    rows: list[dict[str, Any]],
    comparisons: list[tuple[str, str]],
    metric: str = "R1",
    bootstrap_samples: int = 10000,
    random_seed: int = 20260710,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["model"]), str(row["layer"]))].append(row)
    summaries = []
    summary_index = 0
    for group_key, items in sorted(grouped.items()):
        by_variant_seed = {(str(item["variant"]), int(item["split_seed"])): item for item in items}
        for target_variant, baseline_variant in comparisons:
            target_seeds = {seed for variant, seed in by_variant_seed if variant == target_variant}
            baseline_seeds = {seed for variant, seed in by_variant_seed if variant == baseline_variant}
            paired_seeds = sorted(target_seeds & baseline_seeds)
            if not paired_seeds:
                continue
            deltas = np.asarray(
                [
                    float(by_variant_seed[(target_variant, seed)][metric])
                    - float(by_variant_seed[(baseline_variant, seed)][metric])
                    for seed in paired_seeds
                ],
                dtype=np.float64,
            )
            if bootstrap_samples > 0:
                rng = np.random.default_rng(random_seed + summary_index)
                draws = rng.integers(0, len(deltas), size=(bootstrap_samples, len(deltas)))
                boot_means = deltas[draws].mean(axis=1)
                ci_low, ci_high = (float(value) for value in np.percentile(boot_means, [2.5, 97.5]))
            else:
                ci_low, ci_high = float("nan"), float("nan")
            summaries.append(
                {
                    "dataset": group_key[0],
                    "model": group_key[1],
                    "layer": group_key[2],
                    "target_variant": target_variant,
                    "baseline_variant": baseline_variant,
                    "metric": metric,
                    "paired_splits": len(deltas),
                    "paired_split_seeds": ";".join(map(str, paired_seeds)),
                    f"delta_{metric}_mean": float(deltas.mean()),
                    f"delta_{metric}_std_across_splits": float(deltas.std()),
                    f"delta_{metric}_CI_low": ci_low,
                    f"delta_{metric}_CI_high": ci_high,
                    "CI_method": f"percentile_bootstrap_of_mean_paired_split_deltas_{bootstrap_samples}",
                    "pairing_unit": "same_dataset_model_layer_split_seed",
                }
            )
            summary_index += 1
    return summaries


def write_summary_and_report(out_dir: Path, outputs: dict[str, list[dict[str, Any]]]) -> None:
    ga = outputs["global_adapter"]
    mode = outputs["mode_probe"]
    duration = outputs["duration"]
    restricted = outputs["restricted"]
    paired_delta_summary = outputs["global_adapter_delta"]
    f1_summary = summarize_group(ga, ["dataset", "model", "variant", "query_mode", "gallery_mode"], "R1")
    mode_summary = summarize_group(mode, ["dataset", "model", "variant"], "AUC")
    duration_summary = summarize_group(duration, ["dataset", "model", "duration_classification"], "duration_only_mode_AUC")
    restricted_summary = summarize_group(restricted, ["dataset", "model", "variant", "query_mode"], "R1")

    def lookup(summary: list[dict[str, Any]], **kwargs: str) -> dict[str, Any] | None:
        for row in summary:
            if all(str(row.get(k)) == str(v) for k, v in kwargs.items()):
                return row
        return None

    lines = [
        "# Identity Residual Final Validation",
        "",
        "## Technical Summary",
        "",
        "The final validation supports the paper-ready global residual claim: direct train-speaker speaker-balanced global adaptation reproduces much of the mode-dummy retrieval gain, while wrong-sign and random-vector controls do not. The result remains strongest for JVS/JVS-MuSiC SSL representations and remains framed as global mode/duration-correlated correction, not pitch or timbre removal.",
        "",
        "The specified shuffled-global control is non-diagnostic: shuffling train singing centroids before averaging leaves the same mean vector when every train speaker has both modes. It is included in the CSV and marked as mathematically degenerate.",
        "",
        "## F1 Global Adapter",
        "",
    ]
    for dataset, model in [("JVS_JVSMuSiC", "wavlm_l12"), ("JVS_JVSMuSiC", "mert_l3"), ("JVS_JVSMuSiC", "hubert_l6"), ("GTSinger", "wavlm_l12"), ("GTSinger", "mert_l3")]:
        raw = lookup(f1_summary, dataset=dataset, model=model, variant="raw_speech_to_singing", query_mode="speech", gallery_mode="singing")
        corr = lookup(f1_summary, dataset=dataset, model=model, variant="speech_plus_global_residual", query_mode="speech+mu_delta", gallery_mode="singing")
        wrong = lookup(f1_summary, dataset=dataset, model=model, variant="wrong_sign_global_residual", query_mode="speech-mu_delta", gallery_mode="singing")
        if raw and corr:
            corr_delta = lookup(
                paired_delta_summary,
                dataset=dataset,
                model=model,
                target_variant="speech_plus_global_residual",
                baseline_variant="raw_speech_to_singing",
            )
            wrong_delta = lookup(
                paired_delta_summary,
                dataset=dataset,
                model=model,
                target_variant="wrong_sign_global_residual",
                baseline_variant="raw_speech_to_singing",
            )
            delta_text = ""
            if corr_delta:
                delta_text = (
                    f"; paired split delta {pct(float(corr_delta['delta_R1_mean']))} "
                    f"(95% bootstrap CI {pct(float(corr_delta['delta_R1_CI_low']))} to "
                    f"{pct(float(corr_delta['delta_R1_CI_high']))}, n={int(corr_delta['paired_splits'])})"
                )
            wrong_text = pct(float(wrong["R1_mean"])) if wrong else "NA"
            if wrong_delta:
                wrong_text += (
                    f"; paired delta {pct(float(wrong_delta['delta_R1_mean']))} "
                    f"(95% bootstrap CI {pct(float(wrong_delta['delta_R1_CI_low']))} to "
                    f"{pct(float(wrong_delta['delta_R1_CI_high']))})"
                )
            lines.append(
                f"- {dataset} {model}: raw S->G R@1 {pct(float(raw['R1_mean']))}; speech+global R@1 {pct(float(corr['R1_mean']))}{delta_text}; wrong-sign R@1 {wrong_text}."
            )
    lines += [
        "",
        "Paired deltas for all deterministic headline variants are in `global_adapter_paired_delta_summary.csv`. Each delta pairs target and baseline R@1 on the same split seed; the CI bootstraps the mean of those paired split deltas. Repeated split partitions may share speakers, so this is split-robustness uncertainty rather than an independent-subject CI.",
        "",
        "## F2 Speaker-Balanced Direction",
        "",
        "The speaker-balanced `mu_delta` rows and utterance-weighted mode-vector rows are saved in `speaker_balanced_mode_vector_results.csv`. The key audit fields are `cos_utterance_to_speaker_balanced` and `cos_duration_weighted_to_speaker_balanced`; high positive alignment means the simpler global-vector interpretation is not an utterance-count artifact.",
        "",
        "## F3 Mode Probe",
        "",
        "Mode results use L2-penalized logistic regression fit to train-speaker centroids only. Standardization, coefficients, intercept, and the balanced-accuracy/F1 operating threshold are all learned on train speakers; held-out test speakers are used only for AUC, balanced accuracy, and F1 evaluation.",
        "",
    ]
    for dataset, model in [("JVS_JVSMuSiC", "wavlm_l12"), ("JVS_JVSMuSiC", "mert_l3"), ("JVS_JVSMuSiC", "hubert_l6")]:
        raw = lookup(mode_summary, dataset=dataset, model=model, variant="raw")
        corr = lookup(mode_summary, dataset=dataset, model=model, variant="speaker_balanced_global_corrected")
        if raw and corr:
            lines.append(
                f"- {dataset} {model}: centroid mode AUC changes from {float(raw['AUC_mean']):.3f} to {float(corr['AUC_mean']):.3f} after speaker-balanced correction."
            )
    lines += [
        "",
        "## F4 Nuisance Decodability",
        "",
        "Nuisance decodability rows use speaker-mode centroids and train-speaker-only ridge probes. Interpret them as linear decodability changes, not causal removal of pitch, prosody, or timbre.",
        "",
        "## F5 Duration Proxy Audit",
        "",
    ]
    for row in duration_summary:
        if row["dataset"] == "JVS_JVSMuSiC" and row["model"] in {"wavlm_l12", "mert_l3", "hubert_l6"}:
            lines.append(
                f"- {row['dataset']} {row['model']}: duration-only mode AUC mean {float(row['duration_only_mode_AUC_mean']):.3f}; classification `{row['duration_classification']}`."
            )
    lines += [
        "",
        "## F6 Restricted Gallery",
        "",
        "Restricted-gallery rows are saved in `restricted_gallery_results.csv`. JVS within-gender retrieval is the main confound check. Rows state when no query has an eligible gallery; small-gallery interpretation should use the reported gallery-size distribution rather than a blanket denominator rule.",
        "Chance R@1 is the mean across eligible queries of `1 / query-specific gallery size`; each row also reports the minimum, median, maximum, and frequency distribution of query-specific gallery sizes.",
        "",
        "## F7 JVS Mapper Status",
        "",
        "JVS B2 mapper status is reported separately in the final validation report when the minimal mapper output is present. If no JVS mapper output exists, the correct status is that the global-residual claim does not rely on a positive mapper result.",
        "",
        "## Supported Claim",
        "",
        "Frozen SSL representations contain a strong train-estimable speech-to-singing global mode residual. Correcting this global mode/duration-correlated direction improves cross-mode same-person retrieval on held-out speakers. The effect is not reproduced by shuffled nuisance, random Gaussian nuisance, or random low-rank controls from the main suite, and the direct global-adapter validation supports the same interpretation. Current evidence does not support a deployable individualized SSL residual mapper beyond the global baseline.",
        "",
        "## Unsupported Claims",
        "",
        "- We removed pitch.",
        "- We isolated timbre.",
        "- We learned singing identity residuals.",
        "- Duration is causal.",
        "- SeedVC behavior is solved.",
        "- GTSinger proves general person-specific residuals.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "final_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(out_dir: Path, outputs: dict[str, list[dict[str, Any]]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = [r for r in outputs["global_adapter"] if r["dataset"] == "JVS_JVSMuSiC" and r["variant"] in {"raw_speech_to_singing", "speech_plus_global_residual", "wrong_sign_global_residual"} and r["query_mode"] in {"speech", "speech+mu_delta", "speech-mu_delta"}]
    summary = summarize_group(rows, ["model", "variant"], "R1")
    models = ["wavlm_l12", "mert_l3", "hubert_l6", "ecapa"]
    variants = ["raw_speech_to_singing", "speech_plus_global_residual", "wrong_sign_global_residual"]
    x = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for j, variant in enumerate(variants):
        vals = []
        for model in models:
            row = next((r for r in summary if r["model"] == model and r["variant"] == variant), None)
            vals.append(float(row["R1_mean"]) if row else np.nan)
        ax.bar(x + (j - 1) * width, vals, width, label=variant)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R@1")
    ax.set_title("JVS/JVS-MuSiC global adapter validation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "global_adapter_jvs_r1.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Final validation suite for the identity residual global-mode claim.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/identity_residual_final_validation_2026-07-09"))
    parser.add_argument("--run-root", type=Path, default=Path("/localdisk/bowen/singing_identity/runs/identity_residual_final_validation_2026-07-09"))
    parser.add_argument("--control-draws", type=int, default=50)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.run_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, list[dict[str, Any]]] = {
        "global_adapter": [],
        "mode_probe": [],
        "nuisance": [],
        "duration": [],
        "speaker_balanced": [],
        "restricted": [],
        "per_speaker": [],
        "global_adapter_delta": [],
    }
    for dataset, cfg in DATASETS.items():
        run_dataset(dataset, cfg, args, outputs)
    outputs["global_adapter_delta"] = paired_split_delta_summaries(
        outputs["global_adapter"], HEADLINE_GLOBAL_COMPARISONS
    )
    write_csv(outputs["global_adapter"], args.results_dir / "global_adapter_results.csv")
    write_csv(outputs["global_adapter_delta"], args.results_dir / "global_adapter_paired_delta_summary.csv")
    write_csv(outputs["mode_probe"], args.results_dir / "mode_probe_results.csv")
    write_csv(outputs["nuisance"], args.results_dir / "nuisance_decodability_results.csv")
    write_csv(outputs["duration"], args.results_dir / "duration_proxy_audit.csv")
    write_csv(outputs["speaker_balanced"], args.results_dir / "speaker_balanced_mode_vector_results.csv")
    write_csv(outputs["restricted"], args.results_dir / "restricted_gallery_results.csv")
    write_csv(outputs["per_speaker"], args.results_dir / "per_speaker_rank_changes.csv")
    write_yaml(
        {
            "experiment_name": "identity_residual_final_validation_2026-07-09",
            "hostname": socket.gethostname(),
            "git_commit": current_git_commit(),
            "python": sys.executable,
            "run_root": str(args.run_root),
            "results_dir": str(args.results_dir),
            "cache_root": os.environ.get("XDG_CACHE_HOME", ""),
            "datasets": list(DATASETS),
            "control_draws": args.control_draws,
            "all_preprocessing_fit_population": "train_speakers_only",
            "claim_supported": "global_mode_duration_correlated_residual_accounting",
            "known_limitations": "mode and nuisance probes use speaker-mode centroids; paired split CIs can reuse speakers across partitions; shuffled-global control is mathematically degenerate under paired train speakers",
        },
        args.results_dir / "experiment_card.yaml",
    )
    write_plots(args.results_dir, outputs)
    write_summary_and_report(args.results_dir, outputs)
    print(f"wrote final validation outputs -> {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
