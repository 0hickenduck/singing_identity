"""Identity residualization and covariance estimation methods."""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from singing_identity.utils.research_utils import binary_auc


def stable_hash(*arrays: np.ndarray, extra: str = "") -> str:
    digest = hashlib.sha256(extra.encode("utf-8"))
    for array in arrays:
        x = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
        digest.update(str(x.shape).encode("ascii"))
        digest.update(x.tobytes())
    return digest.hexdigest()[:20]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def euclidean_scores(q: np.ndarray, g: np.ndarray) -> np.ndarray:
    q2 = np.sum(q * q, axis=1, keepdims=True)
    g2 = np.sum(g * g, axis=1, keepdims=True).T
    return -(q2 + g2 - 2.0 * q @ g.T)


def cosine_scores(q: np.ndarray, g: np.ndarray) -> np.ndarray:
    return l2_normalize(np.asarray(q, dtype=np.float64)) @ l2_normalize(np.asarray(g, dtype=np.float64)).T


@dataclass
class OASDual:
    center: np.ndarray
    basis: np.ndarray
    empirical_eigenvalues: np.ndarray
    shrinkage: float
    mu: float
    lambda0: float
    fitted_eigenvalues: np.ndarray
    transform_hash: str

    def whiten(self, x: np.ndarray) -> np.ndarray:
        z = np.asarray(x, dtype=np.float64)
        projection = z @ self.basis
        residual = z - projection @ self.basis.T
        return (projection / np.sqrt(self.fitted_eigenvalues)) @ self.basis.T + residual / math.sqrt(self.lambda0)

    def precision_scores(self, q: np.ndarray, g: np.ndarray) -> np.ndarray:
        return euclidean_scores(self.whiten(q), self.whiten(g))


def fit_oas_dual(z: np.ndarray, shrinkage: float | None = None) -> OASDual:
    """Wide OAS covariance using exact dual eigendecomposition."""
    z = np.asarray(z, dtype=np.float64)
    center = z.mean(axis=0)
    xc = z - center
    n, d = xc.shape
    _u, singular, vt = np.linalg.svd(xc, full_matrices=False)
    empirical = singular * singular / n
    keep = empirical > max(float(empirical[0]) if len(empirical) else 0.0, 1.0) * 1e-14
    empirical = empirical[keep]
    basis = vt[keep].T
    trace = float(empirical.sum())
    frob2 = float(np.sum(empirical * empirical))
    mu = trace / d
    if shrinkage is None:
        alpha = frob2 / (d * d)
        mu2 = mu * mu
        denominator = (n + 1) * (alpha - mu2 / d)
        shrinkage = 1.0 if denominator <= 0 else min((alpha + mu2) / denominator, 1.0)
    shrinkage = float(shrinkage)
    lambda0 = max(shrinkage * mu, np.finfo(np.float64).eps * max(mu, 1.0))
    fitted = (1.0 - shrinkage) * empirical + shrinkage * mu
    fitted = np.maximum(fitted, lambda0)
    transform_hash = stable_hash(center, basis, fitted, extra=f"oas:{shrinkage:.17g}:{d}")
    return OASDual(center, basis, empirical, shrinkage, mu, lambda0, fitted, transform_hash)


def remove_subspace(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project out subspace defined by orthogonal columns of basis."""
    values = np.asarray(x, dtype=np.float64)
    directions = np.asarray(basis, dtype=np.float64)
    if directions.size == 0:
        return values.copy()
    return values - (values @ directions) @ directions.T


def fit_diagonal_whitener(z_train: np.ndarray) -> tuple[np.ndarray, float, dict[str, float], str]:
    """Fit feature-wise variance whitener with small regularizer."""
    z_train = np.asarray(z_train, dtype=np.float64)
    variances = np.var(z_train, axis=0, ddof=0)
    mean_variance = float(np.mean(variances))
    epsilon = 1e-8 * mean_variance
    scale = np.sqrt(variances + epsilon)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("diagonal whitening produced non-positive or non-finite scale")
    positive = variances[variances > 0]
    stats = {
        "variance_min": float(np.min(variances)),
        "variance_median": float(np.median(variances)),
        "variance_max": float(np.max(variances)),
        "variance_ratio_max_min": float(np.max(variances) / np.min(positive)) if len(positive) else math.inf,
    }
    return scale, epsilon, stats, stable_hash(scale, extra=f"diag:{epsilon:.17g}")


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
        raise ValueError("logistic probe requires both classes in training data")
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


def mode_probe_centroid(
    z: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, float]:
    """Train logistic mode probe on train centroids and evaluate on test centroids."""
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


def restricted_retrieval(
    query: np.ndarray,
    gallery: np.ndarray,
    labels: list[str],
    attr: dict[str, str],
    attr_name: str,
) -> dict[str, Any]:
    """Restricted gallery retrieval matching metadata attributes."""
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


def speaker_interval(
    speaker_values: dict[str, list[float]],
    seed: int,
    samples: int = 10_000,
) -> tuple[float, float, float, float, int]:
    """Speaker-level bootstrap confidence interval."""
    values = np.asarray(
        [np.mean(speaker_values[speaker]) for speaker in sorted(speaker_values)],
        dtype=np.float64,
    )
    if not len(values):
        raise ValueError("speaker interval received no observations")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    bootstrap = values[indices].mean(axis=1)
    observed = float(values.mean())
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(samples, len(values)))
    null = np.abs(np.mean(signs * values[None, :], axis=1))
    p_value = float((1 + np.sum(null >= abs(observed))) / (samples + 1))
    return (
        observed,
        float(np.percentile(bootstrap, 2.5)),
        float(np.percentile(bootstrap, 97.5)),
        p_value,
        len(values),
    )


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
    control_draws: int = 0,
) -> np.ndarray:
    """Add global residual comparison rows and per-speaker details."""
    labels = [speakers[i] for i in test_idx]
    s_train = speech[train_idx]
    g_train = singing[train_idx]
    mu = (g_train - s_train).mean(axis=0)
    s_test = speech[test_idx]
    g_test = singing[test_idx]

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
    return mu


def paired_split_delta_summaries(
    rows: list[dict[str, Any]],
    comparisons: list[tuple[str, str]],
    metric: str = "R1",
    bootstrap_samples: int = 10000,
    random_seed: int = 20260710,
) -> list[dict[str, Any]]:
    """Compute paired split delta summaries across matching split seeds."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("dataset", "")), str(row.get("model", "")), str(row.get("layer", "")))].append(row)
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


__all__ = [
    "fit_oas_dual",
    "OASDual",
    "cosine_scores",
    "euclidean_scores",
    "remove_subspace",
    "fit_diagonal_whitener",
    "mode_probe_centroid",
    "restricted_retrieval",
    "speaker_interval",
    "add_global_rows",
    "paired_split_delta_summaries",
    "fit_ridge_logistic",
    "select_train_threshold",
    "zscore_fit",
    "zscore_apply",
]

