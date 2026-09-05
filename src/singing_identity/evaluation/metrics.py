"""Evaluation metrics for speaker verification and technique probing."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

from singing_identity.utils.research_utils import binary_auc


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def cosine_matrix(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    return l2(np.asarray(query, dtype=np.float64)) @ l2(np.asarray(gallery, dtype=np.float64)).T


def cosine_scores(q: np.ndarray, g: np.ndarray) -> np.ndarray:
    return cosine_matrix(q, g)


def euclidean_scores(q: np.ndarray, g: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    q2 = np.sum(q * q, axis=1, keepdims=True)
    g2 = np.sum(g * g, axis=1, keepdims=True).T
    return -(q2 + g2 - 2.0 * q @ g.T)


def ranks_from_scores(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, axis=1)
    ranks = np.asarray([int(np.flatnonzero(order[i] == i)[0]) + 1 for i in range(len(scores))])
    return ranks, order


def genuine_impostor(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(scores)
    genuine = np.diag(scores)
    impostor = scores[~np.eye(n, dtype=bool)]
    return genuine, impostor


def eer(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Empirical test-set EER for descriptive reporting."""
    values = np.unique(np.concatenate([genuine, impostor]))
    if not len(values):
        return float("nan")
    candidates = np.concatenate([[np.nextafter(values[0], -np.inf)], values, [np.nextafter(values[-1], np.inf)]])
    fmr = np.asarray([np.mean(impostor >= t) for t in candidates])
    fnmr = np.asarray([np.mean(genuine < t) for t in candidates])
    index = int(np.argmin(np.abs(fmr - fnmr)))
    return float((fmr[index] + fnmr[index]) / 2.0)


def train_threshold_at_fmr(impostor: np.ndarray, target_fmr: float) -> tuple[float, float]:
    """Pick lowest threshold whose FMR is at most target, using training trials."""
    values = np.unique(np.asarray(impostor, dtype=np.float64))
    if not len(values):
        return float("nan"), float("nan")
    candidates = np.concatenate([[np.nextafter(values[0], -np.inf)], values, [np.nextafter(values[-1], np.inf)]])
    eligible = [float(t) for t in candidates if float(np.mean(impostor >= t)) <= target_fmr]
    threshold = min(eligible) if eligible else float(np.nextafter(values[-1], np.inf))
    return threshold, float(np.mean(impostor >= threshold))


def score_metrics(scores: np.ndarray, calibration_scores: np.ndarray | None = None, calibration_source: str = "") -> dict[str, Any]:
    genuine, impostor = genuine_impostor(scores)
    ranks, _ = ranks_from_scores(scores)
    y = np.concatenate([np.ones(len(genuine), dtype=int), np.zeros(len(impostor), dtype=int)])
    values = np.concatenate([genuine, impostor])
    output: dict[str, Any] = {
        "R1": float(np.mean(ranks == 1)),
        "R5": float(np.mean(ranks <= min(5, scores.shape[1]))),
        "MRR": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "ROC_AUC": float(binary_auc(y, values)),
        "EER": eer(genuine, impostor),
        "genuine_mean": float(np.mean(genuine)),
        "genuine_sd": float(np.std(genuine)),
        "impostor_mean": float(np.mean(impostor)),
        "impostor_sd": float(np.std(impostor)),
        "margin": float(np.mean(genuine) - np.mean(impostor)),
        "n_genuine": int(len(genuine)),
        "n_impostor": int(len(impostor)),
        "calibration_source": calibration_source,
        "calibration_threshold": float("nan"),
        "achieved_calibration_FMR": float("nan"),
        "TMR_FMR1": float("nan"),
    }
    if calibration_scores is not None:
        _cg, ci = genuine_impostor(calibration_scores)
        threshold, achieved = train_threshold_at_fmr(ci, 0.01)
        output.update(
            calibration_threshold=threshold,
            achieved_calibration_FMR=achieved,
            TMR_FMR1=float(np.mean(genuine >= threshold)),
        )
    return output


def verification_metrics(
    query: np.ndarray,
    gallery: np.ndarray,
    calibration_query: np.ndarray,
    calibration_gallery: np.ndarray,
) -> dict[str, float | int | str]:
    scores = cosine_matrix(query, gallery)
    cal_scores = cosine_matrix(calibration_query, calibration_gallery)
    genuine, impostor = genuine_impostor(scores)
    cal_genuine, cal_impostor = genuine_impostor(cal_scores)
    ranks, _ = ranks_from_scores(scores)
    output: dict[str, float | int | str] = {
        "R1": float(np.mean(ranks == 1)),
        "R5": float(np.mean(ranks <= min(5, scores.shape[1]))),
        "mean_rank": float(np.mean(ranks)),
        "MRR": float(np.mean(1.0 / ranks)),
        "EER_test_descriptive": eer(genuine, impostor),
        "genuine_score_mean": float(np.mean(genuine)),
        "impostor_score_mean": float(np.mean(impostor)),
        "genuine_minus_impostor": float(np.mean(genuine) - np.mean(impostor)),
        "test_genuine_trials": int(len(genuine)),
        "test_impostor_trials": int(len(impostor)),
        "train_genuine_trials": int(len(cal_genuine)),
        "train_impostor_trials": int(len(cal_impostor)),
    }
    for target, name in [(0.01, "TMR_at_FMR_1pct"), (0.001, "TMR_at_FMR_0_1pct")]:
        if target == 0.001 and len(cal_impostor) < 1000:
            output[name] = float("nan")
            output[f"{name}_status"] = "insufficient_train_impostor_trials_lt_1000"
            continue
        threshold, observed_fmr = train_threshold_at_fmr(cal_impostor, target)
        output[name] = float(np.mean(genuine >= threshold))
        output[f"{name}_status"] = "train_calibrated"
        output[f"{name}_train_threshold"] = threshold
        output[f"{name}_train_FMR"] = observed_fmr
    return output


def roc_det_points(genuine: np.ndarray, impostor: np.ndarray, max_points: int = 256) -> list[dict[str, float]]:
    values = np.unique(np.concatenate([genuine, impostor]))
    if len(values) > max_points:
        values = np.quantile(values, np.linspace(0.0, 1.0, max_points))
    points = []
    for threshold in values:
        fmr = float(np.mean(impostor >= threshold))
        fnmr = float(np.mean(genuine < threshold))
        points.append({"threshold": float(threshold), "FMR": fmr, "FNMR": fnmr, "TMR": 1.0 - fnmr})
    return points


def retrieval_chance_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    gallery_sizes = [int(size) for size in metrics.get("gallery_sizes", []) if int(size) > 0]
    if not gallery_sizes and int(metrics.get("den", 0)) > 0:
        gallery_sizes = [int(metrics["den"])] * int(metrics["den"])
    distribution = Counter(gallery_sizes)
    mean_val = float(np.mean([1.0 / size for size in gallery_sizes])) if gallery_sizes else float("nan")
    return {
        "chance_R1": mean_val,
        "chance_R1_mean_per_query": mean_val,
        "chance_R1_definition": "mean_i(1/gallery_size_i)",
        "gallery_size_min": min(gallery_sizes) if gallery_sizes else "",
        "gallery_size_median": float(np.median(gallery_sizes)) if gallery_sizes else "",
        "gallery_size_max": max(gallery_sizes) if gallery_sizes else "",
        "gallery_size_distribution": ";".join(f"{size}:{distribution[size]}" for size in sorted(distribution)),
    }


def summarize_baseline(
    observed_rows: list[dict[str, Any]],
    expected: dict[tuple[str, str], dict[str, float]],
    tolerance_pp: float = 0.5,
) -> list[dict[str, Any]]:
    headline_models = ["wavlm_l12", "hubert_l6", "mert_l3"]
    baseline_conditions = ("oas_whitened_cosine_raw", "oas_whitened_cosine_query")
    output: list[dict[str, Any]] = []
    for model in headline_models:
        for condition in baseline_conditions:
            selected = [
                row
                for row in observed_rows
                if row["model"] == model and row["condition_id"] == condition
            ]
            if len(selected) != 20:
                raise ValueError(
                    f"expected 20 reproduced rows for {model}/{condition}, found {len(selected)}"
                )
            result: dict[str, Any] = {
                "dataset": "JVS_JVSMuSiC",
                "model": model,
                "condition_id": condition,
                "splits": len(selected),
                "tolerance_pp": tolerance_pp,
            }
            passed = True
            for metric in ("R1", "EER"):
                observed = float(np.mean([float(row[metric]) for row in selected]))
                reference = expected[(model, condition)][metric]
                difference_pp = 100.0 * (observed - reference)
                result[f"expected_{metric}"] = reference
                result[f"observed_{metric}"] = observed
                result[f"difference_pp_{metric}"] = difference_pp
                passed = passed and abs(difference_pp) <= tolerance_pp
            result["status"] = "PASS" if passed else "FAIL_AUDIT_REQUIRED"
            output.append(result)
    return output


def frame_vector(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x.mean(axis=0), x.std(axis=0)]).astype(np.float64)


def crop_starts(n_frames: int, fixed_frames: int, candidate_count: int, rng: np.random.Generator) -> np.ndarray:
    if n_frames < fixed_frames:
        return np.asarray([], dtype=int)
    if n_frames == fixed_frames:
        return np.asarray([0], dtype=int)
    maximum = n_frames - fixed_frames
    starts = {0, maximum}
    while len(starts) < min(candidate_count, maximum + 1):
        starts.add(int(rng.integers(0, maximum + 1)))
    return np.asarray(sorted(starts), dtype=int)


def choose_matched_crops(
    speech_x: np.ndarray,
    speech_voiced: np.ndarray,
    singing_x: np.ndarray,
    singing_voiced: np.ndarray,
    fixed_frames: int,
    candidate_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]] | None:
    """Return equal-length crops whose voiced-frame ratios are as close as possible."""
    s_starts = crop_starts(len(speech_x), fixed_frames, candidate_count, rng)
    g_starts = crop_starts(len(singing_x), fixed_frames, candidate_count, rng)
    if not len(s_starts) or not len(g_starts):
        return None
    s_ratio = {int(start): float(speech_voiced[start : start + fixed_frames].mean()) for start in s_starts}
    g_ratio = {int(start): float(singing_voiced[start : start + fixed_frames].mean()) for start in g_starts}
    choice = min(
        ((abs(s_ratio[s] - g_ratio[g]), s, g) for s in s_starts for g in g_starts),
        key=lambda item: (item[0], item[1], item[2]),
    )
    _, s_start, g_start = choice
    return (
        speech_x[s_start : s_start + fixed_frames],
        singing_x[g_start : g_start + fixed_frames],
        {
            "speech_crop_start": int(s_start),
            "singing_crop_start": int(g_start),
            "speech_voiced_ratio": s_ratio[int(s_start)],
            "singing_voiced_ratio": g_ratio[int(g_start)],
            "voiced_ratio_abs_difference": float(choice[0]),
        },
    )


__all__ = [
    "l2",
    "cosine_matrix",
    "cosine_scores",
    "euclidean_scores",
    "ranks_from_scores",
    "genuine_impostor",
    "eer",
    "train_threshold_at_fmr",
    "score_metrics",
    "verification_metrics",
    "roc_det_points",
    "retrieval_chance_summary",
    "summarize_baseline",
    "frame_vector",
    "crop_starts",
    "choose_matched_crops",
]

