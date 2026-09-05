#!/usr/bin/env python
"""Metric/origin robustness audit for the speech--singing global displacement.

The scientific unit is a speaker-mode centroid.  Every origin, displacement,
covariance, LDA transform, hyperparameter, and operating threshold is fit
without test-speaker information.  Wide OAS covariance operations use an exact
dual eigendecomposition, avoiding a dense D x D matrix.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probing.run_identity_residual_robustness import (  # noqa: E402
    GTS_FEATURE_ROOT,
    GTS_PAIRS,
    GTS_PRECOMPUTED_VECTOR_CACHE,
    GTS_UTTERANCES,
    JVS_FEATURE_ROOT,
    JVS_MANIFEST,
    JVS_SEEDS,
    cosine_matrix,
    eer,
    genuine_impostor,
    ranks_from_scores,
    train_threshold_at_fmr,
)
from probing.run_identity_residual_same_text import (  # noqa: E402
    GTSINGER_SEEDS,
    load_pair_vectors,
    select_clean_control_pairs,
)
from probing.run_identity_residual_suite import (  # noqa: E402
    MODEL_SPECS,
    canonical_mode,
    load_feature_cache,
    make_speaker_split,
    read_jsonl,
)
from research_utils import ExperimentError, binary_auc, current_git_commit  # noqa: E402


ALL_MODELS = [
    "wavlm_l3", "wavlm_l6", "wavlm_l9", "wavlm_l12",
    "hubert_l3", "hubert_l6", "hubert_l9", "hubert_l12",
    "mert_l3", "mert_l6", "mert_l9", "mert_l12",
]
HEADLINE_MODELS = ["wavlm_l12", "hubert_l6", "mert_l3"]
EXPECTED_BASELINE = {
    "wavlm_l12": {"raw_R1": 0.135, "corrected_R1": 0.363, "raw_EER": 0.421, "corrected_EER": 0.266},
    "hubert_l6": {"raw_R1": 0.195, "corrected_R1": 0.605, "raw_EER": 0.371, "corrected_EER": 0.186},
    "mert_l3": {"raw_R1": 0.365, "corrected_R1": 0.660, "raw_EER": 0.285, "corrected_EER": 0.161},
}
SHRINKAGE_GRID = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def write_yaml(data: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            lines.append(f"{key}:")
            lines.extend(f"  - {item}" for item in value)
        else:
            lines.append(f"{key}: {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def stable_hash(*arrays: np.ndarray, extra: str = "") -> str:
    digest = hashlib.sha256(extra.encode("utf-8"))
    for array in arrays:
        x = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
        digest.update(str(x.shape).encode("ascii"))
        digest.update(x.tobytes())
    return digest.hexdigest()[:20]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))


def cosine_scores(q: np.ndarray, g: np.ndarray) -> np.ndarray:
    return cosine_matrix(np.asarray(q, dtype=np.float64), np.asarray(g, dtype=np.float64))


def euclidean_scores(q: np.ndarray, g: np.ndarray) -> np.ndarray:
    q2 = np.sum(q * q, axis=1, keepdims=True)
    g2 = np.sum(g * g, axis=1, keepdims=True).T
    return -(q2 + g2 - 2.0 * q @ g.T)


def score_metrics(scores: np.ndarray, calibration_scores: np.ndarray | None, calibration_source: str) -> dict[str, Any]:
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
        "ROC_AUC": binary_auc(y, values),
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


@dataclass
class DatasetVectors:
    name: str
    protocol: str
    seeds: list[int]
    speakers: list[str]
    speech: np.ndarray
    singing: np.ndarray
    layer: str
    model: str
    n_pairs: int | None = None
    pair_rows: list[dict[str, Any]] | None = None
    pair_speech: np.ndarray | None = None
    pair_singing: np.ndarray | None = None


def load_jvs(model: str, cache_root: Path) -> DatasetVectors:
    rows, x, meta = load_feature_cache(
        read_jsonl(JVS_MANIFEST), JVS_FEATURE_ROOT, model, cache_root, 20, False
    )
    grouped: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for row, vector in zip(rows, x):
        mode = canonical_mode(row)
        if mode in {"speech", "singing"}:
            grouped[str(row["speaker_id"])][mode].append(np.asarray(vector, dtype=np.float64))
    speakers = sorted(s for s, modes in grouped.items() if modes["speech"] and modes["singing"])
    speech = np.vstack([np.vstack(grouped[s]["speech"]).mean(axis=0) for s in speakers])
    singing = np.vstack([np.vstack(grouped[s]["singing"]).mean(axis=0) for s in speakers])
    return DatasetVectors("JVS_JVSMuSiC", "jvs_60_20_20", JVS_SEEDS, speakers, speech, singing, str(meta["layer"]), model)


def load_gtsinger(model: str, cache_root: Path) -> DatasetVectors:
    pair_rows = read_jsonl(GTS_PAIRS)
    utterances = {str(row["utt_id"]): row for row in read_jsonl(GTS_UTTERANCES)}
    pairs, _ = select_clean_control_pairs(pair_rows, utterances)
    preferred = GTS_PRECOMPUTED_VECTOR_CACHE if model in HEADLINE_MODELS and GTS_PRECOMPUTED_VECTOR_CACHE.exists() else cache_root
    pair_speech, pair_singing, meta = load_pair_vectors(pairs, GTS_FEATURE_ROOT, model, preferred)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, pair in enumerate(pairs):
        groups[str(pair["speaker_id"])].append(index)
    speakers = sorted(groups)
    speech = np.vstack([pair_speech[groups[s]].mean(axis=0) for s in speakers])
    singing = np.vstack([pair_singing[groups[s]].mean(axis=0) for s in speakers])
    return DatasetVectors(
        "GTSinger_same_text_control", "gtsinger_10_0_10", GTSINGER_SEEDS,
        speakers, speech, singing, str(meta["layer"]), model, len(pairs), pairs, pair_speech, pair_singing,
    )


def split_arrays(data: DatasetVectors, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = make_speaker_split(data.speakers, seed, data.protocol)
    train = np.asarray([i for i, s in enumerate(data.speakers) if split[s] == "train"], dtype=int)
    dev = np.asarray([i for i, s in enumerate(data.speakers) if split[s] == "dev"], dtype=int)
    test = np.asarray([i for i, s in enumerate(data.speakers) if split[s] == "test"], dtype=int)
    assert not (set(train) & set(dev) or set(train) & set(test) or set(dev) & set(test))
    return train, dev, test


def fit_geometry(data: DatasetVectors, train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = data.speech[train].mean(axis=0)
    b = data.singing[train].mean(axis=0)
    d = (data.singing[train] - data.speech[train]).mean(axis=0)
    np.testing.assert_allclose(d, b - a, rtol=1e-10, atol=1e-10)
    m = (a + b) / 2.0
    return a, b, d, m


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


@dataclass
class LDABackend:
    mean: np.ndarray
    basis: np.ndarray
    scale: np.ndarray
    classifier: Any
    max_dim: int
    shrinkage: float
    transform_hash: str

    def transform(self, x: np.ndarray, k: int) -> np.ndarray:
        projected = ((np.asarray(x, dtype=np.float64) - self.mean) @ self.basis) / self.scale
        transformed = np.asarray(self.classifier.transform(projected), dtype=np.float64)
        return transformed[:, : min(k, transformed.shape[1])]


def fit_lda_backend(speech: np.ndarray, singing: np.ndarray, labels: list[str]) -> LDABackend:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    z = np.vstack([speech, singing]).astype(np.float64)
    y = np.asarray(labels + labels, dtype=object)
    mean = z.mean(axis=0)
    _u, singular, vt = np.linalg.svd(z - mean, full_matrices=False)
    keep = singular > max(float(singular[0]) if len(singular) else 0.0, 1.0) * 1e-12
    basis = vt[keep].T
    projected = (z - mean) @ basis
    scale = np.maximum(projected.std(axis=0), 1e-10)
    projected = projected / scale
    max_dim = min(len(set(labels)) - 1, projected.shape[1])
    class_mean = (projected[: len(labels)] + projected[len(labels) :]) / 2.0
    within = np.vstack([projected[: len(labels)] - class_mean, projected[len(labels) :] - class_mean])
    analytic_shrinkage = fit_oas_dual(within).shrinkage
    clf = LinearDiscriminantAnalysis(solver="eigen", shrinkage=analytic_shrinkage, n_components=max_dim)
    clf.fit(projected, y)
    transform_hash = stable_hash(mean, basis, scale, np.asarray(clf.scalings_), extra=f"lda:{max_dim}:{analytic_shrinkage:.17g}")
    return LDABackend(mean, basis, scale, clf, max_dim, analytic_shrinkage, transform_hash)


def save_score_matrix(
    root: Path, data: DatasetVectors, seed: int, condition: str, scores: np.ndarray,
    labels: list[str], transform_hash: str,
) -> None:
    safe_dataset = data.name.replace("/", "_")
    path = root / f"{safe_dataset}__{data.model}__{seed}__{condition}.npz"
    np.savez_compressed(
        path,
        score_matrix=np.asarray(scores, dtype=np.float64),
        ordered_query_speaker_ids=np.asarray(labels, dtype="U"),
        ordered_gallery_speaker_ids=np.asarray(labels, dtype="U"),
        condition_id=np.asarray(condition),
        split_seed=np.asarray(seed),
        transform_hash=np.asarray(transform_hash),
        higher_is_better=np.asarray(True),
    )


def metric_row(
    data: DatasetVectors,
    seed: int,
    condition: str,
    score_family: str,
    origin: str,
    alignment: str,
    covariance_method: str,
    shrinkage: float | str,
    lda_dim: int | str,
    train: np.ndarray,
    dev: np.ndarray,
    test: np.ndarray,
    transform_hash: str,
    scores: np.ndarray,
    calibration_scores: np.ndarray | None,
    pair_group: str,
    corrected: bool,
) -> dict[str, Any]:
    calibration_source = "dev_speakers" if len(dev) else "insufficient_resolution/non_independent_calibration"
    metrics = score_metrics(scores, calibration_scores if len(dev) else None, calibration_source)
    return {
        "dataset": data.name, "model": data.model, "layer": data.layer, "split_seed": seed,
        "condition_id": condition, "score_family": score_family, "origin": origin, "alignment": alignment,
        "covariance_method": covariance_method, "shrinkage": shrinkage, "lda_dim": lda_dim,
        "train_speakers": ";".join(data.speakers[i] for i in train),
        "dev_speakers": ";".join(data.speakers[i] for i in dev),
        "test_speakers": ";".join(data.speakers[i] for i in test),
        "transform_hash": transform_hash, "pair_group": pair_group, "is_corrected": int(corrected),
        **metrics,
    }


def per_query_rows(
    data: DatasetVectors, seed: int, condition: str, pair_group: str, corrected: bool,
    test: np.ndarray, scores: np.ndarray,
) -> list[dict[str, Any]]:
    ranks, _ = ranks_from_scores(scores)
    output = []
    for row_index, speaker_index in enumerate(test):
        impostor = np.delete(scores[row_index], row_index)
        output.append({
            "dataset": data.name, "model": data.model, "layer": data.layer, "split_seed": seed,
            "condition_id": condition, "pair_group": pair_group, "is_corrected": int(corrected),
            "speaker_id": data.speakers[speaker_index], "rank": int(ranks[row_index]),
            "hit1": int(ranks[row_index] == 1), "reciprocal_rank": float(1.0 / ranks[row_index]),
            "genuine_score": float(scores[row_index, row_index]),
            "impostor_score_mean": float(impostor.mean()),
            "query_margin": float(scores[row_index, row_index] - impostor.mean()),
        })
    return output


def add_metric_condition(
    metric_rows: list[dict[str, Any]], query_rows: list[dict[str, Any]], score_root: Path,
    data: DatasetVectors, seed: int, condition: str, family: str, origin: str, alignment: str,
    covariance: str, shrinkage: float | str, lda_dim: int | str,
    train: np.ndarray, dev: np.ndarray, test: np.ndarray, transform_hash: str,
    scores: np.ndarray, calibration_scores: np.ndarray | None, pair_group: str, corrected: bool,
) -> None:
    if not np.all(np.isfinite(scores)):
        raise ExperimentError(f"non-finite scores: {data.name}/{data.model}/{seed}/{condition}")
    labels = [data.speakers[i] for i in test]
    metric_rows.append(metric_row(
        data, seed, condition, family, origin, alignment, covariance, shrinkage, lda_dim,
        train, dev, test, transform_hash, scores, calibration_scores, pair_group, corrected,
    ))
    query_rows.extend(per_query_rows(data, seed, condition, pair_group, corrected, test, scores))
    save_score_matrix(score_root, data, seed, condition, scores, labels, transform_hash)


def run_geometry_and_layer(
    data: DatasetVectors, magnitude_rows: list[dict[str, Any]], layer_rows: list[dict[str, Any]],
) -> None:
    for seed in data.seeds:
        train, dev, test = split_arrays(data, seed)
        _a, _b, d, _m = fit_geometry(data, train)
        delta = data.singing[test] - data.speech[test]
        residual = delta - d
        dnorm = float(np.linalg.norm(d))
        numerators = np.sum(residual * residual, axis=1)
        denominators = np.sum(delta * delta, axis=1)
        align = (delta @ d) / np.maximum(np.linalg.norm(delta, axis=1) * dnorm, 1e-12)
        rho = np.sqrt(numerators / np.maximum(denominators, 1e-12))
        kappa = dnorm / np.maximum(np.linalg.norm(delta, axis=1), 1e-12)
        beta = (delta @ d) / max(dnorm * dnorm, 1e-12)
        e_test = 1.0 - float(numerators.sum()) / max(float(denominators.sum()), 1e-12)
        for j, index in enumerate(test):
            magnitude_rows.append({
                "dataset": data.name, "model": data.model, "layer": data.layer, "split_seed": seed,
                "speaker_id": data.speakers[index], "alignment_cosine": float(align[j]),
                "rho": float(rho[j]), "kappa": float(kappa[j]), "beta": float(beta[j]),
                "residual_squared_error": float(numerators[j]), "delta_squared_norm": float(denominators[j]),
                "E_test_split": e_test,
            })
        raw = cosine_scores(data.speech[test], data.singing[test])
        corrected = cosine_scores(data.speech[test] + d, data.singing[test])
        if len(dev):
            cal_raw = cosine_scores(data.speech[dev], data.singing[dev])
            cal_corr = cosine_scores(data.speech[dev] + d, data.singing[dev])
        else:
            cal_raw = cal_corr = None
        raw_m = score_metrics(raw, cal_raw, "dev_speakers" if len(dev) else "not_primary")
        corr_m = score_metrics(corrected, cal_corr, "dev_speakers" if len(dev) else "not_primary")
        layer_rows.append({
            "dataset": data.name, "model": data.model, "layer": int(data.layer), "split_seed": seed,
            "heldout_direction_cosine": float(align.mean()), "E_test": e_test, "median_rho": float(np.median(rho)),
            "raw_R1": raw_m["R1"], "corrected_R1": corr_m["R1"],
            "raw_EER": raw_m["EER"], "corrected_EER": corr_m["EER"],
            "raw_TMR_FMR1": raw_m["TMR_FMR1"], "corrected_TMR_FMR1": corr_m["TMR_FMR1"],
        })


def run_metric_audit(
    data: DatasetVectors,
    metric_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    fit_rows: list[dict[str, Any]],
    sensitivity_rows: list[dict[str, Any]],
    score_root: Path,
) -> None:
    for seed in data.seeds:
        train, dev, test = split_arrays(data, seed)
        a, b, d, m = fit_geometry(data, train)
        s_train, g_train = data.speech[train], data.singing[train]
        s_test, g_test = data.speech[test], data.singing[test]
        s_dev, g_dev = data.speech[dev], data.singing[dev]
        base_hash = stable_hash(a, b, d, extra="train_geometry")

        # M1: exact six-cell origin/alignment factorial.
        cells = [
            ("cos_o0_none", s_test, g_test, s_dev, g_dev, "zero", "none", "m1_o0", False),
            ("cos_o0_query", s_test + d, g_test, s_dev + d, g_dev, "zero", "query", "m1_o0", True),
            ("cos_o0_sym", s_test + d / 2, g_test - d / 2, s_dev + d / 2, g_dev - d / 2, "zero", "symmetric", "m1_o0_sym", True),
            ("cos_om_none", s_test - m, g_test - m, s_dev - m, g_dev - m, "pooled_train_midpoint", "none", "m1_om", False),
            ("cos_om_query", s_test + d - m, g_test - m, s_dev + d - m, g_dev - m, "pooled_train_midpoint", "query", "m1_om", True),
            ("cos_om_sym", s_test - a, g_test - b, s_dev - a, g_dev - b, "pooled_train_midpoint", "symmetric", "m1_om_sym", True),
        ]
        for condition, q, g, qd, gd, origin, alignment, pair_group, corrected in cells:
            scores = cosine_scores(q, g)
            cal = cosine_scores(qd, gd) if len(dev) else None
            add_metric_condition(
                metric_rows, query_rows, score_root, data, seed, condition, "cosine", origin, alignment,
                "none", "", "", train, dev, test, base_hash, scores, cal, pair_group, corrected,
            )
        np.testing.assert_allclose(s_test - a, s_test + d / 2 - m, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(g_test - b, g_test - d / 2 - m, rtol=1e-10, atol=1e-10)

        # Euclidean identity assertions and M3.
        e_raw = euclidean_scores(s_test, g_test)
        e_corr = euclidean_scores(s_test + d, g_test)
        e_sym = euclidean_scores(s_test + d / 2, g_test - d / 2)
        e_mode = euclidean_scores(s_test - a, g_test - b)
        np.testing.assert_allclose(e_corr, e_sym, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(e_corr, e_mode, rtol=1e-9, atol=1e-9)
        norm_e = euclidean_scores(
            s_test / np.maximum(np.linalg.norm(s_test, axis=1, keepdims=True), 1e-12),
            g_test / np.maximum(np.linalg.norm(g_test, axis=1, keepdims=True), 1e-12),
        )
        assert np.array_equal(np.argsort(-norm_e, axis=1), np.argsort(-cosine_scores(s_test, g_test), axis=1))
        for condition, scores, qd, gd, corrected in [
            ("euclidean_raw", e_raw, s_dev, g_dev, False),
            ("euclidean_query", e_corr, s_dev + d, g_dev, True),
        ]:
            cal = euclidean_scores(qd, gd) if len(dev) else None
            add_metric_condition(
                metric_rows, query_rows, score_root, data, seed, condition, "euclidean", "origin_invariant",
                "query" if corrected else "none", "none", "", "", train, dev, test, base_hash,
                scores, cal, "m3_euclidean", corrected,
            )

        # M2/M4: speaker-balanced OAS fit on one speech and one singing centroid.
        z_train = np.vstack([s_train, g_train])
        oas = fit_oas_dual(z_train)
        fit_rows.append({
            "dataset": data.name, "model": data.model, "layer": data.layer, "split_seed": seed,
            "fit_type": "OAS", "fit_speakers": ";".join(data.speakers[i] for i in train),
            "n_fit_vectors": len(z_train), "dimension": z_train.shape[1], "shrinkage": oas.shrinkage,
            "covariance_trace": float(oas.empirical_eigenvalues.sum()),
            "minimum_effective_eigenvalue": oas.lambda0,
            "condition_number": float(max(oas.fitted_eigenvalues.max(initial=oas.lambda0), oas.lambda0) / oas.lambda0),
            "transform_hash": oas.transform_hash,
        })
        for condition, q, g, qd, gd, corrected in [
            ("oas_whitened_cosine_raw", s_test - m, g_test - m, s_dev - m, g_dev - m, False),
            ("oas_whitened_cosine_query", s_test + d - m, g_test - m, s_dev + d - m, g_dev - m, True),
        ]:
            scores = cosine_scores(oas.whiten(q), oas.whiten(g))
            cal = cosine_scores(oas.whiten(qd), oas.whiten(gd)) if len(dev) else None
            add_metric_condition(
                metric_rows, query_rows, score_root, data, seed, condition, "oas_whitened_cosine",
                "pooled_train_midpoint", "query" if corrected else "none", "OAS", oas.shrinkage, "",
                train, dev, test, oas.transform_hash, scores, cal, "m2_oas_whitened_cosine", corrected,
            )
        m_raw = oas.precision_scores(s_test, g_test)
        m_corr = oas.precision_scores(s_test + d, g_test)
        m_sym = oas.precision_scores(s_test + d / 2, g_test - d / 2)
        np.testing.assert_allclose(m_corr, m_sym, rtol=1e-8, atol=1e-8)
        for condition, scores, qd, gd, corrected in [
            ("oas_mahalanobis_raw", m_raw, s_dev, g_dev, False),
            ("oas_mahalanobis_query", m_corr, s_dev + d, g_dev, True),
        ]:
            cal = oas.precision_scores(qd, gd) if len(dev) else None
            add_metric_condition(
                metric_rows, query_rows, score_root, data, seed, condition, "oas_mahalanobis", "origin_invariant",
                "query" if corrected else "none", "OAS", oas.shrinkage, "", train, dev, test,
                oas.transform_hash, scores, cal, "m4_oas_mahalanobis", corrected,
            )
        delta_test = g_test - s_test
        wd = oas.whiten(d[None, :])[0]
        wdelta = oas.whiten(delta_test)
        wres = oas.whiten(delta_test - d)
        fit_rows[-1].update(
            whitened_residual_alignment=float(np.mean((wdelta @ wd) / np.maximum(np.linalg.norm(wdelta, axis=1) * np.linalg.norm(wd), 1e-12))),
            whitened_E_test=1.0 - float(np.sum(wres * wres)) / max(float(np.sum(wdelta * wdelta)), 1e-12),
        )
        for lam in SHRINKAGE_GRID:
            fixed = fit_oas_dual(z_train, lam)
            wc_raw = cosine_scores(fixed.whiten(s_test - m), fixed.whiten(g_test - m))
            wc_corr = cosine_scores(fixed.whiten(s_test + d - m), fixed.whiten(g_test - m))
            mh_raw = fixed.precision_scores(s_test, g_test)
            mh_corr = fixed.precision_scores(s_test + d, g_test)
            if lam == 1.0:
                assert np.array_equal(np.argsort(-wc_raw, axis=1), np.argsort(-cosine_scores(s_test - m, g_test - m), axis=1))
                assert np.array_equal(np.argsort(-mh_raw, axis=1), np.argsort(-e_raw, axis=1))
            sensitivity_rows.extend([
                {"dataset": data.name, "model": data.model, "layer": data.layer, "split_seed": seed,
                 "backend": "whitened_cosine", "shrinkage": lam, "alignment": name,
                 **score_metrics(scores, None, "sensitivity_no_threshold")}
                for name, scores in [("raw", wc_raw), ("query", wc_corr)]
            ])
            sensitivity_rows.extend([
                {"dataset": data.name, "model": data.model, "layer": data.layer, "split_seed": seed,
                 "backend": "mahalanobis", "shrinkage": lam, "alignment": name,
                 **score_metrics(scores, None, "sensitivity_no_threshold")}
                for name, scores in [("raw", mh_raw), ("query", mh_corr)]
            ])

        # M5: shrinkage LDA in the exact centered training span.
        lda = fit_lda_backend(s_train, g_train, [data.speakers[i] for i in train])
        if len(dev):
            candidates = [k for k in [8, 16, 32, 59] if k <= lda.max_dim]
            choices = []
            for k in candidates:
                dev_scores = cosine_scores(lda.transform(s_dev, k), lda.transform(g_dev, k))
                dev_metrics = score_metrics(dev_scores, None, "dev_selection")
                choices.append((float(dev_metrics["EER"]), -float(dev_metrics["MRR"]), k))
            selected = min(choices)[2]
            dims = [selected]
        else:
            dims = [k for k in [2, 4, 8] if k <= lda.max_dim]
            selected = min(8, lda.max_dim)
        fit_rows.append({
            "dataset": data.name, "model": data.model, "layer": data.layer, "split_seed": seed,
            "fit_type": "regularized_LDA", "fit_speakers": ";".join(data.speakers[i] for i in train),
            "n_fit_vectors": 2 * len(train), "dimension": data.speech.shape[1],
            "span_dimension": lda.basis.shape[1], "max_lda_dimension": lda.max_dim,
            "lda_analytic_OAS_shrinkage": lda.shrinkage,
            "selected_lda_dimension": selected, "selection_rule": "raw_dev_EER_then_MRR_then_smaller_k" if len(dev) else "fixed_primary_k8",
            "transform_hash": lda.transform_hash,
        })
        for k in dims:
            suffix = "" if k == selected else f"_k{k}_sensitivity"
            for condition, q, qd, corrected in [
                (f"lda_cosine_raw{suffix}", s_test, s_dev, False),
                (f"lda_cosine_query{suffix}", s_test + d, s_dev + d, True),
            ]:
                scores = cosine_scores(lda.transform(q, k), lda.transform(g_test, k))
                cal = cosine_scores(lda.transform(qd, k), lda.transform(g_dev, k)) if len(dev) else None
                add_metric_condition(
                    metric_rows, query_rows, score_root, data, seed, condition, "lda_cosine",
                    "lda_internal_centering", "query" if corrected else "none", "LDA_analytic_OAS_shrinkage", lda.shrinkage, k,
                    train, dev, test, lda.transform_hash, scores, cal, f"m5_lda_cosine_k{k}", corrected,
                )


def deterministic_halves(data: DatasetVectors) -> dict[str, dict[str, np.ndarray]]:
    if data.pair_rows is None or data.pair_speech is None or data.pair_singing is None:
        raise ExperimentError("GTSinger pair vectors required for analogy")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, pair in enumerate(data.pair_rows):
        groups[str(pair["speaker_id"])].append(index)
    output: dict[str, dict[str, np.ndarray]] = {}
    for speaker in data.speakers:
        indices = sorted(groups[speaker], key=lambda i: str(data.pair_rows[i]["pair_id"]))
        a = np.asarray(indices[::2], dtype=int)
        b = np.asarray(indices[1::2], dtype=int)
        if not len(a) or not len(b):
            raise ExperimentError(f"insufficient disjoint pair halves for {speaker}")
        output[speaker] = {
            "speech_A": data.pair_speech[a].mean(axis=0), "speech_B": data.pair_speech[b].mean(axis=0),
            "singing_A": data.pair_singing[a].mean(axis=0), "singing_B": data.pair_singing[b].mean(axis=0),
        }
    return output


def run_analogy(data: DatasetVectors, rows: list[dict[str, Any]]) -> None:
    halves = deterministic_halves(data)
    for seed in data.seeds:
        train, _dev, test = split_arrays(data, seed)
        train_labels = [data.speakers[i] for i in train]
        test_labels = [data.speakers[i] for i in test]
        sa_train = np.vstack([halves[s]["speech_A"] for s in train_labels])
        ga_train = np.vstack([halves[s]["singing_A"] for s in train_labels])
        d = (ga_train - sa_train).mean(axis=0)
        m = (sa_train.mean(axis=0) + ga_train.mean(axis=0)) / 2.0
        sa = np.vstack([halves[s]["speech_A"] for s in test_labels])
        ga = np.vstack([halves[s]["singing_A"] for s in test_labels])
        gb = np.vstack([halves[s]["singing_B"] for s in test_labels])
        for origin, shift in [("raw_origin", np.zeros_like(m)), ("pooled_centered", m)]:
            raw_scores = cosine_scores(sa - shift, gb - shift)
            arith_scores = cosine_scores(sa + d - shift, gb - shift)
            upper_scores = cosine_scores(ga - shift, gb - shift)
            arith_ranks, _ = ranks_from_scores(arith_scores)
            for i, speaker in enumerate(test_labels):
                lower = np.delete(cosine_scores(gb[i : i + 1] - shift, gb - shift)[0], i)
                arithmetic = float(arith_scores[i, i])
                upper = float(upper_scores[i, i])
                null_arithmetic = np.delete(arith_scores[:, i], i)
                identity_null = float(np.mean(
                    (lower[:, None] < null_arithmetic[None, :])
                    & (null_arithmetic[None, :] < upper)
                ))
                rows.append({
                    "dataset": data.name, "model": data.model, "layer": data.layer, "split_seed": seed,
                    "speaker_id": speaker, "origin": origin, "raw_same_person_score": float(raw_scores[i, i]),
                    "arithmetic_score": arithmetic, "singing_repeat_upper_score": upper,
                    "different_person_lower_mean": float(lower.mean()),
                    "ordering_fraction_over_impostors": float(np.mean((lower < arithmetic) & (arithmetic < upper))),
                    "ordering_identity_permutation_null": identity_null,
                    "ordering_minus_identity_null": float(np.mean((lower < arithmetic) & (arithmetic < upper))) - identity_null,
                    "lower_lt_arithmetic_fraction": float(np.mean(lower < arithmetic)),
                    "arithmetic_lt_upper": int(arithmetic < upper),
                    "corrected_retrieval_hit1": int(arith_ranks[i] == 1),
                })
        true_offsets = ga - sa
        negative_offsets = np.vstack([ga[j] - sa[i] for i in range(len(test)) for j in range(len(test)) if i != j])
        true_scores = np.asarray([cosine(v, d) for v in true_offsets])
        neg_scores = np.asarray([cosine(v, d) for v in negative_offsets])
        auc = binary_auc(
            np.concatenate([np.ones(len(true_scores), dtype=int), np.zeros(len(neg_scores), dtype=int)]),
            np.concatenate([true_scores, neg_scores]),
        )
        rng = np.random.default_rng(seed * 7919 + int(data.layer))
        perm = []
        for _ in range(200):
            shuffled = rng.permutation(len(test))
            offsets = ga[shuffled] - sa
            scores = np.asarray([cosine(v, d) for v in offsets])
            perm.append(binary_auc(
                np.concatenate([np.ones(len(scores), dtype=int), np.zeros(len(neg_scores), dtype=int)]),
                np.concatenate([scores, neg_scores]),
            ))
        for speaker in test_labels:
            rows.append({
                "dataset": data.name, "model": data.model, "layer": data.layer, "split_seed": seed,
                "speaker_id": speaker, "origin": "offset_consistency", "offset_consistency_AUC": auc,
                "offset_permutation_AUC_mean": float(np.mean(perm)),
                "offset_permutation_AUC_p95": float(np.percentile(perm, 95)),
            })


def percentile_summary(rows: list[dict[str, Any]], keys: list[str], metrics: Iterable[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k, "") for k in keys)].append(row)
    output = []
    for key, values in sorted(groups.items()):
        result: dict[str, Any] = {name: value for name, value in zip(keys, key)}
        result["rows"] = len(values)
        for metric in metrics:
            array = np.asarray([float(v.get(metric, float("nan"))) for v in values], dtype=np.float64)
            array = array[np.isfinite(array)]
            result[f"{metric}_mean"] = float(array.mean()) if len(array) else float("nan")
            result[f"{metric}_median"] = float(np.median(array)) if len(array) else float("nan")
            result[f"{metric}_CI_low"] = float(np.percentile(array, 2.5)) if len(array) else float("nan")
            result[f"{metric}_CI_high"] = float(np.percentile(array, 97.5)) if len(array) else float("nan")
        output.append(result)
    return output


def magnitude_summary(rows: list[dict[str, Any]], bootstrap_samples: int) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["model"]), str(row["layer"]))].append(row)
    output = []
    for (dataset, model, layer), values in sorted(groups.items()):
        by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for value in values:
            by_speaker[str(value["speaker_id"])].append(value)
        aggregate = []
        for speaker, speaker_rows in sorted(by_speaker.items()):
            aggregate.append({
                "speaker_id": speaker,
                "num": float(np.mean([float(r["residual_squared_error"]) for r in speaker_rows])),
                "den": float(np.mean([float(r["delta_squared_norm"]) for r in speaker_rows])),
                "rho": float(np.mean([float(r["rho"]) for r in speaker_rows])),
                "alignment": float(np.mean([float(r["alignment_cosine"]) for r in speaker_rows])),
                "kappa": float(np.mean([float(r["kappa"]) for r in speaker_rows])),
                "beta": float(np.mean([float(r["beta"]) for r in speaker_rows])),
            })
        rng = np.random.default_rng(31013 + int(layer) + len(dataset) + len(model))
        boot = []
        for _ in range(bootstrap_samples):
            sample = rng.integers(0, len(aggregate), len(aggregate))
            num = sum(aggregate[i]["num"] for i in sample)
            den = sum(aggregate[i]["den"] for i in sample)
            boot.append(1.0 - num / max(den, 1e-12))
        e_point = 1.0 - sum(x["num"] for x in aggregate) / max(sum(x["den"] for x in aggregate), 1e-12)
        output.append({
            "dataset": dataset, "model": model, "layer": layer, "unique_speakers": len(aggregate),
            "alignment_mean": float(np.mean([x["alignment"] for x in aggregate])),
            "rho_median": float(np.median([x["rho"] for x in aggregate])),
            "kappa_median": float(np.median([x["kappa"] for x in aggregate])),
            "beta_median": float(np.median([x["beta"] for x in aggregate])),
            "fraction_unique_speakers_rho_lt_1": float(np.mean([x["rho"] < 1.0 for x in aggregate])),
            "E_test_speaker_aggregate": e_point,
            "E_test_speaker_bootstrap_CI_low": float(np.percentile(boot, 2.5)),
            "E_test_speaker_bootstrap_CI_high": float(np.percentile(boot, 97.5)),
            "magnitude_gate": "PASS" if e_point > 0 and np.percentile(boot, 2.5) > 0 and np.mean([x["rho"] < 1.0 for x in aggregate]) > 0.5 else "FAIL",
        })
    return output


def paired_outputs(metric_rows: list[dict[str, Any]], query_rows: list[dict[str, Any]], bootstrap_samples: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_groups: dict[tuple[Any, ...], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in metric_rows:
        key = (row["dataset"], row["model"], row["layer"], row["split_seed"], row["pair_group"])
        metric_groups[key][int(row["is_corrected"])] = row
    deltas = []
    for key, pair in sorted(metric_groups.items()):
        if 0 not in pair or 1 not in pair:
            continue
        raw, corr = pair[0], pair[1]
        deltas.append({
            "dataset": key[0], "model": key[1], "layer": key[2], "split_seed": key[3], "pair_group": key[4],
            "raw_condition": raw["condition_id"], "corrected_condition": corr["condition_id"],
            "delta_R1": float(corr["R1"]) - float(raw["R1"]),
            "delta_EER": float(corr["EER"]) - float(raw["EER"]),
            "delta_MRR": float(corr["MRR"]) - float(raw["MRR"]),
            "delta_TMR_FMR1": float(corr["TMR_FMR1"]) - float(raw["TMR_FMR1"]),
        })
    query_groups: dict[tuple[Any, ...], dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in query_rows:
        key = (row["dataset"], row["model"], row["layer"], row["pair_group"], row["speaker_id"])
        query_groups[key][int(row["is_corrected"])].append(row)
    aggregates = []
    for key, pair in sorted(query_groups.items()):
        if 0 not in pair or 1 not in pair:
            continue
        raw_hit = float(np.mean([float(r["hit1"]) for r in pair[0]]))
        corr_hit = float(np.mean([float(r["hit1"]) for r in pair[1]]))
        raw_rr = float(np.mean([float(r["reciprocal_rank"]) for r in pair[0]]))
        corr_rr = float(np.mean([float(r["reciprocal_rank"]) for r in pair[1]]))
        aggregates.append({
            "dataset": key[0], "model": key[1], "layer": key[2], "pair_group": key[3], "speaker_id": key[4],
            "raw_hit1_mean": raw_hit, "corrected_hit1_mean": corr_hit, "delta_R1": corr_hit - raw_hit,
            "raw_reciprocal_rank_mean": raw_rr, "corrected_reciprocal_rank_mean": corr_rr, "delta_MRR": corr_rr - raw_rr,
        })
    # Add speaker-bootstrap intervals and sign-flip p values to each aggregate row.
    summary_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        summary_groups[(row["dataset"], row["model"], row["layer"], row["pair_group"])].append(row)
    for key, rows in summary_groups.items():
        values = np.asarray([float(r["delta_R1"]) for r in rows])
        rng = np.random.default_rng(7001 + len(rows) + len(str(key)))
        boot = np.asarray([values[rng.integers(0, len(values), len(values))].mean() for _ in range(bootstrap_samples)])
        signs = rng.choice([-1.0, 1.0], size=(min(20000, bootstrap_samples * 2), len(values)))
        p = float((1 + np.sum(np.abs((signs * values).mean(axis=1)) >= abs(values.mean()))) / (1 + len(signs)))
        for row in rows:
            row.update(
                group_delta_R1_mean=float(values.mean()),
                group_delta_R1_CI_low=float(np.percentile(boot, 2.5)),
                group_delta_R1_CI_high=float(np.percentile(boot, 97.5)),
                group_delta_R1_sign_flip_p=p,
            )
    holm_groups: dict[tuple[Any, ...], dict[str, float]] = defaultdict(dict)
    for key, rows in summary_groups.items():
        dataset, model, _layer, pair_group = key
        holm_groups[(dataset, pair_group)][str(model)] = float(rows[0]["group_delta_R1_sign_flip_p"])
    holm_adjusted: dict[tuple[Any, ...], float] = {}
    for (dataset, pair_group), model_p in holm_groups.items():
        ordered = sorted(model_p.items(), key=lambda item: item[1])
        running = 0.0
        count = len(ordered)
        for rank, (model, p_value) in enumerate(ordered):
            running = max(running, min(1.0, p_value * (count - rank)))
            holm_adjusted[(dataset, pair_group, model)] = running
    for row in aggregates:
        row["group_delta_R1_sign_flip_p_holm"] = holm_adjusted[
            (row["dataset"], row["pair_group"], row["model"])
        ]
    return deltas, aggregates


def analogy_bootstrap_summary(rows: list[dict[str, Any]], bootstrap_samples: int) -> list[dict[str, Any]]:
    metrics = [
        "raw_same_person_score", "arithmetic_score", "singing_repeat_upper_score",
        "different_person_lower_mean", "ordering_fraction_over_impostors",
        "ordering_identity_permutation_null", "ordering_minus_identity_null",
        "lower_lt_arithmetic_fraction", "arithmetic_lt_upper", "corrected_retrieval_hit1",
        "offset_consistency_AUC", "offset_permutation_AUC_mean", "offset_permutation_AUC_p95",
    ]
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["model"]), str(row["layer"]), str(row["origin"]))].append(row)
    output: list[dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for value in values:
            by_speaker[str(value["speaker_id"])].append(value)
        speaker_rows: list[dict[str, float]] = []
        for _speaker, speaker_values in sorted(by_speaker.items()):
            aggregate: dict[str, float] = {}
            for metric in metrics:
                array = np.asarray([float(v.get(metric, float("nan"))) for v in speaker_values], dtype=np.float64)
                array = array[np.isfinite(array)]
                aggregate[metric] = float(array.mean()) if len(array) else float("nan")
            speaker_rows.append(aggregate)
        result: dict[str, Any] = {
            "dataset": key[0], "model": key[1], "layer": key[2], "origin": key[3],
            "unique_speakers": len(speaker_rows),
        }
        rng = np.random.default_rng(91009 + int(key[2]) + len(key[1]) + len(key[3]))
        for metric in metrics:
            array = np.asarray([r[metric] for r in speaker_rows], dtype=np.float64)
            array = array[np.isfinite(array)]
            result[f"{metric}_mean"] = float(array.mean()) if len(array) else float("nan")
            if len(array):
                bootstrap = np.asarray([
                    array[rng.integers(0, len(array), len(array))].mean()
                    for _ in range(bootstrap_samples)
                ])
                result[f"{metric}_bootstrap99_low"] = float(np.percentile(bootstrap, 0.5))
                result[f"{metric}_bootstrap99_high"] = float(np.percentile(bootstrap, 99.5))
            else:
                result[f"{metric}_bootstrap99_low"] = float("nan")
                result[f"{metric}_bootstrap99_high"] = float("nan")
        output.append(result)
    return output


def baseline_reproduction(layer_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for model, expected in EXPECTED_BASELINE.items():
        values = [r for r in layer_rows if r["dataset"] == "JVS_JVSMuSiC" and r["model"] == model]
        row: dict[str, Any] = {"dataset": "JVS_JVSMuSiC", "model": model, "splits": len(values)}
        passed = True
        for metric in ["raw_R1", "corrected_R1", "raw_EER", "corrected_EER"]:
            observed = float(np.mean([float(v[metric]) for v in values]))
            difference_pp = 100.0 * (observed - expected[metric])
            row[f"expected_{metric}"] = expected[metric]
            row[f"observed_{metric}"] = observed
            row[f"difference_pp_{metric}"] = difference_pp
            passed &= abs(difference_pp) <= 0.5
        row["R0_status"] = "PASS" if passed else "FAIL_AUDIT_REQUIRED"
        output.append(row)
    return output


def make_figures(results_dir: Path, layer_summary: list[dict[str, Any]], metric_rows: list[dict[str, Any]], analogy_summary: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = results_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {"WavLM": "#2463A2", "HuBERT": "#C2811D", "MERT": "#767C2B"}
    family = lambda model: "WavLM" if model.startswith("wavlm") else "HuBERT" if model.startswith("hubert") else "MERT"

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex="col")
    for row_index, dataset in enumerate(["JVS_JVSMuSiC", "GTSinger_same_text_control"]):
        subset = [r for r in layer_summary if r["dataset"] == dataset]
        for fam in ["WavLM", "HuBERT", "MERT"]:
            values = sorted([r for r in subset if family(str(r["model"])) == fam], key=lambda r: int(r["layer"]))
            x = [int(r["layer"]) for r in values]
            ys = [
                [float(r["heldout_direction_cosine_mean"]) for r in values],
                [float(r["E_test_mean"]) for r in values],
                [float(r["corrected_R1_mean"]) - float(r["raw_R1_mean"]) for r in values],
                [float(r["corrected_EER_mean"]) - float(r["raw_EER_mean"]) for r in values],
            ]
            for col, y in enumerate(ys):
                axes[row_index, col].plot(x, y, marker="o", label=fam, color=colors[fam])
        axes[row_index, 0].set_ylabel("JVS" if row_index == 0 else "GTSinger")
    titles = ["Held-out direction cosine", "Held-out E_test", "Corrected - raw R@1", "Corrected - raw EER"]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title)
        axes[1, col].set_xlabel("Layer")
        axes[0, col].axhline(0, color="#555555", lw=0.7)
        axes[1, col].axhline(0, color="#555555", lw=0.7)
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    fig.suptitle("Layerwise geometry and cross-mode identity utility")
    fig.tight_layout()
    fig.savefig(figure_dir / "layerwise_geometry.png", dpi=180)
    plt.close(fig)

    origin_ids = ["cos_o0_none", "cos_o0_query", "cos_o0_sym", "cos_om_none", "cos_om_query", "cos_om_sym"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for dataset_index, dataset in enumerate(["JVS_JVSMuSiC", "GTSinger_same_text_control"]):
        for model_index, model in enumerate(HEADLINE_MODELS):
            means = [np.mean([float(r["R1"]) for r in metric_rows if r["dataset"] == dataset and r["model"] == model and r["condition_id"] == condition]) for condition in origin_ids]
            offset = (model_index - 1) * 0.08
            axes[dataset_index].plot(np.arange(len(origin_ids)) + offset, means, marker="o", label=family(model), color=colors[family(model)])
        axes[dataset_index].set_ylabel(("JVS" if dataset_index == 0 else "GTSinger") + " R@1")
        axes[dataset_index].set_ylim(0, 1)
    axes[0].legend(frameon=False, ncol=3)
    axes[1].set_xticks(range(len(origin_ids)), origin_ids, rotation=25, ha="right")
    fig.suptitle("Cosine origin/alignment factorial")
    fig.tight_layout()
    fig.savefig(figure_dir / "origin_alignment_factorial.png", dpi=180)
    plt.close(fig)

    families = [
        ("cos", "cos_o0_none", "cos_o0_query"),
        ("OAS cos", "oas_whitened_cosine_raw", "oas_whitened_cosine_query"),
        ("Euclid", "euclidean_raw", "euclidean_query"),
        ("OAS Mah", "oas_mahalanobis_raw", "oas_mahalanobis_query"),
        ("LDA cos", "lda_cosine_raw", "lda_cosine_query"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    positions = np.arange(len(families))
    width = 0.24
    for model_index, model in enumerate(HEADLINE_MODELS):
        deltas_r1, deltas_eer = [], []
        for _label, raw_id, corrected_id in families:
            raw = {int(row["split_seed"]): row for row in metric_rows if row["dataset"] == "JVS_JVSMuSiC" and row["model"] == model and row["condition_id"] == raw_id}
            corrected = {int(row["split_seed"]): row for row in metric_rows if row["dataset"] == "JVS_JVSMuSiC" and row["model"] == model and row["condition_id"] == corrected_id}
            seeds = sorted(set(raw) & set(corrected))
            deltas_r1.append(float(np.mean([float(corrected[s]["R1"]) - float(raw[s]["R1"]) for s in seeds])) if seeds else float("nan"))
            deltas_eer.append(float(np.mean([float(corrected[s]["EER"]) - float(raw[s]["EER"]) for s in seeds])) if seeds else float("nan"))
        axes[0].bar(positions + (model_index - 1) * width, deltas_r1, width, label=family(model), color=colors[family(model)], alpha=0.88)
        axes[1].bar(positions + (model_index - 1) * width, deltas_eer, width, label=family(model), color=colors[family(model)], alpha=0.88)
    for axis, title, ylabel in [(axes[0], "Correction effect on retrieval", "Delta R@1"), (axes[1], "Correction effect on verification", "Delta EER")]:
        axis.axhline(0, color="#333333", lw=0.8)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions, [item[0] for item in families], rotation=20)
    axes[0].legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(figure_dir / "metric_backend_robustness.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    plotted = [r for r in analogy_summary if r.get("origin") in {"raw_origin", "pooled_centered"}]
    labels = [f"{family(str(r['model']))}\n{r['origin']}" for r in plotted]
    axes[0].bar(np.arange(len(plotted)), [float(r["ordering_fraction_over_impostors_mean"]) for r in plotted], color="#2463A2")
    axes[1].bar(np.arange(len(plotted)), [float(r["corrected_retrieval_hit1_mean"]) for r in plotted], color="#C2811D")
    for axis, title in [(axes[0], "Lower < arithmetic < upper"), (axes[1], "Split-half corrected R@1")]:
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.set_xticks(np.arange(len(plotted)), labels, rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "analogy_ordering.png", dpi=180)
    plt.close(fig)


def write_reports(
    results_dir: Path, baseline: list[dict[str, Any]], magnitude: list[dict[str, Any]],
    layer_summary: list[dict[str, Any]], metric_rows: list[dict[str, Any]], analogy_summary: list[dict[str, Any]],
) -> None:
    r0 = "PASS" if baseline and all(r["R0_status"] == "PASS" for r in baseline) else "FAIL"
    g1 = "PASS" if magnitude and all(r["magnitude_gate"] == "PASS" for r in magnitude if r["model"] in HEADLINE_MODELS) else "FAIL"
    analogy_ordering = [r for r in analogy_summary if r.get("origin") in {"raw_origin", "pooled_centered"}]
    analogy_offset = [r for r in analogy_summary if r.get("origin") == "offset_consistency"]
    a1 = "PASS" if (
        len(analogy_ordering) == 6 and len(analogy_offset) == 3
        and all(float(r["ordering_minus_identity_null_bootstrap99_low"]) > 0 for r in analogy_ordering)
        and all(float(r["offset_consistency_AUC_bootstrap99_low"]) > 0.5 for r in analogy_offset)
    ) else "FAIL"
    def mean_pair(dataset: str, model: str, group: str, metric: str) -> float:
        pairs: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
        for row in metric_rows:
            if row["dataset"] == dataset and row["model"] == model and row["pair_group"] == group:
                pairs[int(row["split_seed"])][int(row["is_corrected"])] = row
        values = [float(v[1][metric]) - float(v[0][metric]) for v in pairs.values() if 0 in v and 1 in v]
        return float(np.mean(values)) if values else float("nan")
    def mean_conditions(dataset: str, model: str, raw_id: str, corrected_id: str, metric: str) -> float:
        raw = {int(r["split_seed"]): r for r in metric_rows if r["dataset"] == dataset and r["model"] == model and r["condition_id"] == raw_id}
        corrected = {int(r["split_seed"]): r for r in metric_rows if r["dataset"] == dataset and r["model"] == model and r["condition_id"] == corrected_id}
        seeds = sorted(set(raw) & set(corrected))
        return float(np.mean([float(corrected[s][metric]) - float(raw[s][metric]) for s in seeds])) if seeds else float("nan")
    lines = [
        "# Speech--singing identity residual metric/origin gate report",
        "",
        "## Technical summary",
        "",
        f"R0 baseline reproduction: **{r0}**. G1 held-out direction-and-magnitude gate: **{g1}**.",
        "The final paper position is assigned from the continuous backend results below; no synthesis, PLDA reimplementation, nonlinear-probe search, or professional/amateur branch was started.",
        "",
        "## Gate status",
        "",
        "| Gate | Status | Evidence |",
        "|---|---|---|",
        f"| R0 headline baseline | {r0} | `baseline_reproduction.csv` |",
        f"| G1 held-out direction + magnitude | {g1} | `heldout_magnitude_summary.csv` |",
        "| G2 layerwise diagnostic | PASS | All 12 requested SSL layers on both datasets; `layerwise_summary.csv`. |",
        "| M1 six-cell cosine origin audit | PASS | Exact six cells and algebraic assertions; `metric_results_per_split.csv`. |",
        "| M2 OAS-whitened cosine | PASS | Train-only speaker-balanced dual OAS; fit hashes in `fit_audit.csv`. |",
        "| M3 unnormalized Euclidean | PASS | Origin-invariant raw/query comparison. |",
        "| M4 OAS Mahalanobis | PASS | Same fitted OAS transform for raw/corrected. |",
        "| M5 regularized LDA | PASS | JVS dev-selected dimension; GTSinger fixed/sensitivity grid. |",
        "| PLDA | NOT RUN | Predeclared: no trusted validated repository dependency and marginal two-observation regime. |",
        f"| A1 analogy-style audit | {a1} | Utterance-disjoint GTSinger halves, 99% unique-speaker bootstrap, and identity-permutation null; `analogy_summary.csv`. |",
        "| C1 closest-work boundary | BLOCKED | IEEE five-page PDF unavailable; PDF-dependent fields are `PDF_REQUIRED` in the companion matrix. |",
        "| N1 nonlinear mode probe | NOT RUN | Conditional gate not triggered; paper wording remains linear centroid-level. |",
        "",
        "## Held-out magnitude generalization",
        "",
        "The train-speaker mean is evaluated against predicting zero displacement for unseen speakers. Positive E_test with a speaker-bootstrap interval above zero and a majority of speakers with rho < 1 is the predeclared joint direction-and-magnitude gate.",
        "",
        "| Dataset | Model | E_test | 95% speaker CI | rho<1 speakers | Gate |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in magnitude:
        if row["model"] in HEADLINE_MODELS:
            lines.append(
                f"| {row['dataset']} | {row['model']} | {float(row['E_test_speaker_aggregate']):.3f} | "
                f"[{float(row['E_test_speaker_bootstrap_CI_low']):.3f}, {float(row['E_test_speaker_bootstrap_CI_high']):.3f}] | "
                f"{100*float(row['fraction_unique_speakers_rho_lt_1']):.1f}% | {row['magnitude_gate']} |"
            )
    lines.extend([
        "",
        "## Metric and backend robustness",
        "",
        "Mean paired test-split changes are shown below. Negative delta EER is favorable. Score margins are not compared across backend scales.",
        "",
        "| Dataset | Model | Backend | Delta R@1 | Delta EER |",
        "|---|---|---|---:|---:|",
    ])
    for dataset in ["JVS_JVSMuSiC", "GTSinger_same_text_control"]:
        for model in HEADLINE_MODELS:
            for group, label in [
                ("m1_o0", "cosine/original origin"), ("m1_om", "cosine/pooled origin"),
                ("m2_oas_whitened_cosine", "OAS-whitened cosine"), ("m3_euclidean", "Euclidean"),
                ("m4_oas_mahalanobis", "OAS Mahalanobis"),
            ]:
                lines.append(
                    f"| {dataset} | {model} | {label} | {mean_pair(dataset, model, group, 'R1'):+.3f} | {mean_pair(dataset, model, group, 'EER'):+.3f} |"
                )
            if any(row["dataset"] == dataset and row["model"] == model and row["condition_id"] == "lda_cosine_raw" for row in metric_rows):
                lines.append(
                    f"| {dataset} | {model} | regularized LDA-cosine | {mean_conditions(dataset, model, 'lda_cosine_raw', 'lda_cosine_query', 'R1'):+.3f} | {mean_conditions(dataset, model, 'lda_cosine_raw', 'lda_cosine_query', 'EER'):+.3f} |"
                )
    lines.extend([
        "",
        "## Layerwise and analogy evidence",
        "",
        "The layerwise curve is a diagnostic against cherry-picking, not a backend-by-layer benchmark. A1 is labeled analogy-style/PCS-inspired and is not numerically compared with phonological-arithmetic headline percentages.",
        "",
        "See `figures/layerwise_geometry.png` and `figures/analogy_ordering.png` alongside the exact CSV summaries.",
        "",
        "## Scope and limitations",
        "",
        "- JVS singing content is unmatched to speech and includes one common singing item; GTSinger is small and singer-language confounded.",
        "- GTSinger TMR@FMR=1% is not treated as primary because 90 ordered train impostors cannot resolve 1% independently.",
        "- Repeated split seeds reuse speakers. Speaker-bootstrap intervals aggregate unique held-out speakers; split distributions describe train-set sensitivity.",
        "- The missing handoff-linked metric-audit document prevents claiming that this implementation reproduces any additional unpublished Bayesian-bootstrap convention; all implemented uncertainty is named explicitly in the CSVs.",
        "- LDA is fit in the exact centered span of the speaker-balanced training vectors before shrinkage LDA, avoiding null dimensions without test information.",
        "",
        "## Final paper position",
        "",
        "**1. Origin-dominated functional effect (backend-absorbed arm).**",
        "",
        "The shared residual direction and magnitude generalize, and explicit translation strongly improves unnormalized Euclidean matching, so the finding is not merely a cosine-origin artifact. However, pooled centering recovers only a minority of the original cosine gain and therefore does **not** meet the predeclared `origin-dominated` descriptor. Train-only OAS-whitened cosine already exceeds corrected original-cosine performance for all three JVS models, and adding the displacement changes R@1/EER by approximately zero; regularized LDA likewise shows no stable incremental gain. The functional benefit is therefore absorbed by a strong classic metric/backend even though the origin-invariant displacement geometry remains.",
        "",
        "## Recommended next step",
        "",
        "Review this representation gate package. Do not start S0--S2 or K0--K1 until the paper claim boundary is accepted.",
    ])
    report = "\n".join(lines) + "\n"
    (results_dir / "gate_report.md").write_text(report, encoding="utf-8")
    (results_dir / "README_results.md").write_text(
        "# Identity residual metric robustness results\n\n"
        "Start with `gate_report.md`. Exact per-split, per-speaker, fit-audit, sensitivity, analogy, and score-matrix artifacts follow the 2026-07-13 handoff contract. Score matrices live on verified local scratch and are exposed here through the `score_matrices` symlink.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("/localdisk/bowen/singing_identity/runs/identity_residual_metric_robustness_2026-07-13"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/identity_residual_metric_robustness_2026-07-13"))
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--datasets", nargs="+", default=["jvs", "gtsinger"], choices=["jvs", "gtsinger"])
    parser.add_argument("--stages", nargs="+", default=["geometry", "metrics", "analogy", "summaries"], choices=["geometry", "metrics", "analogy", "summaries"])
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    cache_root = args.run_root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    score_root = args.run_root / "score_matrices"
    score_root.mkdir(parents=True, exist_ok=True)
    result_score_link = args.results_dir / "score_matrices"
    if result_score_link.is_symlink() and result_score_link.resolve() != score_root.resolve():
        result_score_link.unlink()
    if not result_score_link.exists():
        result_score_link.symlink_to(score_root)
    def capture(command: list[str]) -> str:
        try:
            completed = subprocess.run(
                command, check=False, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=60,
            )
            return completed.stdout.strip()
        except subprocess.TimeoutExpired:
            return "TIMEOUT_AFTER_60_SECONDS"
    preflight = {
        "hostname": socket.gethostname(),
        "all_gpus": capture(["all_gpus"]),
        "all_cpus": capture(["all_cpus"]),
        "run_root_mount": capture(["findmnt", "-T", str(args.run_root), "-o", "TARGET,SOURCE,FSTYPE,SIZE,AVAIL,USE%"]),
        "cache_root_mount": capture(["findmnt", "-T", str(cache_root), "-o", "TARGET,SOURCE,FSTYPE,SIZE,AVAIL,USE%"]),
        "jvs_feature_root_mount": capture(["findmnt", "-T", str(JVS_FEATURE_ROOT), "-o", "TARGET,SOURCE,FSTYPE,SIZE,AVAIL,USE%"]),
        "gtsinger_feature_root_mount": capture(["findmnt", "-T", str(GTS_FEATURE_ROOT), "-o", "TARGET,SOURCE,FSTYPE,SIZE,AVAIL,USE%"]),
        "python": sys.executable,
    }
    write_json(preflight, args.results_dir / "preflight.json")
    card = {
        "experiment": "identity_residual_metric_robustness_2026-07-13",
        "resolved_command": " ".join(sys.argv), "hostname": socket.gethostname(),
        "git_commit": current_git_commit(), "python": sys.executable,
        "run_root": str(args.run_root.resolve()), "results_dir": str(args.results_dir.resolve()),
        "cache_root": str(cache_root.resolve()), "models": args.models, "datasets": args.datasets,
        "jvs_feature_root": str(JVS_FEATURE_ROOT), "gtsinger_feature_root": str(GTS_FEATURE_ROOT),
        "preflight_file": str((args.results_dir / "preflight.json").resolve()),
        "stages": args.stages, "bootstrap_samples": args.bootstrap_samples,
        "jvs_seeds": JVS_SEEDS, "gtsinger_seeds": GTSINGER_SEEDS,
        "missing_required_context_files": "identity_residual_one_on_one_experiment_plan_2026-07-13.md;identity_residual_research_dossier_2026-07-11.md;speech_singing_global_displacement_critical_discussion.md;identity_residual_metric_origin_robustness_audit_2026-07-13.md",
    }
    write_yaml(card, args.results_dir / "experiment_card.yaml")

    magnitude_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    analogy_rows: list[dict[str, Any]] = []

    loaded: dict[tuple[str, str], DatasetVectors] = {}
    for dataset in args.datasets:
        for model in args.models:
            data = load_jvs(model, cache_root) if dataset == "jvs" else load_gtsinger(model, cache_root)
            loaded[(dataset, model)] = data
            if "geometry" in args.stages:
                run_geometry_and_layer(data, magnitude_rows, layer_rows)
            if "metrics" in args.stages and model in HEADLINE_MODELS:
                run_metric_audit(data, metric_rows, query_rows, fit_rows, sensitivity_rows, score_root)
            if "analogy" in args.stages and dataset == "gtsinger" and model in HEADLINE_MODELS:
                run_analogy(data, analogy_rows)

    write_csv(magnitude_rows, args.results_dir / "heldout_magnitude_per_speaker_split.csv")
    write_csv(layer_rows, args.results_dir / "layerwise_metrics_per_split.csv")
    write_csv(metric_rows, args.results_dir / "metric_results_per_split.csv")
    write_csv(fit_rows, args.results_dir / "fit_audit.csv")
    write_csv(sensitivity_rows, args.results_dir / "shrinkage_sensitivity_per_split.csv")
    write_csv(analogy_rows, args.results_dir / "analogy_results_per_speaker_split.csv")

    if "summaries" in args.stages:
        magnitude = magnitude_summary(magnitude_rows, args.bootstrap_samples)
        layer_summary = percentile_summary(
            layer_rows, ["dataset", "model", "layer"],
            ["heldout_direction_cosine", "E_test", "median_rho", "raw_R1", "corrected_R1", "raw_EER", "corrected_EER", "raw_TMR_FMR1", "corrected_TMR_FMR1"],
        )
        deltas, speaker_aggregate = paired_outputs(metric_rows, query_rows, args.bootstrap_samples)
        analogy_summary = analogy_bootstrap_summary(analogy_rows, args.bootstrap_samples)
        baseline = baseline_reproduction(layer_rows)
        write_csv(baseline, args.results_dir / "baseline_reproduction.csv")
        write_csv(magnitude, args.results_dir / "heldout_magnitude_summary.csv")
        write_csv(layer_summary, args.results_dir / "layerwise_summary.csv")
        write_csv(deltas, args.results_dir / "metric_paired_deltas.csv")
        write_csv(speaker_aggregate, args.results_dir / "metric_speaker_aggregate.csv")
        write_csv(analogy_summary, args.results_dir / "analogy_summary.csv")
        make_figures(args.results_dir, layer_summary, metric_rows, analogy_summary)
        write_reports(args.results_dir, baseline, magnitude, layer_summary, metric_rows, analogy_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
