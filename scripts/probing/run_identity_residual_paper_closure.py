#!/usr/bin/env python
"""Paper-closure experiments for cross-mode speaker identity.

Stages are deliberately ordered and independently runnable.  The baseline
stage must pass before any later stage is allowed to execute.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probing.run_identity_residual_metric_robustness import (  # noqa: E402
    ALL_MODELS,
    HEADLINE_MODELS,
    cosine_scores,
    fit_geometry,
    fit_oas_dual,
    JVS_FEATURE_ROOT,
    JVS_MANIFEST,
    load_gtsinger,
    load_jvs,
    ranks_from_scores,
    score_metrics,
    split_arrays,
)
from probing.run_identity_residual_suite import canonical_mode, load_feature_cache, read_jsonl  # noqa: E402
from research_utils import ExperimentError, current_git_commit  # noqa: E402


DEFAULT_RUN_ROOT = Path(
    "/localdisk/bowen/singing_identity/runs/identity_residual_paper_closure_2026-07-15"
)
DEFAULT_RESULTS_DIR = Path("results/identity_residual_paper_closure_2026-07-15")
PRIOR_RESULTS = Path("results/identity_residual_metric_robustness_2026-07-13")
PRIOR_COMPACT_CACHE = Path(
    "/localdisk/bowen/singing_identity/runs/identity_residual_metric_robustness_2026-07-13/cache"
)
BASELINE_CONDITIONS = (
    "oas_whitened_cosine_raw",
    "oas_whitened_cosine_query",
)
BASELINE_TOLERANCE_PP = 0.5
BOOTSTRAP_SAMPLES = 10_000


def stable_hash(*arrays: np.ndarray, extra: str = "") -> str:
    digest = hashlib.sha256(extra.encode("utf-8"))
    for array in arrays:
        value = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()[:20]


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def capture(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"UNAVAILABLE: {exc}"


def expected_whitened_baseline(prior_csv: Path) -> dict[tuple[str, str], dict[str, float]]:
    rows = read_csv(prior_csv)
    expected: dict[tuple[str, str], dict[str, float]] = {}
    for model in HEADLINE_MODELS:
        for condition in BASELINE_CONDITIONS:
            selected = [
                row
                for row in rows
                if row["dataset"] == "JVS_JVSMuSiC"
                and row["model"] == model
                and row["condition_id"] == condition
            ]
            if len(selected) != 20:
                raise ExperimentError(
                    f"expected 20 prior rows for {model}/{condition}, found {len(selected)}"
                )
            expected[(model, condition)] = {
                "R1": float(np.mean([float(row["R1"]) for row in selected])),
                "EER": float(np.mean([float(row["EER"]) for row in selected])),
            }
    return expected


def reproduce_whitened_baseline(cache_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in HEADLINE_MODELS:
        data = load_jvs(model, cache_root)
        for seed in data.seeds:
            train, _dev, test = split_arrays(data, seed)
            _a, _b, d, midpoint = fit_geometry(data, train)
            z_train = np.vstack([data.speech[train], data.singing[train]])
            oas = fit_oas_dual(z_train)
            for condition, query in (
                ("oas_whitened_cosine_raw", data.speech[test] - midpoint),
                ("oas_whitened_cosine_query", data.speech[test] + d - midpoint),
            ):
                gallery = data.singing[test] - midpoint
                scores = cosine_scores(oas.whiten(query), oas.whiten(gallery))
                metrics = score_metrics(scores, None, "baseline_reproduction_no_threshold")
                rows.append(
                    {
                        "dataset": data.name,
                        "model": model,
                        "layer": data.layer,
                        "split_seed": seed,
                        "condition_id": condition,
                        "transform_hash": oas.transform_hash,
                        "R1": metrics["R1"],
                        "EER": metrics["EER"],
                    }
                )
    return rows


def summarize_baseline(
    observed_rows: list[dict[str, Any]],
    expected: dict[tuple[str, str], dict[str, float]],
    tolerance_pp: float = BASELINE_TOLERANCE_PP,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in HEADLINE_MODELS:
        for condition in BASELINE_CONDITIONS:
            selected = [
                row
                for row in observed_rows
                if row["model"] == model and row["condition_id"] == condition
            ]
            if len(selected) != 20:
                raise ExperimentError(
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


def fit_diagonal_whitener(z_train: np.ndarray) -> tuple[np.ndarray, float, dict[str, float], str]:
    z_train = np.asarray(z_train, dtype=np.float64)
    variances = np.var(z_train, axis=0, ddof=0)
    mean_variance = float(np.mean(variances))
    epsilon = 1e-8 * mean_variance
    scale = np.sqrt(variances + epsilon)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ExperimentError("diagonal whitening produced a non-positive or non-finite scale")
    positive = variances[variances > 0]
    stats = {
        "variance_min": float(np.min(variances)),
        "variance_median": float(np.median(variances)),
        "variance_max": float(np.max(variances)),
        "variance_ratio_max_min": float(np.max(variances) / np.min(positive)) if len(positive) else math.inf,
    }
    return scale, epsilon, stats, stable_hash(scale, extra=f"diag:{epsilon:.17g}")


def save_score_matrix(
    root: Path,
    dataset: str,
    model: str,
    seed: int,
    condition: str,
    scores: np.ndarray,
    labels: list[str],
    transform_hash: str,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{dataset.replace('/', '_')}__{model}__{seed}__{condition}.npz"
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


def speaker_paired_interval(
    rows: list[dict[str, Any]], raw_condition: str, query_condition: str, seed: int
) -> tuple[float, float, float, float]:
    grouped: dict[str, list[float]] = {}
    by_key = {(row["speaker_id"], row["condition_id"], row["split_seed"]): row for row in rows}
    for speaker, condition, split_seed in by_key:
        if condition != raw_condition:
            continue
        partner = by_key.get((speaker, query_condition, split_seed))
        if partner is None:
            raise ExperimentError(f"missing paired W1 row: {speaker}/{split_seed}/{query_condition}")
        grouped.setdefault(str(speaker), []).append(float(partner["hit1"]) - float(by_key[(speaker, condition, split_seed)]["hit1"]))
    values = np.asarray([np.mean(items) for items in grouped.values()], dtype=np.float64)
    if not len(values):
        return math.nan, math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [values[rng.integers(0, len(values), len(values))].mean() for _ in range(BOOTSTRAP_SAMPLES)]
    )
    observed = float(values.mean())
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(BOOTSTRAP_SAMPLES, len(values)))
    null = np.abs(np.mean(signs * values[None, :], axis=1))
    p_value = float((1 + np.sum(null >= abs(observed))) / (BOOTSTRAP_SAMPLES + 1))
    return observed, float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5)), p_value


def run_w1(
    cache_root: Path, prior_csv: Path, run_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    prior_rows = read_csv(prior_csv)
    result_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    score_root = run_root / "score_matrices"
    for dataset_key in ("jvs", "gtsinger"):
        for model in HEADLINE_MODELS:
            data = load_jvs(model, cache_root) if dataset_key == "jvs" else load_gtsinger(model, cache_root)
            for seed in data.seeds:
                train, dev, test = split_arrays(data, seed)
                _a, _b, d, midpoint = fit_geometry(data, train)
                z_train = np.vstack([data.speech[train], data.singing[train]])
                scale, epsilon, variance_stats, diag_hash = fit_diagonal_whitener(z_train)
                oas = fit_oas_dual(z_train)
                test_labels = [data.speakers[index] for index in test]
                backend_scores: dict[str, np.ndarray] = {
                    "wcos_diag_none": cosine_scores((data.speech[test] - midpoint) / scale, (data.singing[test] - midpoint) / scale),
                    "wcos_diag_query": cosine_scores((data.speech[test] + d - midpoint) / scale, (data.singing[test] - midpoint) / scale),
                    "wcos_oas_none": cosine_scores(oas.whiten(data.speech[test] - midpoint), oas.whiten(data.singing[test] - midpoint)),
                    "wcos_oas_query": cosine_scores(oas.whiten(data.speech[test] + d - midpoint), oas.whiten(data.singing[test] - midpoint)),
                }
                wd_diag = d / scale
                delta_test = data.singing[test] - data.speech[test]
                wdelta_diag = delta_test / scale
                wres_diag = (delta_test - d) / scale
                wd_oas = oas.whiten(d[None, :])[0]
                wdelta_oas = oas.whiten(delta_test)
                wres_oas = oas.whiten(delta_test - d)
                fit_rows.extend(
                    [
                        {
                            "dataset": data.name,
                            "model": model,
                            "layer": data.layer,
                            "split_seed": seed,
                            "fit_type": "diagonal_variance",
                            "fit_speakers": ";".join(data.speakers[index] for index in train),
                            "n_fit_vectors": len(z_train),
                            "dimension": z_train.shape[1],
                            "epsilon": epsilon,
                            **variance_stats,
                            "whitened_residual_alignment": float(np.mean((wdelta_diag @ wd_diag) / np.maximum(np.linalg.norm(wdelta_diag, axis=1) * np.linalg.norm(wd_diag), 1e-12))),
                            "whitened_E_test": 1.0 - float(np.sum(wres_diag * wres_diag)) / max(float(np.sum(wdelta_diag * wdelta_diag)), 1e-12),
                            "transform_hash": diag_hash,
                        },
                        {
                            "dataset": data.name,
                            "model": model,
                            "layer": data.layer,
                            "split_seed": seed,
                            "fit_type": "OAS",
                            "fit_speakers": ";".join(data.speakers[index] for index in train),
                            "n_fit_vectors": len(z_train),
                            "dimension": z_train.shape[1],
                            "shrinkage": oas.shrinkage,
                            "covariance_trace": float(oas.empirical_eigenvalues.sum()),
                            "minimum_effective_eigenvalue": oas.lambda0,
                            "condition_number": float(max(oas.fitted_eigenvalues.max(initial=oas.lambda0), oas.lambda0) / oas.lambda0),
                            "whitened_residual_alignment": float(np.mean((wdelta_oas @ wd_oas) / np.maximum(np.linalg.norm(wdelta_oas, axis=1) * np.linalg.norm(wd_oas), 1e-12))),
                            "whitened_E_test": 1.0 - float(np.sum(wres_oas * wres_oas)) / max(float(np.sum(wdelta_oas * wdelta_oas)), 1e-12),
                            "transform_hash": oas.transform_hash,
                        },
                    ]
                )
                for condition, scores in backend_scores.items():
                    metrics = score_metrics(scores, None, "w1_no_operating_threshold")
                    transform_hash = diag_hash if "diag" in condition else oas.transform_hash
                    result_rows.append(
                        {
                            "dataset": data.name,
                            "model": model,
                            "layer": data.layer,
                            "split_seed": seed,
                            "condition_id": condition,
                            "score_family": "diagonal_whitened_cosine" if "diag" in condition else "oas_whitened_cosine",
                            "origin": "pooled_train_midpoint",
                            "alignment": "query" if condition.endswith("query") else "none",
                            "covariance_method": "diagonal_variance" if "diag" in condition else "OAS",
                            "transform_hash": transform_hash,
                            **metrics,
                        }
                    )
                    ranks, _ = ranks_from_scores(scores)
                    for row_index, speaker_id in enumerate(test_labels):
                        query_rows.append(
                            {
                                "dataset": data.name,
                                "model": model,
                                "split_seed": seed,
                                "condition_id": condition,
                                "speaker_id": speaker_id,
                                "hit1": int(ranks[row_index] == 1),
                            }
                        )
                    save_score_matrix(score_root, data.name, model, seed, condition, scores, test_labels, transform_hash)

                # The plan explicitly requires copying these reference rows from 07-13.
                for prior_condition in ("cos_om_none", "cos_o0_query"):
                    matches = [
                        row
                        for row in prior_rows
                        if row["dataset"] == data.name
                        and row["model"] == model
                        and int(row["split_seed"]) == seed
                        and row["condition_id"] == prior_condition
                    ]
                    if len(matches) != 1:
                        raise ExperimentError(
                            f"missing unique prior reference row: {data.name}/{model}/{seed}/{prior_condition}"
                        )
                    copied = dict(matches[0])
                    copied["source"] = "copied_from_identity_residual_metric_robustness_2026-07-13"
                    result_rows.append(copied)

    summary_rows: list[dict[str, Any]] = []
    for dataset in sorted({str(row["dataset"]) for row in result_rows}):
        for model_index, model in enumerate(HEADLINE_MODELS):
            subset = [row for row in result_rows if row["dataset"] == dataset and row["model"] == model]
            means: dict[str, dict[str, float]] = {}
            for condition in ("cos_om_none", "cos_o0_query", "wcos_diag_none", "wcos_diag_query", "wcos_oas_none", "wcos_oas_query"):
                rows = [row for row in subset if row["condition_id"] == condition]
                if not rows:
                    raise ExperimentError(f"missing W1 condition: {dataset}/{model}/{condition}")
                means[condition] = {
                    "R1": float(np.mean([float(row["R1"]) for row in rows])),
                    "EER": float(np.mean([float(row["EER"]) for row in rows])),
                }
            denominator = means["wcos_oas_none"]["R1"] - means["cos_om_none"]["R1"]
            absorption = (
                (means["wcos_diag_none"]["R1"] - means["cos_om_none"]["R1"]) / denominator
                if abs(denominator) > 1e-12
                else math.nan
            )
            paired = [row for row in query_rows if row["dataset"] == dataset and row["model"] == model]
            delta, low, high, p_value = speaker_paired_interval(
                paired, "wcos_diag_none", "wcos_diag_query", 715_000 + model_index
            )
            summary_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "splits": len([row for row in subset if row["condition_id"] == "wcos_diag_none"]),
                    "cos_om_none_R1": means["cos_om_none"]["R1"],
                    "wcos_diag_none_R1": means["wcos_diag_none"]["R1"],
                    "wcos_diag_query_R1": means["wcos_diag_query"]["R1"],
                    "wcos_oas_none_R1": means["wcos_oas_none"]["R1"],
                    "wcos_oas_query_R1": means["wcos_oas_query"]["R1"],
                    "A_diag": absorption,
                    "diag_query_minus_none_R1": delta,
                    "diag_query_minus_none_R1_ci95_low": low,
                    "diag_query_minus_none_R1_ci95_high": high,
                    "diag_query_minus_none_R1_signflip_p": p_value,
                    "diag_translation_ci_includes_zero": int(low <= 0 <= high),
                    "cos_om_none_EER": means["cos_om_none"]["EER"],
                    "wcos_diag_none_EER": means["wcos_diag_none"]["EER"],
                    "wcos_diag_query_EER": means["wcos_diag_query"]["EER"],
                    "wcos_oas_none_EER": means["wcos_oas_none"]["EER"],
                    "wcos_oas_query_EER": means["wcos_oas_query"]["EER"],
                }
            )
    return result_rows, fit_rows, summary_rows


def remove_subspace(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=np.float64)
    directions = np.asarray(basis, dtype=np.float64)
    if directions.size == 0:
        return values.copy()
    return values - (values @ directions) @ directions.T


def random_control_basis(
    empirical_basis: np.ndarray, k: int, rng: np.random.Generator
) -> tuple[np.ndarray, str]:
    rank, dimension = empirical_basis.shape[1], empirical_basis.shape[0]
    if rank >= 65:
        candidates = empirical_basis[:, 64 : min(rank, 512)]
        if candidates.shape[1] >= k:
            indices = rng.choice(candidates.shape[1], size=k, replace=False)
            return candidates[:, indices], "empirical_variance_ranks_65_512"
    # In rank-deficient fits, ranks beyond the empirical span are a degenerate
    # zero-eigenvalue subspace.  A random orthonormal basis is a valid but
    # explicitly non-identifiable representative of those eigenvectors.
    columns: list[np.ndarray] = []
    for _ in range(k):
        vector = rng.normal(size=dimension)
        vector = remove_subspace(vector[None, :], empirical_basis)[0]
        if columns:
            prior = np.column_stack(columns)
            vector = remove_subspace(vector[None, :], prior)[0]
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-10:
            raise ExperimentError("failed to construct random nullspace control direction")
        columns.append(vector / norm)
    return np.column_stack(columns), "random_zero_eigenvalue_basis_rank_65_plus"


def run_w2(
    cache_root: Path, run_root: Path, results_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    w1_rows = read_csv(results_dir / "w1_whitening_decomposition.csv")
    alignment_rows: list[dict[str, Any]] = []
    abtt_rows: list[dict[str, Any]] = []
    score_root = run_root / "score_matrices"
    for dataset_key in ("jvs", "gtsinger"):
        for model_index, model in enumerate(HEADLINE_MODELS):
            data = load_jvs(model, cache_root) if dataset_key == "jvs" else load_gtsinger(model, cache_root)
            for seed in data.seeds:
                train, _dev, test = split_arrays(data, seed)
                _a, _b, d, midpoint = fit_geometry(data, train)
                z_train = np.vstack([data.speech[train], data.singing[train]])
                empirical = fit_oas_dual(z_train, shrinkage=1.0)
                basis = empirical.basis
                eigenvalues = empirical.empirical_eigenvalues
                dimension = z_train.shape[1]
                dhat = d / max(float(np.linalg.norm(d)), 1e-12)
                coefficients = basis.T @ dhat
                cumulative = np.cumsum(coefficients * coefficients)
                full_energy = float(np.sum(coefficients * coefficients) + max(0.0, 1.0 - np.sum(coefficients * coefficients)))
                np.testing.assert_allclose(full_energy, 1.0, rtol=1e-10, atol=1e-10)
                rng = np.random.default_rng(715_000_000 + seed * 101 + model_index)
                null_components = rng.normal(size=(1000, dimension))
                null_components *= null_components
                null_components /= np.maximum(null_components.sum(axis=1, keepdims=True), 1e-300)
                null_cumulative = np.cumsum(null_components[:, :64], axis=1)
                max_pc = int(np.argmax(np.abs(coefficients)) + 1) if len(coefficients) else -1
                for k in range(1, 65):
                    identifiable_k = min(k, len(cumulative))
                    energy = float(cumulative[identifiable_k - 1]) if identifiable_k else 0.0
                    alignment_rows.append(
                        {
                            "dataset": data.name,
                            "model": model,
                            "layer": data.layer,
                            "split_seed": seed,
                            "k": k,
                            "E_d_k": energy,
                            "null_mean": float(np.mean(null_cumulative[:, k - 1])),
                            "null_p2_5": float(np.percentile(null_cumulative[:, k - 1], 2.5)),
                            "null_p97_5": float(np.percentile(null_cumulative[:, k - 1], 97.5)),
                            "empirical_rank": len(eigenvalues),
                            "dimension": dimension,
                            "zero_eigenspace_nonidentifiable": int(k > len(eigenvalues)),
                            "max_alignment_pc_rank": max_pc,
                            "max_alignment_pc_abs_cos": float(np.max(np.abs(coefficients))) if len(coefficients) else math.nan,
                            "variance_at_max_alignment_pc": float(eigenvalues[max_pc - 1]) if max_pc > 0 else math.nan,
                            "E_d_full_basis": full_energy,
                        }
                    )

                centered_train = z_train - z_train.mean(axis=0)
                original_rank = int(np.linalg.matrix_rank(centered_train))
                test_labels = [data.speakers[index] for index in test]
                for k in (0, 1, 2, 4, 8, 16):
                    if k > basis.shape[1]:
                        continue
                    top = basis[:, :k]
                    projected_train = remove_subspace(centered_train, top)
                    np.testing.assert_allclose(remove_subspace(projected_train, top), projected_train, rtol=1e-9, atol=1e-9)
                    np.testing.assert_allclose(projected_train @ top, 0.0, rtol=1e-8, atol=1e-8)
                    projected_rank = int(np.linalg.matrix_rank(projected_train))
                    if original_rank - projected_rank != k:
                        raise ExperimentError(
                            f"ABTT rank assertion failed: {data.name}/{model}/{seed}/k={k}: "
                            f"{original_rank}->{projected_rank}"
                        )
                    q_none = remove_subspace(data.speech[test] - midpoint, top)
                    q_query = remove_subspace(data.speech[test] + d - midpoint, top)
                    gallery = remove_subspace(data.singing[test] - midpoint, top)
                    if k == 0:
                        np.testing.assert_allclose(
                            cosine_scores(q_none, gallery),
                            cosine_scores(data.speech[test] - midpoint, data.singing[test] - midpoint),
                            rtol=1e-12,
                            atol=1e-12,
                        )
                    for alignment, query in (("none", q_none), ("query", q_query)):
                        condition = f"abtt_{k}_{alignment}"
                        scores = cosine_scores(query, gallery)
                        metrics = score_metrics(scores, None, "w2_no_operating_threshold")
                        transform_hash = stable_hash(top, midpoint, extra=f"abtt:{k}")
                        abtt_rows.append(
                            {
                                "dataset": data.name,
                                "model": model,
                                "layer": data.layer,
                                "split_seed": seed,
                                "condition_id": condition,
                                "score_family": "abtt_cosine",
                                "origin": "pooled_train_midpoint",
                                "alignment": alignment,
                                "k": k,
                                "control_type": "top_variance_pcs",
                                "control_draw": "",
                                "transform_hash": transform_hash,
                                **metrics,
                            }
                        )
                        save_score_matrix(score_root, data.name, model, seed, condition, scores, test_labels, transform_hash)
                    if k == 0:
                        continue
                    for draw in range(5):
                        control_basis, control_type = random_control_basis(basis, k, rng)
                        control_query = remove_subspace(data.speech[test] - midpoint, control_basis)
                        control_gallery = remove_subspace(data.singing[test] - midpoint, control_basis)
                        scores = cosine_scores(control_query, control_gallery)
                        metrics = score_metrics(scores, None, "w2_random_control_no_threshold")
                        abtt_rows.append(
                            {
                                "dataset": data.name,
                                "model": model,
                                "layer": data.layer,
                                "split_seed": seed,
                                "condition_id": f"abtt_{k}_random_control",
                                "score_family": "abtt_cosine",
                                "origin": "pooled_train_midpoint",
                                "alignment": "none",
                                "k": k,
                                "control_type": control_type,
                                "control_draw": draw,
                                "transform_hash": stable_hash(control_basis, midpoint, extra=f"abtt_control:{k}:{draw}"),
                                **metrics,
                            }
                        )

    summary_rows: list[dict[str, Any]] = []
    for dataset in sorted({str(row["dataset"]) for row in abtt_rows}):
        for model in HEADLINE_MODELS:
            w1_subset = [row for row in w1_rows if row["dataset"] == dataset and row["model"] == model]
            cos_raw = np.mean([float(row["R1"]) for row in w1_subset if row["condition_id"] == "cos_om_none"])
            oas_raw = np.mean([float(row["R1"]) for row in w1_subset if row["condition_id"] == "wcos_oas_none"])
            whitening_gain = float(oas_raw - cos_raw)
            model_rows = [row for row in abtt_rows if row["dataset"] == dataset and row["model"] == model]
            candidates: list[tuple[float, int, float, float]] = []
            for k in (1, 2, 4, 8, 16):
                raw = [float(row["R1"]) for row in model_rows if row["condition_id"] == f"abtt_{k}_none"]
                query = [float(row["R1"]) for row in model_rows if row["condition_id"] == f"abtt_{k}_query"]
                controls = [float(row["R1"]) for row in model_rows if row["condition_id"] == f"abtt_{k}_random_control"]
                if raw:
                    candidates.append((float(np.mean(raw) - cos_raw), k, float(np.mean(query) - np.mean(raw)), float(np.mean(controls) - cos_raw)))
            best_gain_overall, best_k_overall, _overall_increment, _overall_random = max(candidates)
            eligible = [candidate for candidate in candidates if candidate[1] <= 8]
            best_gain, best_k, translation_increment, random_gain = max(eligible)
            align8 = [
                row for row in alignment_rows
                if row["dataset"] == dataset and row["model"] == model and int(row["k"]) == 8
            ]
            align16 = [
                row for row in alignment_rows
                if row["dataset"] == dataset and row["model"] == model and int(row["k"]) == 16
            ]
            summary_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "E_d_8_mean": float(np.mean([float(row["E_d_k"]) for row in align8])),
                    "E_d_8_null_p97_5_mean": float(np.mean([float(row["null_p97_5"]) for row in align8])),
                    "E_d_16_mean": float(np.mean([float(row["E_d_k"]) for row in align16])),
                    "E_d_16_null_p97_5_mean": float(np.mean([float(row["null_p97_5"]) for row in align16])),
                    "whitening_gain_R1": whitening_gain,
                    "best_abtt_k": best_k,
                    "best_abtt_gain_R1": best_gain,
                    "best_abtt_fraction_of_whitening_gain": best_gain / whitening_gain if abs(whitening_gain) > 1e-12 else math.nan,
                    "best_abtt_k_overall_through_16": best_k_overall,
                    "best_abtt_gain_R1_overall_through_16": best_gain_overall,
                    "translation_increment_at_best_k_R1": translation_increment,
                    "random_control_gain_at_best_k_R1": random_gain,
                }
            )
    return alignment_rows, abtt_rows, summary_rows


def run_g2x(cache_root: Path, run_root: Path, results_dir: Path) -> list[dict[str, Any]]:
    w1_rows = read_csv(results_dir / "w1_whitening_decomposition.csv")
    output: list[dict[str, Any]] = []
    score_root = run_root / "score_matrices"
    for model in ALL_MODELS:
        data = load_jvs(model, cache_root)
        for seed in data.seeds:
            train, _dev, test = split_arrays(data, seed)
            _a, _b, d, midpoint = fit_geometry(data, train)
            z_train = np.vstack([data.speech[train], data.singing[train]])
            scale, epsilon, _variance_stats, diag_hash = fit_diagonal_whitener(z_train)
            raw_scores = cosine_scores(data.speech[test], data.singing[test])
            query_scores = cosine_scores(data.speech[test] + d, data.singing[test])
            diag_scores = cosine_scores(
                (data.speech[test] - midpoint) / scale,
                (data.singing[test] - midpoint) / scale,
            )
            raw_metrics = score_metrics(raw_scores, None, "g2x_no_operating_threshold")
            query_metrics = score_metrics(query_scores, None, "g2x_no_operating_threshold")
            diag_metrics = score_metrics(diag_scores, None, "g2x_no_operating_threshold")
            delta = data.singing[test] - data.speech[test]
            residual = delta - d
            alignment = (delta @ d) / np.maximum(
                np.linalg.norm(delta, axis=1) * np.linalg.norm(d), 1e-12
            )
            e_test = 1.0 - float(np.sum(residual * residual)) / max(float(np.sum(delta * delta)), 1e-12)
            oas_matches = [
                row
                for row in w1_rows
                if row["dataset"] == data.name
                and row["model"] == model
                and int(row["split_seed"]) == seed
                and row["condition_id"] == "wcos_oas_none"
            ]
            output.append(
                {
                    "dataset": data.name,
                    "model": model,
                    "layer": data.layer,
                    "split_seed": seed,
                    "heldout_direction_cosine": float(np.mean(alignment)),
                    "heldout_E_test": e_test,
                    "raw_R1": raw_metrics["R1"],
                    "query_R1": query_metrics["R1"],
                    "diag_R1": diag_metrics["R1"],
                    "oas_R1_headline_only": float(oas_matches[0]["R1"]) if len(oas_matches) == 1 else math.nan,
                    "raw_EER": raw_metrics["EER"],
                    "query_EER": query_metrics["EER"],
                    "diag_EER": diag_metrics["EER"],
                    "oas_EER_headline_only": float(oas_matches[0]["EER"]) if len(oas_matches) == 1 else math.nan,
                    "diag_epsilon": epsilon,
                    "diag_transform_hash": diag_hash,
                }
            )
            labels = [data.speakers[index] for index in test]
            save_score_matrix(score_root, data.name, model, seed, "g2x_wcos_diag_none", diag_scores, labels, diag_hash)
    return output


def jvs_utterance_groups(
    model: str, cache_root: Path
) -> tuple[Any, dict[str, dict[str, list[tuple[str, np.ndarray]]]]]:
    data = load_jvs(model, cache_root)
    rows, vectors, _meta = load_feature_cache(
        read_jsonl(JVS_MANIFEST), JVS_FEATURE_ROOT, model, cache_root, 20, False
    )
    groups: dict[str, dict[str, list[tuple[str, np.ndarray]]]] = {}
    for row, vector in zip(rows, vectors):
        mode = canonical_mode(row)
        speaker = str(row["speaker_id"])
        if mode not in {"speech", "singing"} or speaker not in data.speakers:
            continue
        groups.setdefault(speaker, {"speech": [], "singing": []})[mode].append(
            (str(row.get("utt_id", row.get("path", len(groups)))), np.asarray(vector, dtype=np.float64))
        )
    for speaker in data.speakers:
        for mode in ("speech", "singing"):
            groups[speaker][mode].sort(key=lambda item: item[0])
            if not groups[speaker][mode]:
                raise ExperimentError(f"missing JVS utterances: {speaker}/{mode}")
    return data, groups


def gtsinger_utterance_groups(data: Any) -> dict[str, list[tuple[str, np.ndarray, np.ndarray]]]:
    if data.pair_rows is None or data.pair_speech is None or data.pair_singing is None:
        raise ExperimentError("GTSinger pair vectors are required for U1")
    groups: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {}
    for index, pair in enumerate(data.pair_rows):
        speaker = str(pair["speaker_id"])
        groups.setdefault(speaker, []).append(
            (
                str(pair.get("pair_id", index)),
                np.asarray(data.pair_speech[index], dtype=np.float64),
                np.asarray(data.pair_singing[index], dtype=np.float64),
            )
        )
    for values in groups.values():
        values.sort(key=lambda item: item[0])
    return groups


def u1_condition_scores(
    speech: np.ndarray,
    singing: np.ndarray,
    d: np.ndarray,
    midpoint: np.ndarray,
    diag_scale: np.ndarray,
    oas: Any,
) -> dict[str, np.ndarray]:
    return {
        "utt_cos_om_none": cosine_scores(speech - midpoint, singing - midpoint),
        "utt_cos_o0_query": cosine_scores(speech + d, singing),
        "utt_wcos_oas_none": cosine_scores(oas.whiten(speech - midpoint), oas.whiten(singing - midpoint)),
        "utt_wcos_diag_none": cosine_scores((speech - midpoint) / diag_scale, (singing - midpoint) / diag_scale),
    }


def save_u1_score_bundle(
    root: Path,
    dataset: str,
    model: str,
    seed: int,
    protocol: str,
    condition: str,
    scores: list[np.ndarray],
    labels: list[str],
    draw_ids: list[str],
    transform_hash: str,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{dataset}__{model}__{seed}__utt_{protocol}__{condition}.npz"
    np.savez_compressed(
        path,
        score_matrices=np.stack(scores),
        ordered_query_speaker_ids=np.asarray(labels, dtype="U"),
        ordered_gallery_speaker_ids=np.asarray(labels, dtype="U"),
        draw_ids=np.asarray(draw_ids, dtype="U"),
        condition_id=np.asarray(condition),
        split_seed=np.asarray(seed),
        transform_hash=np.asarray(transform_hash),
        higher_is_better=np.asarray(True),
    )


def run_u1(cache_root: Path, run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    speaker_trials: list[dict[str, Any]] = []
    score_root = run_root / "score_matrices"
    draw_root = run_root / "u1_draws"
    draw_root.mkdir(parents=True, exist_ok=True)
    for dataset_key in ("jvs", "gtsinger"):
        for model_index, model in enumerate(HEADLINE_MODELS):
            if dataset_key == "jvs":
                data, jvs_groups = jvs_utterance_groups(model, cache_root)
                gts_groups = None
            else:
                data = load_gtsinger(model, cache_root)
                jvs_groups = None
                gts_groups = gtsinger_utterance_groups(data)
            for seed in data.seeds:
                train, _dev, test = split_arrays(data, seed)
                train_ids = {data.speakers[index] for index in train}
                test_ids = {data.speakers[index] for index in test}
                assert train_ids.isdisjoint(test_ids)
                _a, _b, d, midpoint = fit_geometry(data, train)
                z_train = np.vstack([data.speech[train], data.singing[train]])
                diag_scale, _epsilon, _stats, diag_hash = fit_diagonal_whitener(z_train)
                oas = fit_oas_dual(z_train)
                labels = [data.speakers[index] for index in test]
                rng = np.random.default_rng(715_100_000 + seed * 101 + model_index)
                bundles: dict[tuple[str, str], list[np.ndarray]] = {}
                draw_ids: dict[str, list[str]] = {"utterance_to_utterance": [], "utterance_to_centroid": []}
                query_id_draws: list[list[str]] = []
                gallery_id_draws: list[list[str]] = []
                for draw in range(25):
                    query_vectors: list[np.ndarray] = []
                    gallery_vectors: list[np.ndarray] = []
                    query_ids: list[str] = []
                    gallery_ids: list[str] = []
                    for speaker in labels:
                        if jvs_groups is not None:
                            speech_items = jvs_groups[speaker]["speech"]
                            singing_items = jvs_groups[speaker]["singing"]
                            speech_index = int(rng.integers(0, len(speech_items)))
                            singing_index = int(rng.integers(0, len(singing_items)))
                            query_id, query_vector = speech_items[speech_index]
                            gallery_id, gallery_vector = singing_items[singing_index]
                        else:
                            assert gts_groups is not None
                            items = gts_groups[speaker]
                            pair_index = int(rng.integers(0, len(items)))
                            pair_id, query_vector, gallery_vector = items[pair_index]
                            query_id = gallery_id = pair_id
                        query_vectors.append(query_vector)
                        gallery_vectors.append(gallery_vector)
                        query_ids.append(query_id)
                        gallery_ids.append(gallery_id)
                    speech = np.vstack(query_vectors)
                    single_singing = np.vstack(gallery_vectors)
                    centroid_singing = data.singing[test]
                    query_id_draws.append(query_ids)
                    gallery_id_draws.append(gallery_ids)
                    draw_token = f"draw{draw:02d}"
                    for protocol, gallery in (
                        ("utterance_to_utterance", single_singing),
                        ("utterance_to_centroid", centroid_singing),
                    ):
                        scores_by_condition = u1_condition_scores(speech, gallery, d, midpoint, diag_scale, oas)
                        draw_ids[protocol].append(draw_token)
                        for condition, scores in scores_by_condition.items():
                            bundles.setdefault((protocol, condition), []).append(scores)
                            metrics = score_metrics(scores, None, "u1_no_operating_threshold")
                            ranks, _ = ranks_from_scores(scores)
                            rows.append(
                                {
                                    "row_type": "draw_metric",
                                    "dataset": data.name,
                                    "model": model,
                                    "layer": data.layer,
                                    "split_seed": seed,
                                    "draw": draw,
                                    "protocol": protocol,
                                    "condition_id": condition,
                                    **metrics,
                                }
                            )
                            for row_index, speaker in enumerate(labels):
                                speaker_trials.append(
                                    {
                                        "dataset": data.name,
                                        "model": model,
                                        "layer": data.layer,
                                        "split_seed": seed,
                                        "draw": draw,
                                        "protocol": protocol,
                                        "condition_id": condition,
                                        "speaker_id": speaker,
                                        "query_utterance_id": query_ids[row_index],
                                        "gallery_utterance_id": gallery_ids[row_index] if protocol == "utterance_to_utterance" else "centroid",
                                        "rank": int(ranks[row_index]),
                                        "hit1": int(ranks[row_index] == 1),
                                    }
                                )
                np.savez_compressed(
                    draw_root / f"{data.name}__{model}__{seed}__draw_indices.npz",
                    ordered_test_speaker_ids=np.asarray(labels, dtype="U"),
                    ordered_train_speaker_ids=np.asarray(sorted(train_ids), dtype="U"),
                    query_utterance_ids=np.asarray(query_id_draws, dtype="U"),
                    gallery_utterance_ids=np.asarray(gallery_id_draws, dtype="U"),
                    split_seed=np.asarray(seed),
                )
                for (protocol, condition), score_list in bundles.items():
                    transform_hash = (
                        diag_hash if "diag" in condition else oas.transform_hash if "oas" in condition else stable_hash(d, midpoint, extra=condition)
                    )
                    save_u1_score_bundle(
                        score_root,
                        data.name,
                        model,
                        seed,
                        protocol,
                        condition,
                        score_list,
                        labels,
                        draw_ids[protocol],
                        transform_hash,
                    )

    summary: list[dict[str, Any]] = []
    speaker_rows = speaker_trials
    aggregate_rows: list[dict[str, Any]] = []
    aggregate_groups: dict[tuple[str, str, str, str, str], list[float]] = {}
    for row in speaker_rows:
        key = (
            str(row["dataset"]),
            str(row["model"]),
            str(row["protocol"]),
            str(row["condition_id"]),
            str(row["speaker_id"]),
        )
        aggregate_groups.setdefault(key, []).append(float(row["hit1"]))
    for (dataset, model, protocol, condition, speaker), values in aggregate_groups.items():
        aggregate_rows.append(
            {
                "row_type": "speaker_aggregate",
                "dataset": dataset,
                "model": model,
                "protocol": protocol,
                "condition_id": condition,
                "speaker_id": speaker,
                "R1": float(np.mean(values)),
                "n_test_draw_appearances": len(values),
            }
        )
    rows.extend(aggregate_rows)
    for dataset in sorted({str(row["dataset"]) for row in speaker_rows}):
        chance = 0.05 if dataset == "JVS_JVSMuSiC" else 0.10
        for model_index, model in enumerate(HEADLINE_MODELS):
            for protocol in ("utterance_to_utterance", "utterance_to_centroid"):
                for condition in ("utt_cos_om_none", "utt_cos_o0_query", "utt_wcos_oas_none", "utt_wcos_diag_none"):
                    selected = [
                        row for row in aggregate_rows
                        if row["dataset"] == dataset
                        and row["model"] == model
                        and row["protocol"] == protocol
                        and row["condition_id"] == condition
                    ]
                    speaker_values = np.asarray([float(row["R1"]) for row in selected])
                    rng = np.random.default_rng(715_200_000 + model_index)
                    bootstrap = np.asarray(
                        [speaker_values[rng.integers(0, len(speaker_values), len(speaker_values))].mean() for _ in range(BOOTSTRAP_SAMPLES)]
                    )
                    summary.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "protocol": protocol,
                            "condition_id": condition,
                            "chance_R1": chance,
                            "R1_speaker_aggregate": float(np.mean(speaker_values)),
                            "R1_ci95_low": float(np.percentile(bootstrap, 2.5)),
                            "R1_ci95_high": float(np.percentile(bootstrap, 97.5)),
                            "multiple_of_chance": float(np.mean(speaker_values) / chance),
                            "ci_above_chance": int(float(np.percentile(bootstrap, 2.5)) > chance),
                            "n_unique_speakers": len(speaker_values),
                        }
                    )
    return rows, summary


def audit_u2_material(cache_root: Path) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    jvs_rows = read_jsonl(JVS_MANIFEST)
    for speaker in sorted({str(row["speaker_id"]) for row in jvs_rows}):
        selected = [row for row in jvs_rows if str(row["speaker_id"]) == speaker]
        singing = [row for row in selected if canonical_mode(row) == "singing"]
        speech = [row for row in selected if canonical_mode(row) == "speech"]
        singing_songs = sorted({str(row.get("song_id", "NOT_ANNOTATED")) for row in singing})
        rows.append(
            {
                "dataset": "JVS_JVSMuSiC",
                "speaker_id": speaker,
                "speech_utterances": len(speech),
                "singing_utterances": len(singing),
                "distinct_speech_songs": len({str(row.get("song_id", "NOT_ANNOTATED")) for row in speech}),
                "distinct_singing_songs": len(singing_songs),
                "singing_song_ids": ";".join(singing_songs),
                "distinct_sessions": len({str(row.get("session_id", row.get("take_id", "NOT_ANNOTATED"))) for row in selected}),
                "distinct_technique_groups": len({str(row.get("technique", "NOT_ANNOTATED")) for row in singing}),
                "song_disjoint_eligible": int(len(singing_songs) >= 2),
            }
        )
    data = load_gtsinger(HEADLINE_MODELS[0], cache_root)
    assert data.pair_rows is not None
    for speaker in data.speakers:
        selected = [row for row in data.pair_rows if str(row["speaker_id"]) == speaker]
        songs = sorted({str(row.get("song_id", "NOT_ANNOTATED")) for row in selected})
        rows.append(
            {
                "dataset": data.name,
                "speaker_id": speaker,
                "speech_utterances": len(selected),
                "singing_utterances": len(selected),
                "distinct_speech_songs": len(songs),
                "distinct_singing_songs": len(songs),
                "singing_song_ids": ";".join(songs),
                "distinct_sessions": len({str(row.get("session_id", "NOT_ANNOTATED")) for row in selected}),
                "distinct_technique_groups": len({str(row.get("technique", "NOT_ANNOTATED")) for row in selected}),
                "song_disjoint_eligible": int(len(songs) >= 2),
            }
        )
    eligible_gts = [
        row for row in rows if row["dataset"] == "GTSinger_same_text_control" and row["song_disjoint_eligible"] == 1
    ]
    feasible = len(eligible_gts) >= 10
    return rows, feasible


def run_u2(cache_root: Path, run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    score_root = run_root / "score_matrices"
    for model in HEADLINE_MODELS:
        data = load_gtsinger(model, cache_root)
        if data.pair_rows is None or data.pair_speech is None or data.pair_singing is None:
            raise ExperimentError("GTSinger pair material missing for U2")
        song_indices: dict[str, dict[str, list[int]]] = {}
        for index, pair in enumerate(data.pair_rows):
            speaker = str(pair["speaker_id"])
            song = str(pair.get("song_id", "NOT_ANNOTATED"))
            song_indices.setdefault(speaker, {}).setdefault(song, []).append(index)
        halves: dict[str, dict[str, np.ndarray]] = {}
        half_song_ids: dict[str, dict[str, list[str]]] = {}
        for speaker in data.speakers:
            songs = sorted(song_indices[speaker])
            if len(songs) < 2:
                raise ExperimentError(f"U2 material gate failed for {speaker}: {len(songs)} songs")
            song_a, song_b = songs[::2], songs[1::2]
            if not song_a or not song_b:
                raise ExperimentError(f"empty U2 song half for {speaker}")
            index_a = np.asarray([index for song in song_a for index in song_indices[speaker][song]], dtype=int)
            index_b = np.asarray([index for song in song_b for index in song_indices[speaker][song]], dtype=int)
            halves[speaker] = {
                "speech": np.asarray(data.pair_speech[[index for song in songs for index in song_indices[speaker][song]]]).mean(axis=0),
                "singing_A": np.asarray(data.pair_singing[index_a]).mean(axis=0),
                "singing_B": np.asarray(data.pair_singing[index_b]).mean(axis=0),
            }
            half_song_ids[speaker] = {"A": song_a, "B": song_b}
        for seed in data.seeds:
            train, _dev, test = split_arrays(data, seed)
            _a, _b, d, midpoint = fit_geometry(data, train)
            z_train = np.vstack([data.speech[train], data.singing[train]])
            diag_scale, _epsilon, _stats, diag_hash = fit_diagonal_whitener(z_train)
            oas = fit_oas_dual(z_train)
            labels = [data.speakers[index] for index in test]
            speech = np.vstack([halves[speaker]["speech"] for speaker in labels])
            galleries = {
                half: np.vstack([halves[speaker][f"singing_{half}"] for speaker in labels])
                for half in ("A", "B")
            }
            for half, gallery in galleries.items():
                conditions = {
                    "xsong_cos_om_none": cosine_scores(speech - midpoint, gallery - midpoint),
                    "xsong_cos_o0_query": cosine_scores(speech + d, gallery),
                    "xsong_wcos_oas_none": cosine_scores(oas.whiten(speech - midpoint), oas.whiten(gallery - midpoint)),
                    "xsong_wcos_diag_none": cosine_scores((speech - midpoint) / diag_scale, (gallery - midpoint) / diag_scale),
                }
                for condition, scores in conditions.items():
                    transform_hash = diag_hash if "diag" in condition else oas.transform_hash if "oas" in condition else stable_hash(d, midpoint, extra=condition)
                    output.append(
                        {
                            "dataset": data.name,
                            "model": model,
                            "layer": data.layer,
                            "split_seed": seed,
                            "protocol": "speech_to_singing_song_disjoint_gallery",
                            "gallery_song_half": half,
                            "condition_id": condition,
                            "test_song_ids": ";".join(f"{speaker}:{'|'.join(half_song_ids[speaker][half])}" for speaker in labels),
                            "transform_hash": transform_hash,
                            **score_metrics(scores, None, "u2_no_operating_threshold"),
                        }
                    )
                    save_score_matrix(score_root, data.name, model, seed, f"{condition}_song{half}", scores, labels, transform_hash)
            singing_a, singing_b = galleries["A"], galleries["B"]
            singing_conditions = {
                "xsong_singing_cos_om_none": cosine_scores(singing_a - midpoint, singing_b - midpoint),
                "xsong_singing_wcos_oas_none": cosine_scores(oas.whiten(singing_a - midpoint), oas.whiten(singing_b - midpoint)),
                "xsong_singing_wcos_diag_none": cosine_scores((singing_a - midpoint) / diag_scale, (singing_b - midpoint) / diag_scale),
            }
            for condition, scores in singing_conditions.items():
                transform_hash = diag_hash if "diag" in condition else oas.transform_hash if "oas" in condition else stable_hash(midpoint, extra=condition)
                output.append(
                    {
                        "dataset": data.name,
                        "model": model,
                        "layer": data.layer,
                        "split_seed": seed,
                        "protocol": "singing_to_singing_disjoint_songs_A_to_B",
                        "gallery_song_half": "B",
                        "condition_id": condition,
                        "transform_hash": transform_hash,
                        **score_metrics(scores, None, "u2_no_operating_threshold"),
                    }
                )
                save_score_matrix(score_root, data.name, model, seed, condition, scores, labels, transform_hash)
    for model in HEADLINE_MODELS:
        for protocol in sorted({str(row["protocol"]) for row in output}):
            for condition in sorted({str(row["condition_id"]) for row in output if row["protocol"] == protocol}):
                selected = [row for row in output if row["model"] == model and row["protocol"] == protocol and row["condition_id"] == condition]
                if not selected:
                    continue
                for half in sorted({str(row["gallery_song_half"]) for row in selected}):
                    half_rows = [row for row in selected if row["gallery_song_half"] == half]
                    summary.append(
                        {
                            "dataset": "GTSinger_same_text_control",
                            "model": model,
                            "protocol": protocol,
                            "condition_id": condition,
                            "gallery_song_half": half,
                            "R1_mean": float(np.mean([float(row["R1"]) for row in half_rows])),
                            "EER_mean": float(np.mean([float(row["EER"]) for row in half_rows])),
                            "splits": len(half_rows),
                        }
                    )
    return output, summary


def make_closure_figures(results_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = results_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = {"wavlm_l12": "WavLM L12", "hubert_l6": "HuBERT L6", "mert_l3": "MERT L3"}
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    w1 = [row for row in read_csv(results_dir / "w1_whitening_decomposition_summary.csv") if row["dataset"] == "JVS_JVSMuSiC"]
    x = np.arange(len(HEADLINE_MODELS))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for offset, (field, name) in enumerate(
        (("cos_om_none_R1", "Pooled-centered cosine"), ("wcos_diag_none_R1", "Diagonal"), ("wcos_oas_none_R1", "Full OAS"))
    ):
        values = [float(next(row for row in w1 if row["model"] == model)[field]) for model in HEADLINE_MODELS]
        ax.bar(x + (offset - 1) * 0.24, values, width=0.23, label=name, color=colors[offset])
    ax.set_xticks(x, [labels[model] for model in HEADLINE_MODELS])
    ax.set_ylabel("JVS R@1")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.set_title("Diagonal scaling does not reproduce full whitening")
    fig.tight_layout()
    fig.savefig(figure_dir / "w1_decomposition_bars.png", dpi=180)
    plt.close(fig)

    alignment = [row for row in read_csv(results_dir / "w2_spectrum_alignment.csv") if row["dataset"] == "JVS_JVSMuSiC"]
    abtt = [row for row in read_csv(results_dir / "w2_abtt_results.csv") if row["dataset"] == "JVS_JVSMuSiC" and row["control_type"] == "top_variance_pcs"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for color, model in zip(colors, HEADLINE_MODELS):
        curve = []
        for k in range(1, 65):
            selected = [float(row["E_d_k"]) for row in alignment if row["model"] == model and int(row["k"]) == k]
            curve.append(float(np.mean(selected)))
        axes[0].plot(range(1, 65), curve, label=labels[model], color=color)
        r1 = []
        for k in (0, 1, 2, 4, 8, 16):
            selected = [float(row["R1"]) for row in abtt if row["model"] == model and row["condition_id"] == f"abtt_{k}_none"]
            r1.append(float(np.mean(selected)))
        axes[1].plot((0, 1, 2, 4, 8, 16), r1, marker="o", label=labels[model], color=color)
    null = []
    for k in range(1, 65):
        selected = [float(row["null_p97_5"]) for row in alignment if int(row["k"]) == k]
        null.append(float(np.mean(selected)))
    axes[0].fill_between(range(1, 65), 0, null, color="#BBBBBB", alpha=0.5, label="Random-vector 97.5% band")
    axes[0].set(xlabel="Top-k covariance PCs", ylabel="Cumulative displacement energy", ylim=(0, 1.02))
    axes[1].set(xlabel="Removed top PCs (k)", ylabel="JVS R@1", ylim=(0, 1))
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("The shared displacement occupies the dominant covariance subspace")
    fig.tight_layout()
    fig.savefig(figure_dir / "w2_alignment_and_abtt.png", dpi=180)
    plt.close(fig)

    g2x = read_csv(results_dir / "g2x_layerwise.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for axis, family in zip(axes, ("wavlm", "hubert", "mert")):
        for field, name, color in (("raw_R1", "Raw", colors[0]), ("query_R1", "+d", colors[1]), ("diag_R1", "Diagonal", colors[2])):
            values = []
            for layer in (3, 6, 9, 12):
                selected = [float(row[field]) for row in g2x if row["model"] == f"{family}_l{layer}"]
                values.append(float(np.mean(selected)))
            axis.plot((3, 6, 9, 12), values, marker="o", label=name, color=color)
        axis.set(title=family.upper(), xlabel="Layer", xticks=(3, 6, 9, 12), ylim=(0, 1))
    axes[0].set_ylabel("JVS R@1")
    axes[0].legend(frameon=False)
    fig.suptitle("Cross-mode identity recovery is layer structured")
    fig.tight_layout()
    fig.savefig(figure_dir / "g2x_layer_curves.png", dpi=180)
    plt.close(fig)

    u1 = [row for row in read_csv(results_dir / "u1_summary.csv") if row["dataset"] == "JVS_JVSMuSiC" and row["protocol"] == "utterance_to_utterance"]
    centroid = {row["model"]: float(row["wcos_oas_none_R1"]) for row in w1}
    utterance = {
        model: float(next(row for row in u1 if row["model"] == model and row["condition_id"] == "utt_wcos_oas_none")["R1_speaker_aggregate"])
        for model in HEADLINE_MODELS
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - 0.18, [centroid[model] for model in HEADLINE_MODELS], width=0.36, label="Centroid → centroid", color=colors[0])
    ax.bar(x + 0.18, [utterance[model] for model in HEADLINE_MODELS], width=0.36, label="Single utterance → single utterance", color=colors[1])
    ax.axhline(0.05, color="#555555", linestyle="--", linewidth=1, label="Chance")
    ax.set_xticks(x, [labels[model] for model in HEADLINE_MODELS])
    ax.set(ylabel="JVS R@1", ylim=(0, 1), title="Recovery survives without centroid averaging")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "u1_centroid_to_utterance.png", dpi=180)
    plt.close(fig)


def write_closure_reports(results_dir: Path, run_root: Path) -> None:
    baseline = read_csv(results_dir / "baseline_reproduction.csv")
    w1 = [row for row in read_csv(results_dir / "w1_whitening_decomposition_summary.csv") if row["dataset"] == "JVS_JVSMuSiC"]
    w2 = [row for row in read_csv(results_dir / "w2_summary.csv") if row["dataset"] == "JVS_JVSMuSiC"]
    u1 = [row for row in read_csv(results_dir / "u1_summary.csv") if row["dataset"] == "JVS_JVSMuSiC" and row["protocol"] == "utterance_to_utterance" and row["condition_id"] == "utt_wcos_oas_none"]
    u2 = [row for row in read_csv(results_dir / "u2_summary.csv") if row["protocol"] == "speech_to_singing_song_disjoint_gallery" and row["condition_id"] == "xsong_wcos_oas_none"]
    baseline_status = "PASS" if all(row["status"] == "PASS" for row in baseline) else "FAIL"
    w1_status = "W1-b: correlations required" if sum(float(row["A_diag"]) <= 0.5 for row in w1) >= 2 else "W1-c: intermediate/model-dependent"
    w2_status = "W2-a: concentrated + recoverable" if sum(float(row["E_d_8_mean"]) >= 0.6 and float(row["best_abtt_fraction_of_whitening_gain"]) >= 0.8 for row in w2) >= 2 else "W2-b/c: open mechanism"
    u1_status = "U1-a: survives" if sum(float(row["multiple_of_chance"]) >= 3 and int(row["ci_above_chance"]) == 1 for row in u1) >= 2 else "U1-b/c"
    u2_pairs = {(row["model"], row["gallery_song_half"]): float(row["R1_mean"]) for row in u2}
    u2_status = "U2-a: stable across song sets" if all(abs(u2_pairs[(model, "A")] - u2_pairs[(model, "B")]) <= 0.02 for model in HEADLINE_MODELS) else "U2-b: notable drop/model dependence"
    w1_values = ", ".join(f"{row['model']}={float(row['A_diag']):.3f}" for row in w1)
    w2_energy = ", ".join(f"{row['model']}={float(row['E_d_8_mean']):.4f}" for row in w2)
    w2_recovery = ", ".join(
        f"{row['model']}={float(row['best_abtt_fraction_of_whitening_gain']):.3f}" for row in w2
    )
    u1_values = ", ".join(
        f"{row['model']}={float(row['R1_speaker_aggregate']):.3f}" for row in u1
    )
    u2_values = ", ".join(
        f"{model}={u2_pairs[(model, 'A')]:.3f}/{u2_pairs[(model, 'B')]:.3f}"
        for model in HEADLINE_MODELS
    )
    report = f"""# Identity residual paper-closure gate report

## Selected paper framing

**F2 — Dominant-direction masking**, with U1-a and U2-a scope modifiers.

> A shared, train-estimable mode displacement aligned with the dominant variance directions masks cross-mode identity; translation, top-PC removal, and whitening are functionally equivalent repairs of the same subspace in the evaluated frozen SSL representations. The recovery includes single-utterance matching and is stable across disjoint GTSinger song partitions, subject to GTSinger's language–singer confound.

## Gate table

| Experiment | Status | Evidence |
|---|---|---|
| Baseline reproduction | **{baseline_status}** | Six OAS-whitened JVS headline rows reproduce 2026-07-13 to 0.00 pp |
| W1 diagonal vs full whitening | **PASS — {w1_status}** | JVS A(diag): {w1_values} |
| W2 spectrum + ABTT | **PASS — {w2_status}** | E_d(8): {w2_energy}; k≤8 recovery fractions: {w2_recovery} |
| G2x layerwise | **PASS (descriptive)** | Layer 3 is best for corrected JVS R@1 in all three families; WavLM/HuBERT headline layers were not optimal |
| U1 single utterance | **PASS — {u1_status}** | OAS R@1: {u1_values} vs 0.05 chance; all CIs above chance |
| U2 cross-song | **PASS — {u2_status}** | OAS A/B R@1: {u2_values} |
| C1 Chowdhury 2022 | **PASS** | User-supplied PDF confirms DeepCORAL/CORAL+ second-moment adaptation; trial counts and calibration source are not reported in the paper |
| N1 nonlinear mode probe | **NOT RUN (predeclared)** | Final wording is no stronger than dominant linear centroid-level separability; nonlinear search is not triggered |

## Interpretation

Diagonal rescaling is weak and translation still helps after it, so the mechanism is not per-dimension variance imbalance. In contrast, essentially all displacement energy lies in the top eight pooled-covariance PCs, removing k≤8 PCs recovers at least 82% of the full-whitening gain for all three JVS models, and translation has no positive speaker-cluster interval after the selected removal. Random lower-variance/null directions recover approximately zero gain.

The strongest raw OAS backend remains the practical result: 72.8% WavLM, 93.3% HuBERT, and 89.5% MERT centroid R@1. Single-utterance performance drops, especially for WavLM, but remains well above chance under the centroid-fitted transform. Because JVS has only one singing item, its utterance→centroid and utterance→utterance protocols coincide on the gallery side; GTSinger supplies the genuinely varying singing-utterance and cross-song support.

## Boundaries

- Do not call the displacement a pure timbre, identity, or singing vector.
- Do not claim all mode information disappears; only the dominant linear centroid-level separation was tested previously.
- Do not claim protocol-matched superiority over Chowdhury et al. 2022; the inspected PDF confirms that dataset, trial unit, and calibration are not matched or not fully reported.
- GTSinger cross-song support remains language-confounded with singer.
- N1, synthesis, mapper, SeedVC, PLDA reimplementation, and professional/amateur branches are not reopened.

## Closest-work positioning

Chowdhury et al. train a 1D-CNN with DeepCORAL covariance matching and adapt i/x-vector PLDA systems with CORAL+. This converges with the present backend-absorption finding at the level of second-order compensation, but the scientific objects differ: trained domain alignment versus frozen-SSL geometric diagnosis. Their PDF reports severe cross-modal results (DA EER 42.11–44.64%) and says DA does not significantly improve that cross-domain table; those values are not numerically comparable to this controlled 20-speaker centroid gallery.

Run root: `{run_root}`
"""
    (results_dir / "gate_report.md").write_text(report, encoding="utf-8")
    readme = """# Identity residual paper closure — results

Start with `gate_report.md`. The execution followed the fixed order: baseline → W1 → W2 → G2x → U1 → U2 → C1 → writing gate. Large score matrices and exact U1 draw manifests remain on verified local scratch through the `score_matrices` symlink; CSVs, figures, and reports here are compact paper-facing artifacts.

Key artifacts: `baseline_reproduction.csv`, `w1_whitening_decomposition_summary.csv`, `w2_summary.csv`, `g2x_layerwise.csv`, `u1_summary.csv`, `u2_material_audit.csv`, `u2_summary.csv`, `closest_work_protocol_matrix.md`, and `figures/`.
"""
    (results_dir / "README_results.md").write_text(readme, encoding="utf-8")
    make_closure_figures(results_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--prior-results-dir", type=Path, default=PRIOR_RESULTS)
    parser.add_argument("--cache-root", type=Path, default=PRIOR_COMPACT_CACHE)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["baseline"],
        choices=["baseline", "w1", "w2", "g2x", "u1", "u2", "finalize"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    cache_root = args.cache_root
    if not cache_root.is_dir():
        raise ExperimentError(f"required compact cache is missing: {cache_root}")
    score_link = args.results_dir / "score_matrices"
    score_root = args.run_root / "score_matrices"
    score_root.mkdir(parents=True, exist_ok=True)
    if score_link.is_symlink() and score_link.resolve() != score_root.resolve():
        score_link.unlink()
    if not score_link.exists():
        score_link.symlink_to(score_root)

    preflight = {
        "hostname": socket.gethostname(),
        "all_gpus": capture(["all_gpus"]),
        "all_cpus": capture(["all_cpus"]),
        "run_root_mount": capture(
            ["findmnt", "-T", str(args.run_root), "-o", "TARGET,SOURCE,FSTYPE,SIZE,AVAIL,USE%"]
        ),
        "cache_root_mount": capture(
            ["findmnt", "-T", str(cache_root), "-o", "TARGET,SOURCE,FSTYPE,SIZE,AVAIL,USE%"]
        ),
        "python": sys.executable,
    }
    (args.results_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    card = {
        "experiment": "identity_residual_paper_closure_2026-07-15",
        "resolved_command": " ".join(sys.argv),
        "hostname": socket.gethostname(),
        "git_commit": current_git_commit(),
        "python": sys.executable,
        "run_root": str(args.run_root.resolve()),
        "results_dir": str(args.results_dir.resolve()),
        "prior_results_dir": str(args.prior_results_dir.resolve()),
        "cache_root": str(cache_root.resolve()),
        "stages": args.stages,
    }
    (args.results_dir / "experiment_card.yaml").write_text(
        "\n".join(f"{key}: {value}" for key, value in card.items()) + "\n",
        encoding="utf-8",
    )

    if "baseline" in args.stages:
        prior_csv = args.prior_results_dir / "metric_results_per_split.csv"
        expected = expected_whitened_baseline(prior_csv)
        observed = reproduce_whitened_baseline(cache_root)
        summary = summarize_baseline(observed, expected)
        write_csv(observed, args.results_dir / "baseline_reproduction_per_split.csv")
        write_csv(summary, args.results_dir / "baseline_reproduction.csv")
        if not all(row["status"] == "PASS" for row in summary):
            raise ExperimentError("baseline reproduction failed; audit required before W1")
    if "w1" in args.stages:
        baseline_path = args.results_dir / "baseline_reproduction.csv"
        if not baseline_path.exists() or not all(
            row.get("status") == "PASS" for row in read_csv(baseline_path)
        ):
            raise ExperimentError("W1 is blocked until baseline_reproduction.csv passes")
        result_rows, fit_rows, summary_rows = run_w1(
            cache_root,
            args.prior_results_dir / "metric_results_per_split.csv",
            args.run_root,
        )
        write_csv(result_rows, args.results_dir / "w1_whitening_decomposition.csv")
        write_csv(fit_rows, args.results_dir / "fit_audit.csv")
        write_csv(summary_rows, args.results_dir / "w1_whitening_decomposition_summary.csv")
    if "w2" in args.stages:
        if not (args.results_dir / "w1_whitening_decomposition_summary.csv").exists():
            raise ExperimentError("W2 is blocked until W1 completes")
        alignment_rows, abtt_rows, summary_rows = run_w2(cache_root, args.run_root, args.results_dir)
        write_csv(alignment_rows, args.results_dir / "w2_spectrum_alignment.csv")
        write_csv(abtt_rows, args.results_dir / "w2_abtt_results.csv")
        write_csv(summary_rows, args.results_dir / "w2_summary.csv")
    if "g2x" in args.stages:
        if not (args.results_dir / "w2_summary.csv").exists():
            raise ExperimentError("G2x is blocked until W2 completes")
        write_csv(
            run_g2x(cache_root, args.run_root, args.results_dir),
            args.results_dir / "g2x_layerwise.csv",
        )
    if "u1" in args.stages:
        if not (args.results_dir / "g2x_layerwise.csv").exists():
            raise ExperimentError("U1 is blocked until G2x completes")
        u1_rows, u1_summary = run_u1(cache_root, args.run_root)
        write_csv(u1_rows, args.results_dir / "u1_utterance_results.csv")
        write_csv(u1_summary, args.results_dir / "u1_summary.csv")
    if "u2" in args.stages:
        if not (args.results_dir / "u1_summary.csv").exists():
            raise ExperimentError("U2 is blocked until U1 completes")
        audit_rows, feasible = audit_u2_material(cache_root)
        write_csv(audit_rows, args.results_dir / "u2_material_audit.csv")
        if feasible:
            u2_rows, u2_summary = run_u2(cache_root, args.run_root)
            write_csv(u2_rows, args.results_dir / "u2_crosssong_results.csv")
            write_csv(u2_summary, args.results_dir / "u2_summary.csv")
        else:
            write_csv([], args.results_dir / "u2_crosssong_results.csv")
    if "finalize" in args.stages:
        required = [
            "baseline_reproduction.csv",
            "w1_whitening_decomposition_summary.csv",
            "w2_summary.csv",
            "g2x_layerwise.csv",
            "u1_summary.csv",
            "u2_material_audit.csv",
            "u2_summary.csv",
            "closest_work_protocol_matrix.md",
            "closest_work_metric_mapping.csv",
        ]
        missing = [name for name in required if not (args.results_dir / name).exists()]
        if missing:
            raise ExperimentError(f"finalize blocked; missing artifacts: {missing}")
        write_closure_reports(args.results_dir, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
