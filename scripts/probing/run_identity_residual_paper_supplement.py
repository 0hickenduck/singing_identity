#!/usr/bin/env python
"""Additive paper supplement: OAS at JVS layer 3 and W2-x closure.

Execution is deliberately ordered R0' -> X1 -> X2 -> X3 -> reports.  The
completed 2026-07-15 closure artifacts are read-only inputs.
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
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probing.run_identity_residual_metric_robustness import (  # noqa: E402
    cosine_scores,
    fit_geometry,
    fit_oas_dual,
    load_jvs,
    ranks_from_scores,
    score_metrics,
    split_arrays,
)
from probing.run_identity_residual_paper_closure import (  # noqa: E402
    BOOTSTRAP_SAMPLES,
    capture,
    read_csv,
    reproduce_whitened_baseline,
    save_score_matrix,
    stable_hash,
    summarize_baseline,
    write_csv,
)
from research_utils import ExperimentError, current_git_commit  # noqa: E402


DEFAULT_RUN_ROOT = Path(
    "/localdisk/bowen/singing_identity/runs/identity_residual_paper_supplement_2026-07-16"
)
DEFAULT_RESULTS_DIR = Path("results/identity_residual_paper_supplement_2026-07-16")
DEFAULT_CACHE_ROOT = Path(
    "/localdisk/bowen/singing_identity/runs/identity_residual_metric_robustness_2026-07-13/cache"
)
CLOSURE_RESULTS = Path("results/identity_residual_paper_closure_2026-07-15")
CLOSURE_RUN_ROOT = Path(
    "/localdisk/bowen/singing_identity/runs/identity_residual_paper_closure_2026-07-15"
)
L3_MODELS = ("wavlm_l3", "hubert_l3")
HEADLINE_FOR_FAMILY = {"wavlm_l3": "wavlm_l12", "hubert_l3": "hubert_l6"}
SELECTED_K = {"wavlm_l12": 1, "hubert_l6": 8, "mert_l3": 1}
ALL_K = (1, 2, 4, 8, 16)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:20]


def holm_adjust(p_values: dict[str, float]) -> dict[str, tuple[float, bool]]:
    """Return Holm-adjusted p-values and alpha=.05 decisions."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, tuple[float, bool]] = {}
    running = 0.0
    rejected_prefix = True
    for rank, (key, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p_value))
        threshold = 0.05 / (m - rank)
        rejected = rejected_prefix and p_value <= threshold
        rejected_prefix = rejected
        adjusted[key] = (running, rejected)
    return adjusted


def speaker_interval(
    speaker_values: dict[str, list[float]], seed: int, samples: int = BOOTSTRAP_SAMPLES
) -> tuple[float, float, float, float, int]:
    values = np.asarray(
        [np.mean(speaker_values[speaker]) for speaker in sorted(speaker_values)],
        dtype=np.float64,
    )
    if not len(values):
        raise ExperimentError("speaker interval received no observations")
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


def normal_interval(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(array) < 2:
        return mean, math.nan, math.nan
    half = 1.96 * float(array.std(ddof=1)) / math.sqrt(len(array))
    return mean, mean - half, mean + half


def load_bundle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ExperimentError(f"missing required saved score bundle: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def bundle_r1(bundle: dict[str, Any]) -> float:
    scores = np.asarray(bundle["score_matrix"], dtype=np.float64)
    ranks, _ = ranks_from_scores(scores)
    return float(np.mean(ranks == 1))


def bundle_hit_rows(
    bundle: dict[str, Any], condition: str, split_seed: int, model: str = ""
) -> list[dict[str, Any]]:
    query_ids = [str(value) for value in bundle["ordered_query_speaker_ids"].tolist()]
    gallery_ids = [str(value) for value in bundle["ordered_gallery_speaker_ids"].tolist()]
    if query_ids != gallery_ids:
        raise ExperimentError(f"query/gallery order mismatch for {condition}/{split_seed}")
    ranks, _ = ranks_from_scores(np.asarray(bundle["score_matrix"], dtype=np.float64))
    return [
        {
            "speaker_id": speaker,
            "model": model,
            "condition_id": condition,
            "split_seed": split_seed,
            "hit1": int(ranks[index] == 1),
        }
        for index, speaker in enumerate(query_ids)
    ]


def closure_baseline_expected(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    mapping = {
        "oas_whitened_cosine_raw": "oas_whitened_cosine_raw",
        "oas_whitened_cosine_query": "oas_whitened_cosine_query",
    }
    expected: dict[tuple[str, str], dict[str, float]] = {}
    for row in read_csv(path):
        condition = mapping[row["condition_id"]]
        expected[(row["model"], condition)] = {
            "R1": float(row["observed_R1"]),
            "EER": float(row["observed_EER"]),
        }
    if len(expected) != 6:
        raise ExperimentError(f"expected six closure baseline rows, found {len(expected)}")
    return expected


def assert_w2_summary_reproduction(closure_results: Path) -> list[dict[str, Any]]:
    abtt = read_csv(closure_results / "w2_abtt_results.csv")
    summary = read_csv(closure_results / "w2_summary.csv")
    checks: list[dict[str, Any]] = []
    for row in summary:
        if row["dataset"] != "JVS_JVSMuSiC":
            continue
        model = row["model"]
        k = int(row["best_abtt_k"])
        none = [
            float(item["R1"])
            for item in abtt
            if item["dataset"] == "JVS_JVSMuSiC"
            and item["model"] == model
            and item["condition_id"] == f"abtt_{k}_none"
            and item["control_type"] == "top_variance_pcs"
        ]
        query = [
            float(item["R1"])
            for item in abtt
            if item["dataset"] == "JVS_JVSMuSiC"
            and item["model"] == model
            and item["condition_id"] == f"abtt_{k}_query"
            and item["control_type"] == "top_variance_pcs"
        ]
        if len(none) != 20 or len(query) != 20:
            raise ExperimentError(f"W2 summary source rows incomplete: {model}/k={k}")
        recomputed = float(np.mean(query) - np.mean(none))
        stored = float(row["translation_increment_at_best_k_R1"])
        if not math.isclose(recomputed, stored, rel_tol=0.0, abs_tol=1e-12):
            raise ExperimentError(
                f"W2 summary mismatch {model}/k={k}: {recomputed} != {stored}"
            )
        checks.append(
            {
                "model": model,
                "best_abtt_k": k,
                "stored_translation_increment_R1": stored,
                "recomputed_translation_increment_R1": recomputed,
                "status": "PASS",
            }
        )
    if len(checks) != 3:
        raise ExperimentError(f"expected three JVS W2 summary checks, found {len(checks)}")
    return checks


def run_r0(cache_root: Path, closure_results: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observed = reproduce_whitened_baseline(cache_root)
    expected = closure_baseline_expected(closure_results / "baseline_reproduction.csv")
    summary = summarize_baseline(observed, expected)
    if not all(row["status"] == "PASS" for row in summary):
        raise ExperimentError("R0' failed: closure baseline audit is required before X1")
    checks = assert_w2_summary_reproduction(closure_results)
    return summary, checks


def run_x1(
    cache_root: Path,
    run_root: Path,
    closure_results: Path,
    closure_run_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    g2x = read_csv(closure_results / "g2x_layerwise.csv")
    closure_split = read_csv(closure_results / "baseline_reproduction_per_split.csv")
    closure_fit = read_csv(closure_results / "fit_audit.csv")
    results: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    hit_rows: list[dict[str, Any]] = []
    score_root = run_root / "score_matrices"

    for model in L3_MODELS:
        data = load_jvs(model, cache_root)
        headline = HEADLINE_FOR_FAMILY[model]
        for seed in data.seeds:
            train, _dev, test = split_arrays(data, seed)
            _a, _b, d, midpoint = fit_geometry(data, train)
            z_train = np.vstack([data.speech[train], data.singing[train]])
            oas = fit_oas_dual(z_train)
            headline_fit = [
                row
                for row in closure_fit
                if row["dataset"] == data.name
                and row["model"] == headline
                and int(row["split_seed"]) == seed
                and row["fit_type"] == "OAS"
            ]
            if len(headline_fit) != 1:
                raise ExperimentError(f"missing headline OAS fit audit: {headline}/{seed}")
            if oas.transform_hash == headline_fit[0]["transform_hash"]:
                raise ExperimentError(f"L3 and headline transform hashes unexpectedly match: {model}/{seed}")
            labels = [data.speakers[index] for index in test]
            delta = data.singing[test] - data.speech[test]
            wd = oas.whiten(d[None, :])[0]
            wdelta = oas.whiten(delta)
            wresidual = oas.whiten(delta - d)
            fit_speakers = ";".join(data.speakers[index] for index in train)
            fits.append(
                {
                    "dataset": data.name,
                    "model": model,
                    "layer": data.layer,
                    "split_seed": seed,
                    "fit_type": "OAS",
                    "fit_speakers": fit_speakers,
                    "n_fit_vectors": len(z_train),
                    "dimension": z_train.shape[1],
                    "shrinkage": oas.shrinkage,
                    "covariance_trace": float(oas.empirical_eigenvalues.sum()),
                    "minimum_effective_eigenvalue": oas.lambda0,
                    "condition_number": float(
                        max(oas.fitted_eigenvalues.max(initial=oas.lambda0), oas.lambda0)
                        / oas.lambda0
                    ),
                    "whitened_residual_alignment": float(
                        np.mean(
                            (wdelta @ wd)
                            / np.maximum(
                                np.linalg.norm(wdelta, axis=1) * np.linalg.norm(wd), 1e-12
                            )
                        )
                    ),
                    "whitened_E_test": 1.0
                    - float(np.sum(wresidual * wresidual))
                    / max(float(np.sum(wdelta * wdelta)), 1e-12),
                    "transform_hash": oas.transform_hash,
                    "headline_model": headline,
                    "headline_transform_hash": headline_fit[0]["transform_hash"],
                }
            )
            context = [
                row
                for row in g2x
                if row["model"] == model and int(row["split_seed"]) == seed
            ]
            if len(context) != 1:
                raise ExperimentError(f"missing unique G2x context row: {model}/{seed}")
            context_row = context[0]
            gallery = oas.whiten(data.singing[test] - midpoint)
            conditions = {
                "oasl3_wcos_none": oas.whiten(data.speech[test] - midpoint),
                "oasl3_wcos_query": oas.whiten(data.speech[test] + d - midpoint),
            }
            for condition, query in conditions.items():
                scores = cosine_scores(query, gallery)
                metrics = score_metrics(scores, None, "x1_no_operating_threshold")
                results.append(
                    {
                        "dataset": data.name,
                        "model": model,
                        "layer": data.layer,
                        "split_seed": seed,
                        "condition_id": condition,
                        "score_family": "oas_whitened_cosine",
                        "origin": "pooled_train_midpoint",
                        "alignment": "query" if condition.endswith("query") else "none",
                        "covariance_method": "OAS",
                        "shrinkage": oas.shrinkage,
                        "lda_dim": "",
                        "train_speakers": fit_speakers,
                        "dev_speakers": "",
                        "test_speakers": ";".join(labels),
                        "transform_hash": oas.transform_hash,
                        "raw_l3_R1_copied": float(context_row["raw_R1"]),
                        "query_l3_R1_copied": float(context_row["query_R1"]),
                        "raw_l3_EER_copied": float(context_row["raw_EER"]),
                        "query_l3_EER_copied": float(context_row["query_EER"]),
                        "raw_l3_source": "closure/g2x_layerwise.csv",
                        **metrics,
                    }
                )
                save_score_matrix(
                    score_root, data.name, model, seed, condition, scores, labels, oas.transform_hash
                )
                hit_rows.extend(bundle_hit_rows(load_bundle(
                    score_root / f"{data.name}__{model}__{seed}__{condition}.npz"
                ), condition, seed, model))

            headline_row = [
                row
                for row in closure_split
                if row["model"] == headline
                and int(row["split_seed"]) == seed
                and row["condition_id"] == "oas_whitened_cosine_raw"
            ]
            if len(headline_row) != 1:
                raise ExperimentError(f"missing headline baseline split row: {headline}/{seed}")
            headline_bundle_path = (
                closure_run_root / "score_matrices"
                / f"{data.name}__{headline}__{seed}__wcos_oas_none.npz"
            )
            headline_bundle = load_bundle(headline_bundle_path)
            headline_hash = str(headline_bundle["transform_hash"].item())
            if headline_hash != headline_fit[0]["transform_hash"]:
                raise ExperimentError(f"headline transform hash mismatch: {headline}/{seed}")
            headline_ids = [str(value) for value in headline_bundle["ordered_gallery_speaker_ids"].tolist()]
            if headline_ids != labels:
                raise ExperimentError(f"headline/L3 gallery order mismatch: {model}/{seed}")
            if not math.isclose(
                bundle_r1(headline_bundle), float(headline_row[0]["R1"]), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ExperimentError(f"headline bundle R1 mismatch: {headline}/{seed}")
            hit_rows.extend(bundle_hit_rows(headline_bundle, "headline_wcos_none", seed, model))

    # MERT is already L3: copy, never refit.
    for row in closure_split:
        if row["model"] != "mert_l3":
            continue
        copied = dict(row)
        copied["condition_id"] = (
            "oasl3_reference_wcos_none"
            if row["condition_id"] == "oas_whitened_cosine_raw"
            else "oasl3_reference_wcos_query"
        )
        copied["source"] = "copied_exactly_from_closure_baseline_reproduction_per_split.csv"
        results.append(copied)

    summaries: list[dict[str, Any]] = []
    raw_p: dict[str, float] = {}
    for model in L3_MODELS:
        grouped: dict[str, list[float]] = {}
        by_key = {
            (row["speaker_id"], row["condition_id"], int(row["split_seed"])): row
            for row in hit_rows
            if row["model"] == model
            and row["condition_id"] in {"oasl3_wcos_none", "headline_wcos_none"}
        }
        for (speaker, condition, seed), row in by_key.items():
            if condition != "oasl3_wcos_none":
                continue
            partner = by_key.get((speaker, "headline_wcos_none", seed))
            if partner is None:
                raise ExperimentError(f"missing X1 paired headline hit: {model}/{speaker}/{seed}")
            grouped.setdefault(str(speaker), []).append(float(row["hit1"]) - float(partner["hit1"]))
        mean, low, high, p_value, n_speakers = speaker_interval(
            grouped, seed=716_100 + L3_MODELS.index(model)
        )
        translation_grouped: dict[str, list[float]] = {}
        translation_rows = {
            (row["speaker_id"], row["condition_id"], int(row["split_seed"])): row
            for row in hit_rows
            if row["model"] == model
            and row["condition_id"] in {"oasl3_wcos_none", "oasl3_wcos_query"}
        }
        for (speaker, condition, seed), row in translation_rows.items():
            if condition != "oasl3_wcos_none":
                continue
            partner = translation_rows.get((speaker, "oasl3_wcos_query", seed))
            if partner is None:
                raise ExperimentError(f"missing X1 translation pair: {model}/{speaker}/{seed}")
            translation_grouped.setdefault(str(speaker), []).append(
                float(partner["hit1"]) - float(row["hit1"])
            )
        t_mean, t_low, t_high, t_p, _ = speaker_interval(
            translation_grouped, seed=716_150 + L3_MODELS.index(model)
        )
        l3_rows = [row for row in results if row.get("model") == model and row["condition_id"] == "oasl3_wcos_none"]
        headline = HEADLINE_FOR_FAMILY[model]
        headline_rows = [
            row for row in closure_split
            if row["model"] == headline and row["condition_id"] == "oas_whitened_cosine_raw"
        ]
        query_rows = [row for row in results if row.get("model") == model and row["condition_id"] == "oasl3_wcos_query"]
        raw_p[model] = p_value
        summaries.append(
            {
                "dataset": "JVS_JVSMuSiC",
                "model": model,
                "layer": 3,
                "headline_model": headline,
                "headline_layer": int(headline.removeprefix(headline.split("_")[0] + "_l")),
                "oasl3_wcos_none_R1": float(np.mean([float(row["R1"]) for row in l3_rows])),
                "oasl3_wcos_query_R1": float(np.mean([float(row["R1"]) for row in query_rows])),
                "headline_wcos_none_R1": float(np.mean([float(row["R1"]) for row in headline_rows])),
                "D_R1": mean,
                "D_R1_ci_low": low,
                "D_R1_ci_high": high,
                "D_sign_flip_p": p_value,
                "n_unique_speakers": n_speakers,
                "translation_increment_R1": t_mean,
                "translation_increment_ci_low": t_low,
                "translation_increment_ci_high": t_high,
                "translation_increment_sign_flip_p": t_p,
                "x1x_translation_positive": int(t_low > 0),
            }
        )
    adjusted = holm_adjust(raw_p)
    for row in summaries:
        row["D_holm_p"] = adjusted[row["model"]][0]
        row["D_holm_reject"] = int(adjusted[row["model"]][1])
        row["outcome"] = (
            "X1-a: L3 stronger"
            if float(row["D_R1_ci_low"]) > 0
            and float(row["D_R1"]) >= 0.03
            and int(row["D_holm_reject"]) == 1
            else "X1-c: L3 weaker"
            if float(row["D_R1_ci_high"]) < 0 and int(row["D_holm_reject"]) == 1
            else "X1-b: no stable difference"
        )
    return results, fits, summaries


def run_x2(
    closure_results: Path, closure_run_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    abtt = read_csv(closure_results / "w2_abtt_results.csv")
    summaries = read_csv(closure_results / "w2_summary.csv")
    summary_by_model = {
        row["model"]: row for row in summaries if row["dataset"] == "JVS_JVSMuSiC"
    }
    output: list[dict[str, Any]] = []
    primary_p: dict[str, float] = {}
    for model_index, model in enumerate(SELECTED_K):
        if int(summary_by_model[model]["best_abtt_k"]) != SELECTED_K[model]:
            raise ExperimentError(f"selected-k mismatch for {model}")
        for k in ALL_K:
            grouped: dict[str, list[float]] = {}
            split_differences: list[float] = []
            bundle_hashes: list[str] = []
            transform_hashes: list[str] = []
            for seed in sorted({int(row["split_seed"]) for row in abtt if row["dataset"] == "JVS_JVSMuSiC" and row["model"] == model}):
                bundles: dict[str, dict[str, Any]] = {}
                csv_rows: dict[str, dict[str, str]] = {}
                for alignment in ("none", "query"):
                    condition = f"abtt_{k}_{alignment}"
                    matches = [
                        row for row in abtt
                        if row["dataset"] == "JVS_JVSMuSiC"
                        and row["model"] == model
                        and int(row["split_seed"]) == seed
                        and row["condition_id"] == condition
                        and row["control_type"] == "top_variance_pcs"
                    ]
                    if len(matches) != 1:
                        raise ExperimentError(f"missing unique W2 CSV row: {model}/{seed}/{condition}")
                    path = closure_run_root / "score_matrices" / f"JVS_JVSMuSiC__{model}__{seed}__{condition}.npz"
                    bundle = load_bundle(path)
                    stored_transform = str(bundle["transform_hash"].item())
                    if stored_transform != matches[0]["transform_hash"]:
                        raise ExperimentError(f"W2 transform hash mismatch: {model}/{seed}/{condition}")
                    if not math.isclose(bundle_r1(bundle), float(matches[0]["R1"]), rel_tol=0.0, abs_tol=1e-12):
                        raise ExperimentError(f"W2 bundle R1 mismatch: {model}/{seed}/{condition}")
                    bundles[alignment] = bundle
                    csv_rows[alignment] = matches[0]
                    bundle_hashes.append(file_hash(path))
                    transform_hashes.append(stored_transform)
                none_ids = [str(value) for value in bundles["none"]["ordered_gallery_speaker_ids"].tolist()]
                query_ids = [str(value) for value in bundles["query"]["ordered_gallery_speaker_ids"].tolist()]
                if none_ids != query_ids:
                    raise ExperimentError(f"W2 paired gallery mismatch: {model}/{seed}/k={k}")
                none_ranks, _ = ranks_from_scores(np.asarray(bundles["none"]["score_matrix"], dtype=np.float64))
                query_ranks, _ = ranks_from_scores(np.asarray(bundles["query"]["score_matrix"], dtype=np.float64))
                differences = (query_ranks == 1).astype(float) - (none_ranks == 1).astype(float)
                split_differences.append(float(differences.mean()))
                for speaker, difference in zip(none_ids, differences):
                    grouped.setdefault(speaker, []).append(float(difference))
            mean, low, high, p_value, n_speakers = speaker_interval(
                grouped, seed=716_200 + 10 * model_index + k
            )
            split_mean, split_low, split_high = normal_interval(split_differences)
            csv_none = [
                float(row["R1"]) for row in abtt
                if row["dataset"] == "JVS_JVSMuSiC" and row["model"] == model
                and row["condition_id"] == f"abtt_{k}_none" and row["control_type"] == "top_variance_pcs"
            ]
            csv_query = [
                float(row["R1"]) for row in abtt
                if row["dataset"] == "JVS_JVSMuSiC" and row["model"] == model
                and row["condition_id"] == f"abtt_{k}_query" and row["control_type"] == "top_variance_pcs"
            ]
            csv_increment = float(np.mean(csv_query) - np.mean(csv_none))
            if not math.isclose(split_mean, csv_increment, rel_tol=0.0, abs_tol=1e-12):
                raise ExperimentError(f"X2 split/bundle cross-check failed: {model}/k={k}")
            selected = int(k == SELECTED_K[model])
            if selected:
                stored = float(summary_by_model[model]["translation_increment_at_best_k_R1"])
                if not math.isclose(csv_increment, stored, rel_tol=0.0, abs_tol=1e-12):
                    raise ExperimentError(f"X2 summary cross-check failed: {model}/k={k}")
                primary_p[model] = p_value
            output.append(
                {
                    "dataset": "JVS_JVSMuSiC",
                    "model": model,
                    "condition_id": f"x2_abtt_{k}_query_minus_none",
                    "k": k,
                    "selected_k": selected,
                    "mean_R1_difference": mean,
                    "speaker_ci_low": low,
                    "speaker_ci_high": high,
                    "sign_flip_p": p_value,
                    "n_unique_speakers": n_speakers,
                    "split_mean_R1_difference": split_mean,
                    "split_normal_ci_low": split_low,
                    "split_normal_ci_high": split_high,
                    "csv_recomputed_R1_difference": csv_increment,
                    "bundle_hashes": ";".join(sorted(set(bundle_hashes))),
                    "transform_hashes": ";".join(sorted(set(transform_hashes))),
                    "bundle_count": len(bundle_hashes),
                    "bundle_reconstruction_status": "PASS",
                }
            )
    adjusted = holm_adjust(primary_p)
    primary_rows: list[dict[str, Any]] = []
    for row in output:
        if not int(row["selected_k"]):
            row["holm_p"] = ""
            row["holm_reject"] = ""
            continue
        holm_p, rejected = adjusted[row["model"]]
        row["holm_p"] = holm_p
        row["holm_reject"] = int(rejected)
        row["positive_after_holm"] = int(
            float(row["speaker_ci_low"]) > 0 and rejected
        )
        primary_rows.append(row)

    # W2-a recovery fractions were defined on none rows only; assert the source construction.
    for model in SELECTED_K:
        k = SELECTED_K[model]
        none = [
            float(row["R1"]) for row in abtt
            if row["dataset"] == "JVS_JVSMuSiC" and row["model"] == model
            and row["condition_id"] == f"abtt_{k}_none" and row["control_type"] == "top_variance_pcs"
        ]
        if len(none) != 20:
            raise ExperimentError(f"W2 recovery none-row assertion failed: {model}")
    return output, primary_rows


def make_figures(results_dir: Path, closure_results: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = results_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary = read_csv(results_dir / "x1_oas_l3_summary.csv")
    labels = {"wavlm_l3": "WavLM", "hubert_l3": "HuBERT"}
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.bar(x - 0.18, [float(row["headline_wcos_none_R1"]) for row in summary], 0.36, label="Headline layer")
    ax.bar(x + 0.18, [float(row["oasl3_wcos_none_R1"]) for row in summary], 0.36, label="Layer 3")
    ax.set_xticks(x, [labels[row["model"]] for row in summary])
    ax.set(ylabel="JVS OAS-whitened R@1", ylim=(0, 1), title="Full OAS whitening at layer 3")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "x1_oas_layer_bars.png", dpi=180)
    plt.close(fig)

    alignment = [
        row for row in read_csv(closure_results / "w2_spectrum_alignment.csv")
        if row["dataset"] == "JVS_JVSMuSiC"
    ]
    colors = {"wavlm_l12": "#4C78A8", "hubert_l6": "#F58518", "mert_l3": "#54A24B"}
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for model, color in colors.items():
        ks = list(range(1, 65))
        remaining = []
        for k in ks:
            values = [float(row["E_d_k"]) for row in alignment if row["model"] == model and int(row["k"]) == k]
            remaining.append(max(1e-8, 1.0 - float(np.mean(values))))
        ax.plot(ks, remaining, label=model, color=color)
    null_low, null_high = [], []
    for k in range(1, 65):
        selected = [row for row in alignment if int(row["k"]) == k]
        null_low.append(max(1e-8, 1.0 - float(np.mean([float(row["null_p97_5"]) for row in selected]))))
        null_high.append(max(1e-8, 1.0 - float(np.mean([float(row["null_p2_5"]) for row in selected]))))
    ax.fill_between(range(1, 65), null_low, null_high, color="#BBBBBB", alpha=0.45, label="Random-vector 95% band")
    ax.set_yscale("log")
    ax.set(xlabel="Top-k covariance PCs", ylabel="Remaining displacement energy 1 - E_d(k)", title="Dominant-direction alignment on a log scale")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "w2_alignment_log_inset.png", dpi=180)
    plt.close(fig)


def outcome_x1(summary: list[dict[str, str]]) -> tuple[str, list[str]]:
    outcomes = [row["outcome"] for row in summary]
    if any(value.startswith("X1-a") for value in outcomes):
        return "X1-a: L3 stronger", [row["model"] for row in summary if row["outcome"].startswith("X1-a")]
    if any(value.startswith("X1-c") for value in outcomes):
        return "X1-c: L3 weaker", []
    return "X1-b: no stable difference", []


def write_reports(
    results_dir: Path,
    run_root: Path,
    closure_run_root: Path,
    x3_status: str,
) -> None:
    x1 = read_csv(results_dir / "x1_oas_l3_summary.csv")
    x2 = [row for row in read_csv(results_dir / "x2_w2x_speaker_bootstrap.csv") if int(row["selected_k"]) == 1]
    x1_outcome, added_models = outcome_x1(x1)
    positive = [row for row in x2 if int(row.get("positive_after_holm") or 0) == 1]
    if len(positive) >= 2:
        x2_outcome = "X2-c: positive for more than one model"
    elif any(row["model"] == "wavlm_l12" for row in positive):
        x2_outcome = "X2-a: W2-x confirmed for WavLM"
    else:
        x2_outcome = "X2-b: no positive interval"

    if x1_outcome.startswith("X1-a"):
        x1_wording = "Layer selection compounds with second-order normalization for " + ", ".join(added_models) + "."
    elif x1_outcome.startswith("X1-b"):
        x1_wording = "The L3 advantage under translation correction does not transfer to the whitened backend; the headline layers remain representative."
    else:
        x1_wording = "The existing headline layer remains stronger under full OAS whitening; the layer-3 contrast is retained in the appendix table."
    if x2_outcome.startswith("X2-a"):
        x2_wording = "For WavLM, a small first-moment residual persists after removing 1-2 dominant PCs and vanishes by k=8; second-moment repair does not fully subsume translation."
    elif x2_outcome.startswith("X2-c"):
        x2_wording = "A small first-moment residual persists after selected dominant-PC removal in multiple SSL families; second-moment repair does not fully subsume translation."
    else:
        x2_wording = "Translation has no positive speaker-cluster interval after the selected removal; split-level trends are descriptive because the 20 splits reuse the same 100 speakers."

    rows = [
        "- WavLM OAS-whitened cosine at layer 12 (closure headline)",
        "- HuBERT OAS-whitened cosine at layer 6 (closure headline)",
        "- MERT OAS-whitened cosine at layer 3 (closure headline; copied reference)",
    ]
    for model in added_models:
        rows.append(f"- {model.split('_')[0].title()} OAS-whitened cosine at layer 3 (added by X1)")
    report = f"""# Identity residual paper-supplement gate addendum

## Unchanged framing

**F2 — Dominant-direction masking**, with **U1-a** and **U2-a** scope modifiers. This supplement does not change the selected framing.

## Gate table

| Experiment | Status | Outcome |
|---|---|---|
| R0' closure baseline | **PASS** | Six OAS headline rows reproduce within 0.5 pp; stored W2 selected-k increments reproduce within 1e-12 |
| X1 OAS at layer 3 | **PASS** | {x1_outcome} |
| X2 W2-x speaker bootstrap | **PASS** | {x2_outcome} |
| X3 log-scale figure | **{x3_status}** | Cosmetic figure only |

## Licensed wording

X1: {x1_wording}

X2: {x2_wording}

The closure gate report remains authoritative except that its W2-x sentence is {'amended by the X2 wording above' if positive else 'reaffirmed by the X2 wording above'}.

## Final main-table row list

{chr(10).join(rows)}

Run root: `{run_root}`  
Closure score source: `{closure_run_root}`
"""
    (results_dir / "gate_report_addendum.md").write_text(report, encoding="utf-8")
    readme = f"""# Identity residual paper supplement results

Additive closure of OAS-at-layer-3 (X1) and the W2-x speaker-cluster interval (X2). The completed 2026-07-15 closure artifacts were not modified.

- Run root: `{run_root}`
- Gate addendum: `gate_report_addendum.md`
- Framing remains F2 + U1-a + U2-a.
"""
    (results_dir / "README_results.md").write_text(readme, encoding="utf-8")


def write_experiment_card(
    results_dir: Path,
    run_root: Path,
    cache_root: Path,
    closure_results: Path,
    closure_run_root: Path,
) -> None:
    card = {
        "experiment": "identity_residual_paper_supplement_2026-07-16",
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "git_commit": current_git_commit(),
        "run_root": str(run_root.resolve()),
        "results_dir": str(results_dir.resolve()),
        "cache_root": str(cache_root.resolve()),
        "closure_results": str(closure_results.resolve()),
        "closure_run_root": str(closure_run_root.resolve()),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "jvs_models_x1": list(L3_MODELS),
        "x2_selected_k": SELECTED_K,
        "argv": sys.argv,
        "mount_facts": capture(["findmnt", "-T", str(run_root), "-o", "TARGET,SOURCE,FSTYPE,SIZE,AVAIL,USE%"]),
        "all_gpus": capture(["all_gpus"]),
        "all_cpus": capture(["all_cpus"]),
    }
    lines = []
    for key, value in card.items():
        if isinstance(value, (dict, list)):
            lines.append(f"{key}: {json.dumps(value, sort_keys=True)}")
        else:
            rendered = str(value).replace("\n", "\\n")
            lines.append(f"{key}: {rendered}")
    (results_dir / "experiment_card.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--closure-results", type=Path, default=CLOSURE_RESULTS)
    parser.add_argument("--closure-run-root", type=Path, default=CLOSURE_RUN_ROOT)
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Validate completed blocking artifacts and write metadata/reports without recomputation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.run_root / "score_matrices").mkdir(parents=True, exist_ok=True)
    score_link = args.results_dir / "score_matrices"
    if not score_link.exists() and not score_link.is_symlink():
        score_link.symlink_to((args.run_root / "score_matrices").resolve(), target_is_directory=True)

    if args.finalize_only:
        required = (
            "baseline_reproduction.csv",
            "fit_audit.csv",
            "x1_oas_l3_results.csv",
            "x1_oas_l3_summary.csv",
            "x2_w2x_speaker_bootstrap.csv",
            "figures/x1_oas_layer_bars.png",
            "figures/w2_alignment_log_inset.png",
        )
        missing = [name for name in required if not (args.results_dir / name).exists()]
        if missing:
            raise ExperimentError(f"cannot finalize; missing artifacts: {missing}")
        baseline_rows = read_csv(args.results_dir / "baseline_reproduction.csv")
        x2_rows = read_csv(args.results_dir / "x2_w2x_speaker_bootstrap.csv")
        if not all(row["status"] == "PASS" and row["w2_status"] == "PASS" for row in baseline_rows):
            raise ExperimentError("cannot finalize; R0' status is not PASS")
        if not all(row["bundle_reconstruction_status"] == "PASS" for row in x2_rows):
            raise ExperimentError("cannot finalize; an X2 bundle reconstruction did not pass")
        write_experiment_card(
            args.results_dir, args.run_root, args.cache_root, args.closure_results, args.closure_run_root
        )
        write_reports(args.results_dir, args.run_root, args.closure_run_root, "PASS")
        print(json.dumps({"status": "PASS", "mode": "finalize-only", "results_dir": str(args.results_dir)}))
        return 0

    baseline, w2_checks = run_r0(args.cache_root, args.closure_results)
    for row in baseline:
        match = next(check for check in w2_checks if check["model"] == row["model"])
        row.update({f"w2_{key}": value for key, value in match.items() if key != "model"})
    write_csv(baseline, args.results_dir / "baseline_reproduction.csv")

    x1_results, fit_rows, x1_summary = run_x1(
        args.cache_root, args.run_root, args.closure_results, args.closure_run_root
    )
    write_csv(fit_rows, args.results_dir / "fit_audit.csv")
    write_csv(x1_results, args.results_dir / "x1_oas_l3_results.csv")
    write_csv(x1_summary, args.results_dir / "x1_oas_l3_summary.csv")

    x2_rows, _primary = run_x2(args.closure_results, args.closure_run_root)
    write_csv(x2_rows, args.results_dir / "x2_w2x_speaker_bootstrap.csv")

    x3_status = "PASS"
    try:
        make_figures(args.results_dir, args.closure_results)
    except Exception as exc:  # X3 is explicitly non-blocking.
        x3_status = f"FAIL (non-blocking: {type(exc).__name__}: {exc})"
    write_experiment_card(
        args.results_dir, args.run_root, args.cache_root, args.closure_results, args.closure_run_root
    )
    write_reports(args.results_dir, args.run_root, args.closure_run_root, x3_status)
    print(json.dumps({"status": "PASS", "results_dir": str(args.results_dir), "x3": x3_status}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
