#!/usr/bin/env python
"""Protocol robustness analyses for the global speech-to-singing displacement.

All transformations are estimated from train speakers.  This runner deliberately
does not fit a per-speaker residual mapper or use test singing vectors to select
a correction direction.
"""
from __future__ import annotations

import argparse
import math
import os
import socket
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probing.run_identity_residual_final_validation import mode_probe_centroid  # noqa: E402
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
    write_csv,
    write_yaml,
)
from research_utils import current_git_commit  # noqa: E402


JVS_SEEDS = [13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
MODELS = ["wavlm_l12", "mert_l3", "hubert_l6"]
JVS_MANIFEST = Path("/localdisk/bowen/singing_identity/runs/jvs_music_retrieval_2026-07-08/manifests/jvs_music_utterances.jsonl")
JVS_FEATURE_ROOT = Path("/localdisk/bowen/singing_identity/features/jvs_music_2026-07-08")
GTS_PAIRS = Path("/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local/manifests/gtsinger_pairs.jsonl")
GTS_UTTERANCES = Path("/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local/manifests/gtsinger_utterances.jsonl")
GTS_FEATURE_ROOT = Path("/localdisk/bowen/singing_identity/features/stage1_repaired_200_fresh_local")
GTS_PRECOMPUTED_VECTOR_CACHE = Path("/localdisk/bowen/singing_identity/runs/identity_residual_same_text_2026-07-10/cache")


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def cosine_matrix(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    return l2(query) @ l2(gallery).T


def split_indices(speakers: list[str], seed: int, protocol: str) -> tuple[dict[str, str], np.ndarray, np.ndarray]:
    split = make_speaker_split(speakers, seed, protocol)
    train = np.asarray([i for i, speaker in enumerate(speakers) if split[speaker] == "train"], dtype=int)
    test = np.asarray([i for i, speaker in enumerate(speakers) if split[speaker] == "test"], dtype=int)
    return split, train, test


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
    """Empirical test-set EER for descriptive ROC reporting only."""
    values = np.unique(np.concatenate([genuine, impostor]))
    if not len(values):
        return float("nan")
    candidates = np.concatenate([[np.nextafter(values[0], -np.inf)], values, [np.nextafter(values[-1], np.inf)]])
    fmr = np.asarray([np.mean(impostor >= t) for t in candidates])
    fnmr = np.asarray([np.mean(genuine < t) for t in candidates])
    index = int(np.argmin(np.abs(fmr - fnmr)))
    return float((fmr[index] + fnmr[index]) / 2.0)


def train_threshold_at_fmr(impostor: np.ndarray, target_fmr: float) -> tuple[float, float]:
    """Pick the lowest threshold whose FMR is at most target, using train trials."""
    values = np.unique(np.asarray(impostor, dtype=np.float64))
    if not len(values):
        return float("nan"), float("nan")
    candidates = np.concatenate([[np.nextafter(values[0], -np.inf)], values, [np.nextafter(values[-1], np.inf)]])
    eligible = [float(t) for t in candidates if float(np.mean(impostor >= t)) <= target_fmr]
    threshold = min(eligible) if eligible else float(np.nextafter(values[-1], np.inf))
    return threshold, float(np.mean(impostor >= threshold))


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


def vector_mean(vectors: dict[str, np.ndarray], speakers: list[str], indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    speech = np.vstack([vectors["speech"][speakers[i]].mean(axis=0) for i in indices])
    singing = np.vstack([vectors["singing"][speakers[i]].mean(axis=0) for i in indices])
    return speech, singing


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def load_jvs(model: str, cache_dir: Path) -> tuple[list[str], dict[str, dict[str, np.ndarray]], str]:
    rows, x, meta = load_feature_cache(read_jsonl(JVS_MANIFEST), JVS_FEATURE_ROOT, model, cache_dir, 20, False)
    vectors: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for row, vector in zip(rows, x):
        mode = canonical_mode(row)
        if mode in {"speech", "singing"}:
            vectors[str(row["speaker_id"])][mode].append(vector)
    speakers = sorted(s for s, modes in vectors.items() if modes["speech"] and modes["singing"])
    return speakers, {s: {m: np.vstack(vectors[s][m]) for m in ["speech", "singing"]} for s in speakers}, str(meta["layer"])


def load_gtsinger(model: str, cache_dir: Path) -> tuple[list[str], dict[str, dict[str, np.ndarray]], str, int]:
    pair_rows = read_jsonl(GTS_PAIRS)
    utterances = {str(row["utt_id"]): row for row in read_jsonl(GTS_UTTERANCES)}
    pairs, _audit = select_clean_control_pairs(pair_rows, utterances)
    # This is the exact clean-pair vector cache used by the completed same-text
    # validation. Reusing it avoids rescanning thousands of frame archives.
    speech, singing, meta = load_pair_vectors(pairs, GTS_FEATURE_ROOT, model, GTS_PRECOMPUTED_VECTOR_CACHE if GTS_PRECOMPUTED_VECTOR_CACHE.exists() else cache_dir)
    vectors: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for pair, sv, gv in zip(pairs, speech, singing):
        speaker = str(pair["speaker_id"])
        vectors[speaker]["speech"].append(sv)
        vectors[speaker]["singing"].append(gv)
    speakers = sorted(vectors)
    return speakers, {s: {m: np.vstack(vectors[s][m]) for m in ["speech", "singing"]} for s in speakers}, str(meta["layer"]), len(pairs)


def add_verification_rows(
    rows: list[dict[str, Any]],
    curves: list[dict[str, Any]],
    dataset: str,
    model: str,
    layer: str,
    seed: int,
    test_speakers: list[str],
    speech_train: np.ndarray,
    singing_train: np.ndarray,
    speech_test: np.ndarray,
    singing_test: np.ndarray,
    mu: np.ndarray,
    control_draws: int,
) -> None:
    variants: list[tuple[str, np.ndarray, np.ndarray, int | str]] = [
        ("raw", speech_test, speech_train, ""),
        ("correct_sign_global", speech_test + mu, speech_train + mu, ""),
        ("wrong_sign_global", speech_test - mu, speech_train - mu, ""),
    ]
    rng = np.random.default_rng(seed * 1009 + len(model))
    for draw in range(control_draws):
        v = rng.normal(size=mu.shape)
        v *= np.linalg.norm(mu) / max(np.linalg.norm(v), 1e-12)
        variants.append(("random_same_norm", speech_test + v, speech_train + v, draw))
    for variant, query, calibration_query, draw in variants:
        metrics = verification_metrics(query, singing_test, calibration_query, singing_train)
        scores = cosine_matrix(query, singing_test)
        genuine, impostor = genuine_impostor(scores)
        rows.append({
            "dataset": dataset, "model": model, "layer": layer, "split_seed": seed,
            "variant": variant, "control_draw": draw,
            "test_speakers": ";".join(test_speakers),
            "trial_construction": "one_speaker_one_vote_centroids; diagonal_cross_mode_pairs_genuine; all_offdiagonal_cross_mode_pairs_impostor; train_speakers_excluded_from_test_trials",
            "threshold_policy": "TMR thresholds chosen only from train-speaker impostor trials; test EER is descriptive ROC crossing",
            **metrics,
        })
        if draw == "":
            for point in roc_det_points(genuine, impostor):
                curves.append({
                    "dataset": dataset, "model": model, "layer": layer, "split_seed": seed,
                    "variant": variant, "control_draw": draw, **point,
                })


def add_geometry_rows(
    per_speaker: list[dict[str, Any]],
    energy_rows: list[dict[str, Any]],
    relationship_rows: list[dict[str, Any]],
    dataset: str,
    model: str,
    layer: str,
    seed: int,
    speakers: list[str],
    train: np.ndarray,
    test: np.ndarray,
    speech: np.ndarray,
    singing: np.ndarray,
    mu: np.ndarray,
) -> None:
    train_delta = singing[train] - speech[train]
    centered = train_delta - mu
    residual_norm = np.linalg.norm(train_delta, axis=1)
    centered_norm = np.linalg.norm(centered, axis=1)
    denom = float(np.sum(train_delta ** 2))
    energy = 1.0 - float(np.sum(centered ** 2)) / max(denom, 1e-12)
    # The non-zero covariance spectrum is exactly recoverable from the much
    # smaller speaker Gram matrix, avoiding repeated wide-matrix SVDs.
    eig = np.linalg.eigvalsh(centered @ centered.T)[::-1] / max(len(train_delta) - 1, 1)
    eig = np.maximum(eig, 0.0)
    p = eig / max(float(eig.sum()), 1e-12)
    effective_rank = float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))) if np.any(p > 0) else 0.0
    energy_rows.append({
        "dataset": dataset, "model": model, "layer": layer, "split_seed": seed,
        "train_speakers": ";".join(speakers[i] for i in train),
        "shared_mean_explained_energy": energy,
        "mean_residual_norm": float(residual_norm.mean()), "mean_centered_residual_norm": float(centered_norm.mean()),
        "global_mean_norm": float(np.linalg.norm(mu)), "effective_rank_centered": effective_rank,
        "top1_centered_variance": float(p[:1].sum()), "top2_centered_variance": float(p[:2].sum()),
        "top4_centered_variance": float(p[:4].sum()), "top8_centered_variance": float(p[:8].sum()),
        "largest_centered_eigenvalue": float(eig[0]) if len(eig) else float("nan"),
        "analysis_population": "train_speakers_only",
    })
    raw_scores = cosine_matrix(speech[test], singing[test])
    corr_scores = cosine_matrix(speech[test] + mu, singing[test])
    raw_rank, _ = ranks_from_scores(raw_scores)
    corr_rank, _ = ranks_from_scores(corr_scores)
    alignment = np.asarray([cosine(singing[i] - speech[i], mu) for i in test])
    for j, index in enumerate(test):
        per_speaker.append({
            "dataset": dataset, "model": model, "layer": layer, "split_seed": seed,
            "speaker_id": speakers[index], "alignment_cosine_to_train_mu": float(alignment[j]),
            "test_residual_norm": float(np.linalg.norm(singing[index] - speech[index])),
            "raw_rank": int(raw_rank[j]), "corrected_rank": int(corr_rank[j]),
            "rank_improvement": int(raw_rank[j] - corr_rank[j]),
            "raw_hit1": int(raw_rank[j] == 1), "corrected_hit1": int(corr_rank[j] == 1),
            "retrieval_success_change": int(corr_rank[j] == 1) - int(raw_rank[j] == 1),
            "raw_genuine_score": float(raw_scores[j, j]),
            "corrected_genuine_score": float(corr_scores[j, j]),
            "raw_impostor_score_mean": float(np.delete(raw_scores[j], j).mean()),
            "corrected_impostor_score_mean": float(np.delete(corr_scores[j], j).mean()),
        })
    relationship_rows.append({
        "dataset": dataset, "model": model, "layer": layer, "split_seed": seed,
        "n_test_speakers": len(test),
        "spearman_alignment_vs_raw_rank": spearman(alignment, raw_rank.astype(float)),
        "spearman_alignment_vs_rank_improvement": spearman(alignment, (raw_rank - corr_rank).astype(float)),
    })


def add_reference_rows(
    rows: list[dict[str, Any]],
    dataset: str,
    model: str,
    layer: str,
    seed: int,
    vectors: dict[str, dict[str, np.ndarray]],
    speakers: list[str],
    train: np.ndarray,
    test: np.ndarray,
    mu: np.ndarray,
    draws: int,
    budgets: list[int],
) -> None:
    train_labels = [speakers[i] for i in train]
    test_labels = [speakers[i] for i in test]
    speech_train, singing_train = vector_mean({"speech": {s: vectors[s]["speech"] for s in speakers}, "singing": {s: vectors[s]["singing"] for s in speakers}}, speakers, train)
    gallery = np.vstack([vectors[s]["singing"].mean(axis=0) for s in test_labels])
    for budget in budgets:
        full = budget == -1
        for draw in range(1 if full else draws):
            rng = np.random.default_rng(seed * 1000003 + draw * 1009 + (budget if budget > 0 else 997))
            references = []
            effective = []
            for speaker in test_labels:
                source = vectors[speaker]["speech"]
                count = len(source) if full else min(budget, len(source))
                selected = source if full else source[rng.choice(len(source), size=count, replace=False)]
                references.append(selected.mean(axis=0))
                effective.append(count)
            query = np.vstack(references)
            for variant, q in [("raw", query), ("correct_sign_global", query + mu)]:
                rows.append({
                    "dataset": dataset, "model": model, "layer": layer, "split_seed": seed,
                    "reference_budget": "full" if full else budget, "reference_draw": draw,
                    "effective_reference_min": min(effective), "effective_reference_max": max(effective),
                    "variant": variant, "test_speakers": ";".join(test_labels),
                    "gallery": "full_test_singing_centroid_per_speaker",
                    **verification_metrics(q, gallery, speech_train if variant == "raw" else speech_train + mu, singing_train),
                })


def add_gallery_rows(
    rows: list[dict[str, Any]],
    dataset: str,
    model: str,
    layer: str,
    seed: int,
    speakers: list[str],
    train: np.ndarray,
    test: np.ndarray,
    speech: np.ndarray,
    singing: np.ndarray,
    mu: np.ndarray,
    draws: int,
) -> None:
    eligible = np.asarray(test, dtype=int)
    sizes = [size for size in [5, 10, 20] if size <= len(eligible)]
    for size in sizes:
        for draw in range(draws):
            rng = np.random.default_rng(seed * 1000003 + size * 1009 + draw)
            selected = np.sort(rng.choice(eligible, size=size, replace=False))
            test_labels = [speakers[i] for i in selected]
            chance = 1.0 / size
            for variant, query in [("raw", speech[selected]), ("correct_sign_global", speech[selected] + mu)]:
                calibration_query = speech[train] if variant == "raw" else speech[train] + mu
                metrics = verification_metrics(query, singing[selected], calibration_query, singing[train])
                rows.append({
                    "dataset": dataset, "model": model, "layer": layer, "split_seed": seed,
                    "gallery_draw": draw, "gallery_size": size, "variant": variant,
                    "test_speakers": ";".join(test_labels), "chance_R1": chance,
                    "normalized_R1_above_chance": (float(metrics["R1"]) - chance) / max(1.0 - chance, 1e-12),
                    "gallery_policy": "sampled_from_held_out_test_speakers_only; target_included_once; identical_gallery_for_raw_and_corrected",
                    **metrics,
                })


def summary(rows: list[dict[str, Any]], keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    output = []
    for group, items in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        result: dict[str, Any] = {key: value for key, value in zip(keys, group)}
        result["rows"] = len(items)
        for metric in metrics:
            values = np.asarray([float(item[metric]) for item in items if metric in item and math.isfinite(float(item[metric]))])
            result[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            result[f"{metric}_p2_5"] = float(np.percentile(values, 2.5)) if len(values) else float("nan")
            result[f"{metric}_p97_5"] = float(np.percentile(values, 97.5)) if len(values) else float("nan")
        output.append(result)
    return output


def random_means(rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["variant"] == "random_same_norm":
            groups[(row["dataset"], row["model"], row["layer"], int(row["split_seed"]))].append(row)
    output = []
    for (dataset, model, layer, seed), values in groups.items():
        averaged = {}
        for metric in metrics:
            metric_values = np.asarray([float(value[metric]) for value in values], dtype=float)
            metric_values = metric_values[np.isfinite(metric_values)]
            averaged[metric] = float(metric_values.mean()) if len(metric_values) else float("nan")
        output.append({"dataset": dataset, "model": model, "layer": layer, "split_seed": seed, "variant": "random_same_norm_mean", **averaged})
    return output


def paired_deltas(rows: list[dict[str, Any]], metrics: list[str], bootstrap_samples: int) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    for row in rows:
        if row["variant"] in {"raw", "correct_sign_global", "wrong_sign_global", "random_same_norm_mean"}:
            by_key[(row["dataset"], row["model"], row["layer"], int(row["split_seed"]), row["variant"])] = row
    output = []
    rng = np.random.default_rng(20260710)
    for target in ["correct_sign_global", "wrong_sign_global", "random_same_norm_mean"]:
        groups: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for (dataset, model, layer, seed, variant), value in by_key.items():
            if variant == target and (dataset, model, layer, seed, "raw") in by_key:
                groups[(dataset, model, layer)].append((value, by_key[(dataset, model, layer, seed, "raw")]))
        for (dataset, model, layer), pairs in groups.items():
            for metric in metrics:
                delta = np.asarray([float(a[metric]) - float(b[metric]) for a, b in pairs if math.isfinite(float(a[metric])) and math.isfinite(float(b[metric]))])
                if not len(delta):
                    continue
                draws = rng.integers(0, len(delta), size=(bootstrap_samples, len(delta)))
                boot = delta[draws].mean(axis=1)
                output.append({
                    "dataset": dataset, "model": model, "layer": layer, "target_variant": target, "baseline_variant": "raw", "metric": metric,
                    "paired_splits": len(delta), "delta_mean": float(delta.mean()), "delta_CI_low": float(np.percentile(boot, 2.5)), "delta_CI_high": float(np.percentile(boot, 97.5)),
                    "CI_method": f"paired_split_bootstrap_{bootstrap_samples}; repeated_splits_not_independent_speakers",
                })
    return output


def write_roc_det_plots(curves: list[dict[str, Any]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in curves:
        grouped[(row["dataset"], row["model"], row["layer"], row["variant"])].append(row)
    by_model: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for (*key, variant), rows in grouped.items():
        by_model[tuple(key)][variant] = rows
    grid = np.linspace(0.0, 1.0, 201)
    for (dataset, model, layer), variants in by_model.items():
        fig, ax = plt.subplots(figsize=(5.0, 4.2))
        for variant, rows in sorted(variants.items()):
            by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                by_seed[int(row["split_seed"])].append(row)
            interpolated = []
            for points in by_seed.values():
                points = sorted(points, key=lambda row: float(row["FMR"]))
                x = np.asarray([float(row["FMR"]) for row in points])
                y = np.asarray([float(row["TMR"]) for row in points])
                x_unique, inverse = np.unique(x, return_inverse=True)
                y_unique = np.asarray([y[inverse == i].max() for i in range(len(x_unique))])
                interpolated.append(np.interp(grid, x_unique, y_unique, left=y_unique[0], right=y_unique[-1]))
            if interpolated:
                ax.plot(grid, np.mean(interpolated, axis=0), label=variant)
        ax.plot([0, 1], [0, 1], color="0.55", linestyle="--", linewidth=1, label="chance")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False match rate", ylabel="True match rate", title=f"{dataset} {model} L{layer}: cross-mode ROC")
        ax.legend(fontsize=7)
        fig.tight_layout()
        safe = f"{dataset}_{model}_L{layer}".replace("/", "_")
        fig.savefig(out_dir / f"verification_roc_{safe}.png", dpi=170)
        plt.close(fig)


def speaker_uncertainty(rows: list[dict[str, Any]], bootstrap_samples: int, permutation_samples: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["model"], row["speaker_id"])].append(row)
    aggregate = []
    for (dataset, model, speaker), values in groups.items():
        aggregate.append({
            "dataset": dataset, "model": model, "speaker_id": speaker, "test_split_appearances": len(values),
            "raw_rank_mean": float(np.mean([v["raw_rank"] for v in values])), "corrected_rank_mean": float(np.mean([v["corrected_rank"] for v in values])),
            "rank_improvement_mean": float(np.mean([v["rank_improvement"] for v in values])),
            "raw_hit1_rate": float(np.mean([v["raw_hit1"] for v in values])), "corrected_hit1_rate": float(np.mean([v["corrected_hit1"] for v in values])),
            "retrieval_success_change_mean": float(np.mean([v["retrieval_success_change"] for v in values])),
            "alignment_cosine_mean": float(np.mean([v["alignment_cosine_to_train_mu"] for v in values])),
            "raw_genuine_score_mean": float(np.mean([v["raw_genuine_score"] for v in values])),
            "corrected_genuine_score_mean": float(np.mean([v["corrected_genuine_score"] for v in values])),
            "raw_impostor_score_mean": float(np.mean([v["raw_impostor_score_mean"] for v in values])),
            "corrected_impostor_score_mean": float(np.mean([v["corrected_impostor_score_mean"] for v in values])),
        })
    summaries = []
    rng = np.random.default_rng(20260710)
    for (dataset, model), values in defaultdict(list, {(r["dataset"], r["model"]): [] for r in aggregate}).items():
        pass
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate:
        grouped[(row["dataset"], row["model"])].append(row)
    for (dataset, model), values in sorted(grouped.items()):
        delta = np.asarray([float(v["rank_improvement_mean"]) for v in values])
        score_delta = np.asarray([
            (float(v["corrected_genuine_score_mean"]) - float(v["corrected_impostor_score_mean"]))
            - (float(v["raw_genuine_score_mean"]) - float(v["raw_impostor_score_mean"]))
            for v in values
        ])
        draw = rng.integers(0, len(delta), size=(bootstrap_samples, len(delta)))
        boot = delta[draw].mean(axis=1)
        score_boot = score_delta[draw].mean(axis=1)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(permutation_samples, len(delta)))
        observed = abs(float(delta.mean()))
        pvalue = float((1 + np.sum(np.abs((signs * delta).mean(axis=1)) >= observed)) / (permutation_samples + 1))
        summaries.append({
            "dataset": dataset, "model": model, "unique_test_speakers": len(delta), "mean_rank_improvement": float(delta.mean()), "median_rank_improvement": float(np.median(delta)),
            "mean_rank_improvement_CI_low": float(np.percentile(boot, 2.5)), "mean_rank_improvement_CI_high": float(np.percentile(boot, 97.5)),
            "positive_speaker_fraction": float(np.mean(delta > 0)), "negative_speaker_fraction": float(np.mean(delta < 0)), "unchanged_speaker_fraction": float(np.mean(delta == 0)),
            "paired_sign_permutation_pvalue_two_sided": pvalue, "unit": "speaker_aggregate_over_test_split_appearances",
            "mean_score_separation_improvement": float(score_delta.mean()),
            "mean_score_separation_improvement_CI_low": float(np.percentile(score_boot, 2.5)),
            "mean_score_separation_improvement_CI_high": float(np.percentile(score_boot, 97.5)),
        })
    return aggregate, summaries


def run(args: argparse.Namespace) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.run_root.mkdir(parents=True, exist_ok=True)
    verification: list[dict[str, Any]] = []
    verification_curves: list[dict[str, Any]] = []
    reference: list[dict[str, Any]] = []
    gallery: list[dict[str, Any]] = []
    geometry: list[dict[str, Any]] = []
    energy: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for model in args.models:
            if dataset == "JVS_JVSMuSiC":
                speakers, vectors, layer = load_jvs(model, args.run_root / dataset / model / "cache")
                seeds, protocol, clean_pairs = JVS_SEEDS, "jvs_60_20_20", ""
            else:
                speakers, vectors, layer, clean_pairs = load_gtsinger(model, args.run_root / dataset / model / "cache")
                seeds, protocol = GTSINGER_SEEDS, "gtsinger_10_0_10"
            audit.append({"dataset": dataset, "model": model, "layer": layer, "speakers": len(speakers), "clean_same_text_control_pairs": clean_pairs, "split_protocol": protocol, "gallery_sizes_not_run": "JVS:50 unavailable under fixed 60/20/20 test split; GTSinger:20 unavailable under fixed 10/10 split"})
            speech_all = np.vstack([vectors[s]["speech"].mean(axis=0) for s in speakers])
            singing_all = np.vstack([vectors[s]["singing"].mean(axis=0) for s in speakers])
            for seed in seeds:
                _split, train, test = split_indices(speakers, seed, protocol)
                speech_train, singing_train = speech_all[train], singing_all[train]
                speech_test, singing_test = speech_all[test], singing_all[test]
                mu = (singing_train - speech_train).mean(axis=0)
                test_labels = [speakers[i] for i in test]
                add_verification_rows(verification, verification_curves, dataset, model, layer, seed, test_labels, speech_train, singing_train, speech_test, singing_test, mu, args.control_draws)
                add_reference_rows(reference, dataset, model, layer, seed, vectors, speakers, train, test, mu, args.reference_draws, [1, 2, 5, 10, -1])
                add_gallery_rows(gallery, dataset, model, layer, seed, speakers, train, test, speech_all, singing_all, mu, args.gallery_draws)
                add_geometry_rows(geometry, energy, relationships, dataset, model, layer, seed, speakers, train, test, speech_all, singing_all, mu)
    metrics = ["R1", "R5", "mean_rank", "MRR", "EER_test_descriptive", "TMR_at_FMR_1pct", "TMR_at_FMR_0_1pct", "genuine_score_mean", "impostor_score_mean", "genuine_minus_impostor"]
    random_rows = random_means(verification, metrics)
    all_verification = verification + random_rows
    speaker_aggregate, speaker_summary = speaker_uncertainty(geometry, args.bootstrap_samples, args.permutation_samples)
    write_csv(audit, args.results_dir / "data_and_protocol_audit.csv")
    write_csv(verification, args.results_dir / "verification_per_split.csv")
    write_csv(verification_curves, args.results_dir / "verification_roc_det_per_split.csv")
    write_roc_det_plots(verification_curves, args.results_dir)
    write_csv(summary(all_verification, ["dataset", "model", "layer", "variant"], metrics), args.results_dir / "verification_summary.csv")
    write_csv(paired_deltas(all_verification, ["R1", "EER_test_descriptive", "TMR_at_FMR_1pct", "genuine_minus_impostor"], args.bootstrap_samples), args.results_dir / "verification_paired_split_deltas.csv")
    write_csv(reference, args.results_dir / "reference_budget_per_split.csv")
    write_csv(summary(reference, ["dataset", "model", "layer", "reference_budget", "variant"], metrics), args.results_dir / "reference_budget_summary.csv")
    write_csv(gallery, args.results_dir / "gallery_size_per_split.csv")
    write_csv(summary(gallery, ["dataset", "model", "layer", "gallery_size", "variant"], metrics + ["normalized_R1_above_chance"]), args.results_dir / "gallery_size_summary.csv")
    write_csv(geometry, args.results_dir / "heldout_residual_alignment_per_speaker_split.csv")
    write_csv(speaker_aggregate, args.results_dir / "speaker_level_aggregate.csv")
    write_csv(speaker_summary, args.results_dir / "speaker_level_uncertainty.csv")
    write_csv(summary(speaker_aggregate, ["dataset", "model"], ["alignment_cosine_mean", "rank_improvement_mean", "retrieval_success_change_mean"]), args.results_dir / "heldout_alignment_summary.csv")
    write_csv(energy, args.results_dir / "global_mean_explained_energy_per_split.csv")
    write_csv(summary(energy, ["dataset", "model", "layer"], ["shared_mean_explained_energy", "mean_residual_norm", "mean_centered_residual_norm", "global_mean_norm", "effective_rank_centered", "top1_centered_variance", "top2_centered_variance", "top4_centered_variance", "top8_centered_variance"]), args.results_dir / "global_mean_explained_energy_summary.csv")
    write_csv(relationships, args.results_dir / "alignment_retrieval_relationship_per_split.csv")
    write_yaml({
        "experiment_name": "identity_residual_protocol_robustness_2026-07-10", "hostname": socket.gethostname(), "python": sys.executable, "git_commit": current_git_commit(),
        "run_root": str(args.run_root), "results_dir": str(args.results_dir), "cache_root": os.environ.get("XDG_CACHE_HOME", ""), "models": list(args.models), "datasets": list(args.datasets),
        "gtsinger_full_utterance_vector_cache": str(GTS_PRECOMPUTED_VECTOR_CACHE),
        "global_direction_fit_population": "train_speakers_only", "verification_threshold_policy": "TMR thresholds use train-speaker trial labels only; EER is held-out descriptive ROC statistic", "low_rank_sweep": "not_run: a non-oracle test-speech-only coefficient rule was not pre-specified; fitting one would resume the stopped individualized mapper line", "known_limitations": "JVS speech/singing contents are unmatched; GTSinger has 20 speakers with language confounding; repeated split intervals are supplemented by speaker-aggregated bootstrap",
    }, args.results_dir / "experiment_card.yaml")
    report = [
        "# Identity Residual Protocol Robustness", "", "## Scope", "", "This suite reevaluates the train-estimated global displacement with verification trials, limited speech references, held-out residual geometry, speaker-level uncertainty, and gallery-size changes. All corrected test queries use only a vector estimated from train speakers.", "", "## Threshold Policy", "", "Genuine trials are same-speaker speech--singing centroid pairs; impostors are all cross-speaker pairs. Train speakers never occur in test trials. EER is reported as a descriptive held-out ROC crossing. TMR operating thresholds are determined from train-speaker impostor trials only; 0.1% FMR is marked insufficient when fewer than 1,000 train impostors exist.", "", "## Low-rank Decision", "", "Only the residual spectrum is reported. Applying train PCA directions to a held-out speaker requires a pre-specified test-speech-only rule for the coefficients; the available alternative is to fit a predictor, which would restart the individualized mapper line that this project has stopped. No target-singing or oracle projection is used.", "", "## Output Map", "", "- `verification_summary.csv` and `verification_paired_split_deltas.csv`: cross-mode verification and controls.", "- `reference_budget_summary.csv`: one, two, five, ten, and full speech-reference budgets.", "- `heldout_alignment_summary.csv` and `global_mean_explained_energy_summary.csv`: alignment, variance, and residual-spectrum accounting.", "- `speaker_level_uncertainty.csv`: speaker-aggregated bootstrap and paired sign-permutation result.", "- `gallery_size_summary.csv`: sampled held-out gallery robustness. A requested gallery size is omitted when the fixed split has too few held-out speakers.", "",
    ]
    (args.results_dir / "README_results.md").write_text("\n".join(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Protocol robustness analyses for the global identity residual.")
    parser.add_argument("--results-dir", type=Path, default=Path("results/identity_residual_protocol_robustness_2026-07-10"))
    parser.add_argument("--run-root", type=Path, default=Path("/localdisk/bowen/singing_identity/runs/identity_residual_protocol_robustness_2026-07-10"))
    parser.add_argument("--datasets", nargs="+", choices=["JVS_JVSMuSiC", "GTSinger_same_text_control"], default=["JVS_JVSMuSiC", "GTSinger_same_text_control"])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--control-draws", type=int, default=50)
    parser.add_argument("--reference-draws", type=int, default=25)
    parser.add_argument("--gallery-draws", type=int, default=25)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--permutation-samples", type=int, default=10000)
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
