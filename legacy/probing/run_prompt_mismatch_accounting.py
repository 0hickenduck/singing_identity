#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
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
    cosine_similarity_matrix,
    deterministic_speaker_split,
    kfold_speaker_splits,
    load_feature_vector,
    read_table,
    recall_at_k,
    validate_pairs,
    validate_utterances,
    write_json,
    write_table,
)


MODEL_ORDER = ["M0", "M1", "M2", "M3", "M4", "M5"]
ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
SPECIAL_PHONES = {"<AP>", "<SP>", "<SIL>", "<sil>", "sil", "sp", ""}


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def as_bool(value: Any) -> float:
    return 1.0 if str(value).strip().lower() in {"1", "true", "yes", "y"} else 0.0


def phone_tokens(seq: Any) -> list[str]:
    tokens = [token.strip() for token in str(seq or "").split()]
    return [token for token in tokens if token and token not in SPECIAL_PHONES]


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    return np.sum(a * b, axis=1) / denom


def pearson_corr(x: list[float], y: list[float]) -> float:
    if len(x) < 3 or len(y) < 3:
        return float("nan")
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if int(mask.sum()) < 3:
        return float("nan")
    xa = xa[mask]
    ya = ya[mask]
    xs = xa.std()
    ys = ya.std()
    if xs < 1e-12 or ys < 1e-12:
        return float("nan")
    return float(np.mean(((xa - xa.mean()) / xs) * ((ya - ya.mean()) / ys)))


def ranks(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr)
    out = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        out[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return out.tolist()


def corr_summary(x: list[float], y: list[float]) -> dict[str, float | int]:
    return {
        "n": int(len([1 for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)])),
        "pearson": pearson_corr(x, y),
        "spearman": pearson_corr(ranks(x), ranks(y)) if len(x) >= 3 else float("nan"),
    }


def one_hot(values: list[str], categories: list[str]) -> tuple[np.ndarray, list[str]]:
    if not categories:
        return np.zeros((len(values), 0), dtype=np.float64), []
    index = {value: i for i, value in enumerate(categories)}
    mat = np.zeros((len(values), len(categories)), dtype=np.float64)
    for row_idx, value in enumerate(values):
        if value in index:
            mat[row_idx, index[value]] = 1.0
    return mat, categories


def zscore_from_train(x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (x - mean) / std, mean, std


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.column_stack([np.ones(x_train.shape[0]), x_train])
    penalty = np.eye(x_aug.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    lhs = x_aug.T @ x_aug + penalty
    rhs = x_aug.T @ y_train
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


def predict_ridge(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(x.shape[0]), x]) @ beta


def fit_predict_model(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    dev_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    x_z, _, _ = zscore_from_train(x, train_idx)
    best_alpha = ALPHA_GRID[0]
    best_dev_mse = float("inf")
    if len(dev_idx) == 0:
        dev_idx = train_idx
    for alpha in ALPHA_GRID:
        beta = fit_ridge(x_z[train_idx], y[train_idx], alpha)
        dev_pred = predict_ridge(x_z[dev_idx], beta)
        dev_mse = float(np.mean(np.square(y[dev_idx] - dev_pred)))
        if dev_mse < best_dev_mse:
            best_dev_mse = dev_mse
            best_alpha = alpha
    beta = fit_ridge(x_z[train_idx], y[train_idx], best_alpha)
    return predict_ridge(x_z[test_idx], beta), float(best_alpha), float(best_dev_mse)


def retrieval_metrics(
    query: np.ndarray,
    gallery: np.ndarray,
    labels: list[str],
    rng: np.random.Generator,
) -> dict[str, float]:
    shuffled_labels = labels.copy()
    rng.shuffle(shuffled_labels)
    return {
        "same_singer_retrieval_r1": recall_at_k(query, gallery, labels, labels, 1),
        "same_singer_retrieval_r5": recall_at_k(query, gallery, labels, labels, 5),
        "shuffled_speaker_label_retrieval_r1": recall_at_k(query, gallery, labels, shuffled_labels, 1),
        "shuffled_speaker_label_retrieval_r5": recall_at_k(query, gallery, labels, shuffled_labels, 5),
    }


def load_prompt_gaps(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for row in payload.get("pair_deltas", []):
        pair_id = str(row.get("target_pair_id", ""))
        if pair_id:
            out[pair_id] = float(row.get("singing_prompt_delta_to_target_singing", float("nan")))
    return out


def build_feature_cache(args: argparse.Namespace):
    cache: dict[tuple[str, str, str, str, str, bool], np.ndarray] = {}

    def cached(
        extractor: str,
        checkpoint_hash: str,
        layer: str,
        utt_id: str,
        voiced_only: bool,
    ) -> np.ndarray:
        key = (str(args.feature_root), extractor, checkpoint_hash, str(layer), utt_id, voiced_only)
        if key not in cache:
            cache[key] = load_feature_vector(
                args.feature_root,
                extractor,
                checkpoint_hash,
                str(layer),
                utt_id,
                voiced_only,
            )
        return cache[key]

    return cached


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    pairs = read_table(args.pairs)
    utterances = read_table(args.utterances)
    validate_utterances(utterances)
    validate_pairs(pairs, utterances)
    utt_by_id = {str(row["utt_id"]): row for row in utterances}
    prompt_gaps = load_prompt_gaps(args.seedvc_summary)
    cached = build_feature_cache(args)

    rows: list[dict[str, Any]] = []
    z_speech: list[np.ndarray] = []
    z_sing: list[np.ndarray] = []
    acoustic_delta: list[np.ndarray] = []
    skipped: list[str] = []
    for row in pairs:
        if str(row.get("pair_type")) != "same_singer_same_phrase":
            continue
        if not as_bool(row.get("same_text_flag")):
            continue
        speech_id = str(row["speech_utt_id"])
        sing_id = str(row["singing_utt_id"])
        speech = utt_by_id.get(speech_id)
        sing = utt_by_id.get(sing_id)
        if speech is None or sing is None:
            skipped.append(f"{row.get('pair_id')}: missing utterance row")
            continue
        try:
            speech_vec = cached(args.extractor, args.checkpoint_hash, args.layer, speech_id, args.voiced_only)
            sing_vec = cached(args.extractor, args.checkpoint_hash, args.layer, sing_id, args.voiced_only)
            if args.use_acoustic_covariates:
                speech_ac = cached("acoustic_baseline", "local_wave_v1", "frame25ms_hop20ms", speech_id, False)
                sing_ac = cached("acoustic_baseline", "local_wave_v1", "frame25ms_hop20ms", sing_id, False)
                ac_delta = sing_ac - speech_ac
            else:
                ac_delta = np.zeros(0, dtype=np.float64)
        except ExperimentError as exc:
            skipped.append(str(exc))
            continue
        phones = phone_tokens(speech.get("phone_seq", ""))
        speech_duration = max(as_float(speech, "duration_sec"), 1e-6)
        sing_duration = max(as_float(sing, "duration_sec"), 1e-6)
        speech_f0_range = as_float(speech, "f0_max_hz") - as_float(speech, "f0_min_hz")
        sing_f0_range = as_float(sing, "f0_max_hz") - as_float(sing, "f0_min_hz")
        item = {
            **row,
            "phone_tokens": phones,
            "phone_count": len(phones),
            "text_len_chars": len(str(speech.get("text", ""))),
            "speech_duration_sec": speech_duration,
            "singing_duration_sec": sing_duration,
            "delta_f0_mean": as_float(sing, "f0_mean_hz") - as_float(speech, "f0_mean_hz"),
            "delta_f0_std": as_float(sing, "f0_std_hz") - as_float(speech, "f0_std_hz"),
            "delta_f0_range": sing_f0_range - speech_f0_range,
            "delta_f0_voiced_pct": as_float(sing, "f0_voiced_pct") - as_float(speech, "f0_voiced_pct"),
            "duration_ratio": sing_duration / speech_duration,
            "delta_duration_sec": sing_duration - speech_duration,
            "speech_phone_rate": len(phones) / speech_duration,
            "singing_phone_rate": len(phone_tokens(sing.get("phone_seq", ""))) / sing_duration,
            "delta_energy_mean": as_float(sing, "energy_mean") - as_float(speech, "energy_mean"),
            "delta_energy_std": as_float(sing, "energy_std") - as_float(speech, "energy_std"),
            "delta_rms_db": as_float(sing, "rms_db") - as_float(speech, "rms_db"),
            "vocal_range": str(sing.get("vocal_range") or speech.get("vocal_range") or ""),
            "prompt_gap": prompt_gaps.get(str(row["pair_id"]), float("nan")),
        }
        rows.append(item)
        z_speech.append(speech_vec)
        z_sing.append(sing_vec)
        acoustic_delta.append(ac_delta)
        if args.max_pairs and len(rows) >= args.max_pairs:
            break

    if len(rows) < 10:
        detail = "; ".join(skipped[:5])
        raise ExperimentError(f"too few usable same-text pairs: {len(rows)}. {detail}")
    z_speech_arr = np.vstack(z_speech)
    z_sing_arr = np.vstack(z_sing)
    ac_delta_arr = np.vstack(acoustic_delta) if acoustic_delta and acoustic_delta[0].size else np.zeros((len(rows), 0))
    return {
        "rows": rows,
        "z_speech": z_speech_arr,
        "z_sing": z_sing_arr,
        "delta": z_sing_arr - z_speech_arr,
        "acoustic_delta": ac_delta_arr,
        "skipped": skipped,
        "prompt_gaps_loaded": len(prompt_gaps),
    }


def design_matrices(rows: list[dict[str, Any]], acoustic_delta: np.ndarray, seed: int) -> dict[str, dict[str, Any]]:
    phone_counts = Counter(token for row in rows for token in row["phone_tokens"])
    phone_vocab = sorted(token for token, count in phone_counts.items() if count >= 5)
    phone_index = {phone: i for i, phone in enumerate(phone_vocab)}
    phone_hist = np.zeros((len(rows), len(phone_vocab)), dtype=np.float64)
    for row_idx, row in enumerate(rows):
        for token in row["phone_tokens"]:
            if token in phone_index:
                phone_hist[row_idx, phone_index[token]] += 1.0

    content_numeric = np.asarray(
        [
            [as_bool(row.get("same_text_flag")), float(row["text_len_chars"]), float(row["phone_count"])]
            for row in rows
        ],
        dtype=np.float64,
    )
    content = np.hstack([content_numeric, phone_hist])

    prosody = np.asarray(
        [
            [row["delta_f0_mean"], row["delta_f0_std"], row["delta_f0_range"], row["delta_f0_voiced_pct"]]
            for row in rows
        ],
        dtype=np.float64,
    )
    timing_energy = np.asarray(
        [
            [
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
    technique, technique_names = one_hot(
        [str(row.get("technique", "")) for row in rows],
        sorted({str(row.get("technique", "")) for row in rows}),
    )
    metadata_lang, lang_names = one_hot(
        [str(row.get("language", "")) for row in rows],
        sorted({str(row.get("language", "")) for row in rows}),
    )
    metadata_range, range_names = one_hot(
        [str(row.get("vocal_range", "")) for row in rows],
        sorted({str(row.get("vocal_range", "")) for row in rows}),
    )
    metadata = np.hstack([metadata_lang, metadata_range])

    rng = np.random.default_rng(seed)
    shuffled_technique = technique[rng.permutation(len(rows))]

    m1 = content
    m2 = np.hstack([m1, prosody])
    m3 = np.hstack([m2, timing_energy])
    m4 = np.hstack([m3, acoustic_delta])
    m5 = np.hstack([m4, technique])
    return {
        "M1": {"x": m1, "description": "content/text proxies"},
        "M2": {"x": m2, "description": "M1 + F0/prosody deltas"},
        "M3": {"x": m3, "description": "M2 + timing/energy deltas"},
        "M4": {"x": m4, "description": "M3 + acoustic-baseline deltas"},
        "M5": {"x": m5, "description": "M4 + technique labels"},
        "C_f0_duration_energy_only": {
            "x": np.hstack([prosody, timing_energy]),
            "description": "negative control: F0/timing/energy only",
            "control": True,
        },
        "C_metadata_only": {
            "x": metadata,
            "description": "negative control: language/vocal-range metadata only",
            "control": True,
        },
        "C_m5_shuffle_technique": {
            "x": np.hstack([m4, shuffled_technique]),
            "description": "negative control: M5 with shuffled technique labels",
            "control": True,
        },
        "_meta": {
            "phone_vocab_size": len(phone_vocab),
            "technique_categories": technique_names,
            "language_categories": lang_names,
            "vocal_range_categories": range_names,
            "acoustic_covariate_dim": int(acoustic_delta.shape[1]),
        },
    }


def shuffled_singing_delta(rows: list[dict[str, Any]], z_sing: np.ndarray, z_speech: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = z_sing.copy()
    for speaker in sorted({str(row["speaker_id"]) for row in rows}):
        idx = np.asarray([i for i, row in enumerate(rows) if str(row["speaker_id"]) == speaker], dtype=int)
        if len(idx) > 1:
            out[idx] = out[rng.permutation(idx)]
    return out - z_speech


def collect_metrics(
    model_name: str,
    true_delta: np.ndarray,
    pred_delta: np.ndarray,
    z_speech: np.ndarray,
    z_sing: np.ndarray,
    labels: list[str],
    m0_mse: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    residual = true_delta - pred_delta
    delta_mse = float(np.mean(np.square(residual)))
    raw_norm = np.linalg.norm(true_delta, axis=1)
    residual_norm = np.linalg.norm(residual, axis=1)
    pred_sing = z_speech + pred_delta
    metrics = {
        "model": model_name,
        "test_items": int(len(true_delta)),
        "delta_cosine_mean": float(np.mean(row_cosine(pred_delta, true_delta))),
        "delta_mse": delta_mse,
        "delta_mse_reduction_vs_m0": 0.0 if model_name == "M0" else float(1.0 - delta_mse / max(m0_mse, 1e-12)),
        "raw_delta_norm_mean": float(np.mean(raw_norm)),
        "residual_norm_mean": float(np.mean(residual_norm)),
        "residual_norm_reduction_vs_raw": float(1.0 - np.mean(residual_norm) / max(float(np.mean(raw_norm)), 1e-12)),
        "cos_speech_plus_pred_to_singing_mean": float(np.mean(row_cosine(pred_sing, z_sing))),
        "cos_speech_to_singing_raw_mean": float(np.mean(row_cosine(z_speech, z_sing))),
    }
    metrics.update({f"before_{k}": v for k, v in retrieval_metrics(z_speech, z_sing, labels, rng).items()})
    metrics.update({f"after_{k}": v for k, v in retrieval_metrics(pred_sing, z_sing, labels, rng).items()})
    return metrics


def run_accounting(args: argparse.Namespace) -> dict[str, Any]:
    dataset = build_dataset(args)
    rows = dataset["rows"]
    z_speech = dataset["z_speech"]
    z_sing = dataset["z_sing"]
    delta = dataset["delta"]
    designs = design_matrices(rows, dataset["acoustic_delta"], args.seed)
    split_maps = (
        [deterministic_speaker_split(rows, args.seed)]
        if args.folds <= 1
        else kfold_speaker_splits(rows, args.folds, args.seed)
    )
    rng = np.random.default_rng(args.seed)
    all_model_names = MODEL_ORDER + ["C_f0_duration_energy_only", "C_metadata_only", "C_m5_shuffle_technique"]
    fold_alpha_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    pred_by_model: dict[str, dict[int, np.ndarray]] = {name: {} for name in all_model_names}
    true_by_model: dict[str, dict[int, np.ndarray]] = {name: {} for name in all_model_names}
    shuffled_delta = shuffled_singing_delta(rows, z_sing, z_speech, args.seed + 99)
    pred_by_model["C_shuffle_singing_within_speaker"] = {}
    true_by_model["C_shuffle_singing_within_speaker"] = {}

    for fold_idx, speaker_split in enumerate(split_maps):
        splits = np.asarray([speaker_split[str(row["speaker_id"])] for row in rows])
        assert_no_speaker_leakage(rows, splits)
        train_idx = np.where(splits == "train")[0]
        dev_idx = np.where(splits == "dev")[0]
        test_idx = np.where(splits == "test")[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        fold_labels = [str(rows[i]["speaker_id"]) for i in test_idx]
        print(
            f"fold {fold_idx}: train/dev/test pairs "
            f"{len(train_idx)}/{len(dev_idx)}/{len(test_idx)}; "
            f"speakers {len(set(rows[i]['speaker_id'] for i in train_idx))}/"
            f"{len(set(rows[i]['speaker_id'] for i in dev_idx))}/"
            f"{len(set(rows[i]['speaker_id'] for i in test_idx))}"
        )

        mean_delta = delta[train_idx].mean(axis=0, keepdims=True)
        fold_predictions: dict[str, np.ndarray] = {
            "M0": np.repeat(mean_delta, len(test_idx), axis=0)
        }
        fold_alphas: dict[str, float] = {"M0": 0.0}
        fold_dev_mse: dict[str, float] = {"M0": float("nan")}
        for model_name in all_model_names:
            if model_name == "M0":
                continue
            pred, alpha, dev_mse = fit_predict_model(designs[model_name]["x"], delta, train_idx, dev_idx, test_idx)
            fold_predictions[model_name] = pred
            fold_alphas[model_name] = alpha
            fold_dev_mse[model_name] = dev_mse
        pred, alpha, dev_mse = fit_predict_model(
            designs["M5"]["x"], shuffled_delta, train_idx, dev_idx, test_idx
        )
        fold_predictions["C_shuffle_singing_within_speaker"] = pred
        fold_alphas["C_shuffle_singing_within_speaker"] = alpha
        fold_dev_mse["C_shuffle_singing_within_speaker"] = dev_mse

        for model_name, pred in fold_predictions.items():
            true_target = shuffled_delta[test_idx] if model_name == "C_shuffle_singing_within_speaker" else delta[test_idx]
            for local_i, global_i in enumerate(test_idx):
                pred_by_model[model_name][int(global_i)] = pred[local_i]
                true_by_model[model_name][int(global_i)] = true_target[local_i]
            fold_alpha_rows.append(
                {
                    "fold": fold_idx,
                    "model": model_name,
                    "selected_alpha": float(fold_alphas[model_name]),
                    "dev_mse": float(fold_dev_mse[model_name]),
                    "train_items": int(len(train_idx)),
                    "dev_items": int(len(dev_idx)),
                    "test_items": int(len(test_idx)),
                }
            )
            residual = true_target - pred
            cos_delta = row_cosine(pred, true_target)
            cos_pred_sing = row_cosine(z_speech[test_idx] + pred, z_sing[test_idx])
            for local_i, global_i in enumerate(test_idx):
                row = rows[int(global_i)]
                predictions.append(
                    {
                        "run_id": args.run_id,
                        "fold": fold_idx,
                        "split": "test",
                        "model": model_name,
                        "pair_id": row["pair_id"],
                        "speech_utt_id": row["speech_utt_id"],
                        "singing_utt_id": row["singing_utt_id"],
                        "speaker_id": row["speaker_id"],
                        "language": row["language"],
                        "technique": row["technique"],
                        "prompt_gap": float(row["prompt_gap"]),
                        "true_delta_norm": float(np.linalg.norm(true_target[local_i])),
                        "pred_delta_norm": float(np.linalg.norm(pred[local_i])),
                        "residual_norm": float(np.linalg.norm(residual[local_i])),
                        "delta_cosine": float(cos_delta[local_i]),
                        "cos_speech_plus_pred_to_singing": float(cos_pred_sing[local_i]),
                    }
                )

    if not pred_by_model["M0"]:
        raise ExperimentError("no valid test predictions were produced")

    model_metrics: dict[str, dict[str, Any]] = {}
    m0_indices = sorted(pred_by_model["M0"])
    m0_true = np.vstack([true_by_model["M0"][idx] for idx in m0_indices])
    m0_pred = np.vstack([pred_by_model["M0"][idx] for idx in m0_indices])
    m0_mse = float(np.mean(np.square(m0_true - m0_pred)))
    for model_name in list(pred_by_model):
        indices = sorted(pred_by_model[model_name])
        if not indices:
            continue
        true_mat = np.vstack([true_by_model[model_name][idx] for idx in indices])
        pred_mat = np.vstack([pred_by_model[model_name][idx] for idx in indices])
        z_speech_mat = z_speech[indices]
        z_sing_mat = z_sing[indices]
        labels = [str(rows[idx]["speaker_id"]) for idx in indices]
        model_metrics[model_name] = collect_metrics(
            model_name, true_mat, pred_mat, z_speech_mat, z_sing_mat, labels, m0_mse, rng
        )
        if model_name in designs:
            model_metrics[model_name]["description"] = designs[model_name]["description"]
            model_metrics[model_name]["control"] = bool(designs[model_name].get("control", False))
            model_metrics[model_name]["num_covariates"] = int(designs[model_name]["x"].shape[1])
        elif model_name == "M0":
            model_metrics[model_name]["description"] = "global mean Delta_z only"
            model_metrics[model_name]["control"] = False
            model_metrics[model_name]["num_covariates"] = 0
        else:
            model_metrics[model_name]["description"] = "negative control: shuffled singing target within speaker"
            model_metrics[model_name]["control"] = True
            model_metrics[model_name]["num_covariates"] = int(designs["M5"]["x"].shape[1])

    increments = []
    for prev_name, next_name, label in [
        ("M1", "M2", "added F0/prosody"),
        ("M2", "M3", "added timing/energy"),
        ("M3", "M4", "added acoustic/phonation proxy"),
        ("M4", "M5", "added technique labels"),
    ]:
        if prev_name in model_metrics and next_name in model_metrics:
            prev = model_metrics[prev_name]
            nxt = model_metrics[next_name]
            increments.append(
                {
                    "increment": label,
                    "from_model": prev_name,
                    "to_model": next_name,
                    "delta_mse_change": float(prev["delta_mse"] - nxt["delta_mse"]),
                    "delta_mse_reduction_vs_previous": float(1.0 - nxt["delta_mse"] / max(prev["delta_mse"], 1e-12)),
                    "retrieval_r1_change": float(
                        nxt["after_same_singer_retrieval_r1"] - prev["after_same_singer_retrieval_r1"]
                    ),
                    "delta_cosine_change": float(nxt["delta_cosine_mean"] - prev["delta_cosine_mean"]),
                }
            )

    downstream = downstream_correlations(rows, delta, pred_by_model)
    metrics = {
        "run_id": args.run_id,
        "stage": "track1_prompt_mismatch_accounting",
        "extractor": args.extractor,
        "checkpoint_hash": args.checkpoint_hash,
        "layer_or_stream": args.layer,
        "representation": "mean_std",
        "voiced_only": bool(args.voiced_only),
        "pairs": len(rows),
        "speakers": len({str(row["speaker_id"]) for row in rows}),
        "techniques": dict(Counter(str(row["technique"]) for row in rows)),
        "languages": dict(Counter(str(row["language"]) for row in rows)),
        "folds_requested": int(args.folds),
        "folds_completed": len({row["fold"] for row in fold_alpha_rows}),
        "prompt_gaps_loaded": int(dataset["prompt_gaps_loaded"]),
        "prompt_gap_pairs_in_predictions": int(
            len({row["pair_id"] for row in predictions if math.isfinite(row["prompt_gap"])})
        ),
        "skipped_pairs": len(dataset["skipped"]),
        "skipped_examples": dataset["skipped"][:10],
        "design": designs["_meta"],
        "model_metrics": model_metrics,
        "factor_increments": increments,
        "downstream_correlations": downstream,
        "fold_fit": fold_alpha_rows,
        "seed": args.seed,
    }
    return {"metrics": metrics, "predictions": predictions}


def downstream_correlations(
    rows: list[dict[str, Any]],
    delta: np.ndarray,
    pred_by_model: dict[str, dict[int, np.ndarray]],
) -> dict[str, Any]:
    needed = ["M3", "M4", "M5"]
    xs: dict[str, list[float]] = {
        "raw_delta_norm": [],
        "m5_residual_norm": [],
        "m5_predicted_delta_norm": [],
        "technique_increment_norm_m5_minus_m4": [],
        "acoustic_increment_norm_m4_minus_m3": [],
    }
    y: list[float] = []
    pair_ids = []
    for idx, row in enumerate(rows):
        prompt_gap = float(row.get("prompt_gap", float("nan")))
        if not math.isfinite(prompt_gap):
            continue
        if any(idx not in pred_by_model.get(name, {}) for name in needed):
            continue
        m3 = pred_by_model["M3"][idx]
        m4 = pred_by_model["M4"][idx]
        m5 = pred_by_model["M5"][idx]
        xs["raw_delta_norm"].append(float(np.linalg.norm(delta[idx])))
        xs["m5_residual_norm"].append(float(np.linalg.norm(delta[idx] - m5)))
        xs["m5_predicted_delta_norm"].append(float(np.linalg.norm(m5)))
        xs["technique_increment_norm_m5_minus_m4"].append(float(np.linalg.norm(m5 - m4)))
        xs["acoustic_increment_norm_m4_minus_m3"].append(float(np.linalg.norm(m4 - m3)))
        y.append(prompt_gap)
        pair_ids.append(row["pair_id"])
    return {
        "prompt_gap_name": "Seed-VC singing_prompt_delta_to_target_singing",
        "pairs": pair_ids,
        "num_pairs": len(pair_ids),
        "correlations": {name: corr_summary(values, y) for name, values in xs.items()},
    }


def write_markdown_summary(metrics: dict[str, Any], path: Path) -> None:
    lines = [
        "# Track 1 Prompt-Mismatch Accounting",
        "",
        f"Run: `{metrics['run_id']}`",
        "",
        (
            f"Representation: `{metrics['extractor']}` / `{metrics['checkpoint_hash']}` / "
            f"`{metrics['layer_or_stream']}`"
        ),
        "",
        (
            f"Dataset: {metrics['pairs']} same-singer same-text pairs, "
            f"{metrics['speakers']} speakers, {metrics['folds_completed']} speaker-disjoint folds."
        ),
        "",
        "## Model Metrics",
        "",
        "| model | covariates | delta cosine | MSE red. vs M0 | residual norm red. | R@1 after | control |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    ordered = MODEL_ORDER + [
        "C_f0_duration_energy_only",
        "C_metadata_only",
        "C_m5_shuffle_technique",
        "C_shuffle_singing_within_speaker",
    ]
    for name in ordered:
        item = metrics["model_metrics"].get(name)
        if not item:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(item["num_covariates"]),
                    f"{item['delta_cosine_mean']:.4f}",
                    f"{item['delta_mse_reduction_vs_m0']:.4f}",
                    f"{item['residual_norm_reduction_vs_raw']:.4f}",
                    f"{item['after_same_singer_retrieval_r1']:.4f}",
                    str(bool(item["control"])).lower(),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Factor Increments", ""])
    if metrics["factor_increments"]:
        lines.extend(
            [
                "| increment | MSE change | MSE red. vs previous | R@1 change | delta cosine change |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for item in metrics["factor_increments"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        item["increment"],
                        f"{item['delta_mse_change']:.6f}",
                        f"{item['delta_mse_reduction_vs_previous']:.4f}",
                        f"{item['retrieval_r1_change']:.4f}",
                        f"{item['delta_cosine_change']:.4f}",
                    ]
                )
                + " |"
            )
    else:
        lines.append("No factor increments were available.")
    downstream = metrics["downstream_correlations"]
    lines.extend(
        [
            "",
            "## Seed-VC Prompt Gap Link",
            "",
            f"Matched Seed-VC prompt-gap pairs: {downstream['num_pairs']}",
            "",
            "| predictor | n | Pearson | Spearman |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, item in downstream["correlations"].items():
        lines.append(f"| {name} | {item['n']} | {item['pearson']:.4f} | {item['spearman']:.4f} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Values are out-of-fold over speaker-disjoint splits.",
            "- Technique is a label-associated covariate, not a causal factor.",
            "- Acoustic-baseline covariates are included as a shortcut/phonation proxy check.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Track 1 prompt-mode mismatch accounting.")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--utterances", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--extractor", required=True)
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--run-id", default="track1_prompt_mismatch_accounting")
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--predictions-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--seedvc-summary", type=Path)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--voiced-only", action="store_true")
    parser.add_argument("--no-acoustic-covariates", dest="use_acoustic_covariates", action="store_false")
    parser.set_defaults(use_acoustic_covariates=True)
    args = parser.parse_args()
    if args.extractor == "acoustic_baseline" and args.use_acoustic_covariates:
        print(
            "warning: disabling acoustic covariates for acoustic-baseline target to avoid target leakage",
            file=sys.stderr,
        )
        args.use_acoustic_covariates = False

    try:
        result = run_accounting(args)
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_json(result["metrics"], args.metrics_out)
    write_table(result["predictions"], args.predictions_out)
    write_markdown_summary(result["metrics"], args.summary_out)
    m5 = result["metrics"]["model_metrics"].get("M5", {})
    print(
        "M5 delta_cosine_mean="
        f"{m5.get('delta_cosine_mean', float('nan')):.4f}; "
        "M5 MSE reduction vs M0="
        f"{m5.get('delta_mse_reduction_vs_m0', float('nan')):.4f}"
    )
    print(f"wrote metrics -> {args.metrics_out}")
    print(f"wrote summary -> {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
