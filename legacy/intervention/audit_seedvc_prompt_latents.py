#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_utils import ExperimentError, write_json, write_table  # noqa: E402


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.column_stack([np.ones(x.shape[0]), x])
    ident = np.eye(x_aug.shape[1], dtype=np.float64)
    ident[0, 0] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + alpha * ident, x_aug.T @ y)


def ridge_predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(x.shape[0]), x]) @ weights


def component_vector(npz_path: str, component: str) -> np.ndarray:
    with np.load(npz_path) as data:
        if component == "pooled":
            return np.asarray(data["pooled_vector"], dtype=np.float64).reshape(-1)
        if component == "style":
            return np.asarray(data["style"], dtype=np.float64).reshape(-1)
        if component == "prompt_stats":
            return np.concatenate(
                [
                    np.asarray(data["prompt_mean"], dtype=np.float64).reshape(-1),
                    np.asarray(data["prompt_std"], dtype=np.float64).reshape(-1),
                ]
            )
        if component == "semantic_stats":
            return np.concatenate(
                [
                    np.asarray(data["semantic_mean"], dtype=np.float64).reshape(-1),
                    np.asarray(data["semantic_std"], dtype=np.float64).reshape(-1),
                ]
            )
        if component == "mel_stats":
            return np.concatenate(
                [
                    np.asarray(data["mel_mean"], dtype=np.float64).reshape(-1),
                    np.asarray(data["mel_std"], dtype=np.float64).reshape(-1),
                ]
            )
        raise ExperimentError(f"Unknown component: {component}")


def corr(xs: list[float], ys: list[float]) -> float:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit speech-to-singing residuals in Seed-VC native prompt latents.")
    parser.add_argument("--latent-manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--component", choices=["pooled", "style", "prompt_stats", "semantic_stats", "mel_stats"], default="pooled")
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    args = parser.parse_args()

    try:
        rows = load_rows(args.latent_manifest)
        by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in rows:
            by_pair[str(row["pair_id"])][str(row["condition"])] = row
        pairs = []
        for pair_id, conds in sorted(by_pair.items()):
            if "speech_prompt" in conds and "singing_prompt" in conds:
                speech = conds["speech_prompt"]
                singing = conds["singing_prompt"]
                x_speech = component_vector(str(speech["latent_npz"]), args.component)
                x_singing = component_vector(str(singing["latent_npz"]), args.component)
                pairs.append(
                    {
                        "pair_id": pair_id,
                        "speaker_id": speech["target_speaker_id"],
                        "language": speech["language"],
                        "speech": speech,
                        "singing": singing,
                        "x_speech": x_speech,
                        "x_singing": x_singing,
                        "delta": x_singing - x_speech,
                    }
                )
        if len(pairs) < 4:
            raise ExperimentError(f"Need at least 4 paired latents, found {len(pairs)}")

        speakers = sorted({str(row["speaker_id"]) for row in pairs})
        heldout = set(speakers[-max(1, len(speakers) // 5):])
        train = [row for row in pairs if str(row["speaker_id"]) not in heldout]
        test = [row for row in pairs if str(row["speaker_id"]) in heldout]
        if not train or not test:
            train = pairs[: max(1, int(len(pairs) * 0.8))]
            test = pairs[len(train):]
        if not test:
            test = train

        global_delta = np.mean([row["delta"] for row in train], axis=0)
        x_train = np.stack([row["x_speech"] for row in train])
        y_train = np.stack([row["delta"] for row in train])
        weights = ridge_fit(x_train, y_train, args.ridge_alpha)

        detail_rows = []
        for split_name, split_rows in (("train", train), ("test", test)):
            for row in split_rows:
                speech = row["x_speech"]
                singing = row["x_singing"]
                mapped_global = speech + global_delta
                mapped_ridge = speech + ridge_predict(speech[None, :], weights)[0]
                true_delta = row["delta"]
                nuisance = {
                    "duration_delta": float(row["singing"]["duration_sec"]) - float(row["speech"]["duration_sec"]),
                    "rms_delta": float(row["singing"]["rms"]) - float(row["speech"]["rms"]),
                    "f0_voiced_pct_delta": float(row["singing"]["f0_voiced_pct"]) - float(row["speech"]["f0_voiced_pct"]),
                }
                detail_rows.append(
                    {
                        "pair_id": row["pair_id"],
                        "speaker_id": row["speaker_id"],
                        "language": row["language"],
                        "split": split_name,
                        "component": args.component,
                        "speech_to_singing_cosine": cosine(speech, singing),
                        "global_mapped_to_singing_cosine": cosine(mapped_global, singing),
                        "ridge_mapped_to_singing_cosine": cosine(mapped_ridge, singing),
                        "global_delta_to_true_delta_cosine": cosine(global_delta, true_delta),
                        "ridge_delta_to_true_delta_cosine": cosine(mapped_ridge - speech, true_delta),
                        "true_delta_norm": float(np.linalg.norm(true_delta)),
                        "global_delta_norm": float(np.linalg.norm(global_delta)),
                        "ridge_delta_norm": float(np.linalg.norm(mapped_ridge - speech)),
                        **nuisance,
                    }
                )

        def mean_metric(split: str, key: str) -> float:
            vals = [float(row[key]) for row in detail_rows if row["split"] == split and np.isfinite(float(row[key]))]
            return float(np.mean(vals)) if vals else float("nan")

        test_rows = [row for row in detail_rows if row["split"] == "test"]
        summary = {
            "stage": "seedvc_native_prompt_latent_audit",
            "component": args.component,
            "pairs": len(pairs),
            "train_pairs": len(train),
            "test_pairs": len(test),
            "speakers": len(speakers),
            "heldout_speakers": sorted(heldout),
            "vector_dim": int(pairs[0]["x_speech"].shape[0]),
            "ridge_alpha": args.ridge_alpha,
            "test_speech_to_singing_cosine_mean": mean_metric("test", "speech_to_singing_cosine"),
            "test_global_mapped_to_singing_cosine_mean": mean_metric("test", "global_mapped_to_singing_cosine"),
            "test_ridge_mapped_to_singing_cosine_mean": mean_metric("test", "ridge_mapped_to_singing_cosine"),
            "test_global_delta_to_true_delta_cosine_mean": mean_metric("test", "global_delta_to_true_delta_cosine"),
            "test_ridge_delta_to_true_delta_cosine_mean": mean_metric("test", "ridge_delta_to_true_delta_cosine"),
            "test_delta_norm_mean": mean_metric("test", "true_delta_norm"),
            "nuisance_correlations_test": {
                "delta_norm_vs_duration_delta": corr(
                    [row["true_delta_norm"] for row in test_rows],
                    [row["duration_delta"] for row in test_rows],
                ),
                "delta_norm_vs_rms_delta": corr(
                    [row["true_delta_norm"] for row in test_rows],
                    [row["rms_delta"] for row in test_rows],
                ),
                "delta_norm_vs_f0_voiced_pct_delta": corr(
                    [row["true_delta_norm"] for row in test_rows],
                    [row["f0_voiced_pct_delta"] for row in test_rows],
                ),
            },
        }
    except (ExperimentError, OSError, RuntimeError, np.linalg.LinAlgError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.out_root.mkdir(parents=True, exist_ok=True)
    write_table(detail_rows, args.out_root / f"{args.component}_pair_metrics.jsonl")
    write_json(summary, args.out_root / f"{args.component}_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
