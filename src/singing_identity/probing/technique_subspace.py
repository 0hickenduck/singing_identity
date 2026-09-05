#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from singing_identity.utils.research_utils import (  # noqa: E402
    ExperimentError,
    angular_degrees,
    kfold_speaker_splits,
    load_feature_interval_vector,
    read_table,
    write_json,
)


SPECIAL_PHONES = {"<AP>", "<SP>", "<SIL>", "<sil>", "sil", "sp", ""}


def row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    return np.sum(a * b, axis=1) / denom


def orthonormal_basis(x: np.ndarray, k: int) -> np.ndarray:
    if len(x) == 0:
        return np.zeros((0, 0), dtype=np.float64)
    _, _, vt = np.linalg.svd(np.asarray(x, dtype=np.float64), full_matrices=False)
    return vt[: min(k, vt.shape[0])]


def random_basis(dim: int, k: int, rng: np.random.Generator) -> np.ndarray:
    mat = rng.normal(size=(dim, k))
    q, _ = np.linalg.qr(mat)
    return q[:, :k].T


def projection_metrics(delta: np.ndarray, basis: np.ndarray) -> dict[str, float]:
    if basis.size == 0:
        return {
            "capture_mean": float("nan"),
            "capture_std": float("nan"),
            "projection_cosine_mean": float("nan"),
            "projection_norm_ratio_mean": float("nan"),
        }
    proj = (delta @ basis.T) @ basis
    denom = np.maximum(np.sum(delta * delta, axis=1), 1e-12)
    capture = np.sum(proj * proj, axis=1) / denom
    return {
        "capture_mean": float(np.mean(capture)),
        "capture_std": float(np.std(capture)),
        "projection_cosine_mean": float(np.mean(row_cosine(proj, delta))),
        "projection_norm_ratio_mean": float(np.mean(np.linalg.norm(proj, axis=1) / np.maximum(np.linalg.norm(delta, axis=1), 1e-12))),
    }


def load_delta(args: argparse.Namespace, row: dict[str, Any]) -> np.ndarray:
    base = load_feature_interval_vector(
        args.feature_root,
        args.extractor,
        args.checkpoint_hash,
        args.layer,
        str(row["base_utt_id"]),
        float(row["base_phone_start_sec"]),
        float(row["base_phone_end_sec"]),
        args.voiced_only,
    )
    target = load_feature_interval_vector(
        args.feature_root,
        args.extractor,
        args.checkpoint_hash,
        args.layer,
        str(row["technique_utt_id"]),
        float(row["technique_phone_start_sec"]),
        float(row["technique_phone_end_sec"]),
        args.voiced_only,
    )
    return target - base


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], np.ndarray]:
    all_rows = read_table(args.pairs)
    rows = []
    deltas = []
    cache: dict[str, np.ndarray] = {}
    for row in all_rows:
        phone = str(row.get("phone", ""))
        if phone in SPECIAL_PHONES or phone.startswith("<"):
            continue
        tech = str(row.get("target_technique", ""))
        if tech != args.technique and not args.include_controls:
            continue
        key = str(row.get("pair_id", len(rows)))
        if key not in cache:
            cache[key] = load_delta(args, row)
        rows.append({**row, "_idx": len(rows)})
        deltas.append(cache[key])
        if args.max_rows and len(rows) >= args.max_rows:
            break
    if len(rows) < 20:
        raise ExperimentError(f"too few rows: {len(rows)}")
    return rows, np.vstack(deltas)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows, delta = build_rows(args)
    target_indices = [i for i, row in enumerate(rows) if str(row["target_technique"]) == args.technique]
    target_rows = [rows[i] for i in target_indices]
    if len(target_rows) < 20:
        raise ExperimentError(f"too few target-technique rows: {len(target_rows)}")
    split_maps = kfold_speaker_splits(target_rows, args.folds, args.seed)
    rng = np.random.default_rng(args.seed)
    rows_by_pair_idx = {int(row["_idx"]): row for row in rows}
    fold_rows = []
    for fold_idx, split_map in enumerate(split_maps):
        target_train_idx = []
        target_test_idx = []
        train_speakers = set()
        test_speakers = set()
        for global_idx in target_indices:
            row = rows_by_pair_idx[global_idx]
            split = split_map[str(row["speaker_id"])]
            if split == "train":
                target_train_idx.append(global_idx)
                train_speakers.add(str(row["speaker_id"]))
            elif split == "test":
                target_test_idx.append(global_idx)
                test_speakers.add(str(row["speaker_id"]))
        if not target_train_idx or not target_test_idx:
            continue
        wrong_train_idx = [
            i
            for i, row in enumerate(rows)
            if str(row["target_technique"]) != args.technique and str(row["speaker_id"]) in train_speakers
        ]
        train_delta = delta[target_train_idx]
        test_delta = delta[target_test_idx]
        wrong_delta = delta[wrong_train_idx] if wrong_train_idx else np.empty((0, delta.shape[1]), dtype=np.float64)
        mean_direction = train_delta.mean(axis=0, keepdims=True)
        mean_angle = float(np.mean([angular_degrees(mean_direction[0], d) for d in test_delta]))
        for k in args.components:
            target_basis = orthonormal_basis(train_delta, k)
            wrong_basis = orthonormal_basis(wrong_delta, k) if len(wrong_delta) >= k else np.zeros((0, 0), dtype=np.float64)
            coord_permuted_train = train_delta[:, rng.permutation(train_delta.shape[1])]
            coord_permuted_basis = orthonormal_basis(coord_permuted_train, k)
            rand_basis = random_basis(delta.shape[1], min(k, delta.shape[1]), rng)
            for basis_name, basis in [
                ("target_breathy_subspace", target_basis),
                ("wrong_technique_subspace", wrong_basis),
                ("coordinate_permuted_subspace", coord_permuted_basis),
                ("random_gaussian_subspace", rand_basis),
            ]:
                metrics = projection_metrics(test_delta, basis)
                fold_rows.append(
                    {
                        "fold": int(fold_idx),
                        "components": int(k),
                        "basis": basis_name,
                        "train_items": int(len(target_train_idx)),
                        "test_items": int(len(target_test_idx)),
                        "wrong_train_items": int(len(wrong_train_idx)),
                        "train_speakers": int(len(train_speakers)),
                        "test_speakers": int(len(test_speakers)),
                        "mean_direction_angle_to_test_delta_deg": mean_angle,
                        **metrics,
                    }
                )
    if not fold_rows:
        raise ExperimentError("no valid folds")
    aggregate = {}
    for k in args.components:
        aggregate[str(k)] = {}
        for basis_name in sorted({row["basis"] for row in fold_rows}):
            selected = [row for row in fold_rows if row["components"] == k and row["basis"] == basis_name]
            aggregate[str(k)][basis_name] = {
                "folds": len(selected),
                "capture_mean": float(np.nanmean([row["capture_mean"] for row in selected])),
                "projection_cosine_mean": float(np.nanmean([row["projection_cosine_mean"] for row in selected])),
                "projection_norm_ratio_mean": float(np.nanmean([row["projection_norm_ratio_mean"] for row in selected])),
            }
    return {
        "run_id": args.run_id,
        "stage": "technique_subspace_probe",
        "extractor": args.extractor,
        "checkpoint_hash": args.checkpoint_hash,
        "layer_or_stream": args.layer,
        "technique": args.technique,
        "rows": len(rows),
        "target_rows": len(target_indices),
        "speakers": len({str(rows[i]["speaker_id"]) for i in target_indices}),
        "phones": len({str(rows[i]["phone"]) for i in target_indices}),
        "components": [int(k) for k in args.components],
        "aggregate": aggregate,
        "fold_metrics": fold_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe whether technique deltas form a stable cross-speaker subspace.")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--extractor", required=True)
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--run-id", default="technique_subspace_probe")
    parser.add_argument("--technique", default="breathy")
    parser.add_argument("--components", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--voiced-only", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args)
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_json(result, args.metrics_out)
    for k, block in result["aggregate"].items():
        target = block.get("target_breathy_subspace", {})
        wrong = block.get("wrong_technique_subspace", {})
        random = block.get("random_gaussian_subspace", {})
        print(
            f"k={k} target_capture={target.get('capture_mean', float('nan')):.4f} "
            f"wrong={wrong.get('capture_mean', float('nan')):.4f} "
            f"random={random.get('capture_mean', float('nan')):.4f}"
        )
    print(f"wrote metrics -> {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
