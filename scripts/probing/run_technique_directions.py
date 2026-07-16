#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_utils import (  # noqa: E402
    TECHNIQUE_PAIR_COLUMNS,
    ExperimentError,
    angular_degrees,
    cosine_similarity_matrix,
    load_feature_interval_vector,
    load_feature_vector,
    read_table,
    validate_required_columns,
    write_json,
    write_table,
)


def pair_feature_vectors(args: argparse.Namespace, row: dict) -> tuple[np.ndarray, np.ndarray]:
    if row.get("base_phone_start_sec") not in {None, ""}:
        base = load_feature_interval_vector(
            args.feature_root,
            args.extractor,
            args.checkpoint_hash,
            args.layer,
            row["base_utt_id"],
            float(row["base_phone_start_sec"]),
            float(row["base_phone_end_sec"]),
            args.voiced_only,
        )
        target = load_feature_interval_vector(
            args.feature_root,
            args.extractor,
            args.checkpoint_hash,
            args.layer,
            row["technique_utt_id"],
            float(row["technique_phone_start_sec"]),
            float(row["technique_phone_end_sec"]),
            args.voiced_only,
        )
    else:
        base = load_feature_vector(
            args.feature_root,
            args.extractor,
            args.checkpoint_hash,
            args.layer,
            row["base_utt_id"],
            args.voiced_only,
        )
        target = load_feature_vector(
            args.feature_root,
            args.extractor,
            args.checkpoint_hash,
            args.layer,
            row["technique_utt_id"],
            args.voiced_only,
        )
    return base, target


def mean_pairwise_cosine(x: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    sims = cosine_similarity_matrix(x, x)
    mask = ~np.eye(len(x), dtype=bool)
    return float(sims[mask].mean())


def cross_cosine_mean(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(cosine_similarity_matrix(a, b).mean())


def analogy_top1(
    base_vectors: np.ndarray,
    target_vectors: np.ndarray,
    delta_vectors: np.ndarray,
    group_indices: list[int],
    wrong_direction: np.ndarray | None,
    rng: np.random.Generator,
) -> dict[str, float]:
    if len(group_indices) < 2:
        return {
            "analogy_top1": float("nan"),
            "wrong_technique_top1": float("nan"),
            "shuffled_direction_top1": float("nan"),
            "analogy_evaluable": 0,
        }
    gallery = target_vectors[group_indices]
    correct = 0
    wrong_correct = 0
    shuffled_correct = 0
    shuffled_pool = np.asarray(
        [idx for idx in range(len(delta_vectors)) if idx not in set(group_indices)],
        dtype=int,
    )
    for local_idx, global_idx in enumerate(group_indices):
        train_indices = [idx for idx in group_indices if idx != global_idx]
        group_direction = delta_vectors[train_indices].mean(axis=0)
        query = base_vectors[global_idx] + group_direction
        sims = cosine_similarity_matrix(query[None, :], gallery)[0]
        correct += int(int(np.argmax(sims)) == local_idx)
        if wrong_direction is not None:
            wrong_query = base_vectors[global_idx] + wrong_direction
            wrong_sims = cosine_similarity_matrix(wrong_query[None, :], gallery)[0]
            wrong_correct += int(int(np.argmax(wrong_sims)) == local_idx)
        if len(shuffled_pool) > 0:
            sample_size = max(1, min(len(train_indices), len(shuffled_pool)))
            shuffled_direction = delta_vectors[rng.choice(shuffled_pool, size=sample_size, replace=False)].mean(axis=0)
            shuffled_query = base_vectors[global_idx] + shuffled_direction
            shuffled_sims = cosine_similarity_matrix(shuffled_query[None, :], gallery)[0]
            shuffled_correct += int(int(np.argmax(shuffled_sims)) == local_idx)
    return {
        "analogy_top1": correct / len(group_indices),
        "wrong_technique_top1": float("nan") if wrong_direction is None else wrong_correct / len(group_indices),
        "shuffled_direction_top1": float("nan") if len(shuffled_pool) == 0 else shuffled_correct / len(group_indices),
        "analogy_evaluable": len(group_indices),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate singing-technique directions from paired cached features.")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--extractor", required=True)
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--directions-out", type=Path, required=True)
    parser.add_argument("--run-id", default="technique_directions")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--voiced-only", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    try:
        pairs = read_table(args.pairs)
        validate_required_columns(pairs, TECHNIQUE_PAIR_COLUMNS, "technique pair manifest")
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        pair_records = []
        base_vectors = []
        target_vectors = []
        deltas_by_pair = []
        for pair_idx, row in enumerate(pairs):
            base, target = pair_feature_vectors(args, row)
            base_vectors.append(base)
            target_vectors.append(target)
            deltas_by_pair.append(target - base)
            record = {**row, "_pair_idx": pair_idx}
            grouped[(row["phone"], row["target_technique"])].append(record)
            pair_records.append(record)
        base_matrix = np.vstack(base_vectors)
        target_matrix = np.vstack(target_vectors)
        delta_matrix_all = np.vstack(deltas_by_pair)

        direction_rows = []
        metrics = {
            "run_id": args.run_id,
            "stage": "technique_stage_a",
            "extractor": args.extractor,
            "checkpoint_hash": args.checkpoint_hash,
            "layer_or_stream": args.layer,
            "bootstrap": args.bootstrap,
            "groups": [],
        }
        for (phone, target_technique), group in sorted(grouped.items()):
            group_indices = [int(row["_pair_idx"]) for row in group]
            delta_mat = delta_matrix_all[group_indices]
            direction = delta_mat.mean(axis=0)
            other_same_phone_indices = [
                idx
                for idx, row in enumerate(pairs)
                if row["phone"] == phone and row["target_technique"] != target_technique
            ]
            other_tech_indices = [
                idx for idx, row in enumerate(pairs) if row["target_technique"] != target_technique
            ]
            between_source = other_same_phone_indices or other_tech_indices
            between_mean = cross_cosine_mean(delta_mat, delta_matrix_all[between_source]) if between_source else float("nan")
            wrong_direction = None
            if between_source:
                wrong_direction = delta_matrix_all[between_source].mean(axis=0)
            analogy = analogy_top1(base_matrix, target_matrix, delta_matrix_all, group_indices, wrong_direction, rng)
            angles = []
            for _ in range(args.bootstrap):
                sample_idx = rng.choice(len(delta_mat), size=max(1, int(0.8 * len(delta_mat))), replace=True)
                sample_dir = delta_mat[sample_idx].mean(axis=0)
                angles.append(angular_degrees(direction, sample_dir))
            angle_arr = np.asarray(angles, dtype=np.float64)
            group_metrics = {
                "phone": phone,
                "target_technique": target_technique,
                "num_pairs": len(group),
                "num_speakers": len({row["speaker_id"] for row in group}),
                "num_languages": len({row["language"] for row in group}),
                "direction_norm": float(np.linalg.norm(direction)),
                "within_delta_cosine_mean": mean_pairwise_cosine(delta_mat),
                "between_delta_cosine_mean": between_mean,
                "analogy_top1": analogy["analogy_top1"],
                "wrong_technique_top1": analogy["wrong_technique_top1"],
                "shuffled_direction_top1": analogy["shuffled_direction_top1"],
                "analogy_evaluable": analogy["analogy_evaluable"],
                "bootstrap_angle_mean_deg": float(np.nanmean(angle_arr)),
                "bootstrap_angle_std_deg": float(np.nanstd(angle_arr)),
            }
            metrics["groups"].append(group_metrics)
            for i, value in enumerate(direction):
                direction_rows.append(
                    {
                        "run_id": args.run_id,
                        "phone": phone,
                        "target_technique": target_technique,
                        "dim": i,
                        "value": float(value),
                    }
                )
            print(f"{phone}/{target_technique} direction shape: {direction.shape}")
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    write_json(metrics, args.metrics_out)
    write_table(direction_rows, args.directions_out)
    print(f"wrote metrics -> {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
