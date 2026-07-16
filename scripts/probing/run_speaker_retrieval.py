#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_utils import (  # noqa: E402
    ExperimentError,
    apply_nuisance_residualizer,
    assert_no_speaker_leakage,
    build_matrix,
    deterministic_speaker_split,
    fit_nuisance_residualizer,
    group_rows,
    manifest_summary,
    mean_average_precision,
    raw_nuisance_matrix,
    read_table,
    recall_at_k,
    validate_utterances,
    write_json,
)


def centroids(rows: list[dict], x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    groups = group_rows([{**row, "_idx": i} for i, row in enumerate(rows)], "speaker_id")
    labels = []
    values = []
    for speaker_id, speaker_rows in sorted(groups.items()):
        idx = [int(row["_idx"]) for row in speaker_rows]
        labels.append(speaker_id)
        values.append(x[idx].mean(axis=0))
    return np.vstack(values), labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-mode same-person retrieval from cached features.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--extractor", required=True)
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--run-id", default="speaker_retrieval")
    parser.add_argument("--voiced-only", action="store_true")
    parser.add_argument("--residualize-nuisance", action="store_true",
                        help="Remove F0/energy/duration/voiced-rate nuisance from features before retrieval.")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    try:
        rows = read_table(args.manifest)
        validate_utterances(rows)
        speech_rows = [row for row in rows if row["mode"] == "speech"]
        singing_rows = [row for row in rows if row["mode"] in {"singing", "singing_control", "singing_technique"}]
        if not speech_rows or not singing_rows:
            raise ExperimentError("retrieval requires speech and singing rows")
        x_speech, _ = build_matrix(speech_rows, args.feature_root, args.extractor, args.checkpoint_hash, args.layer, args.voiced_only)
        x_sing, _ = build_matrix(singing_rows, args.feature_root, args.extractor, args.checkpoint_hash, args.layer, args.voiced_only)

        controls_label = "none"
        split_map = deterministic_speaker_split(rows, args.seed)
        all_rows = speech_rows + singing_rows
        all_splits = np.asarray([split_map[str(row["speaker_id"])] for row in all_rows])
        assert_no_speaker_leakage(all_rows, all_splits)
        split_counts = {split: int(np.sum(all_splits == split)) for split in sorted(set(all_splits))}
        speaker_split_counts = {
            split: len({str(row["speaker_id"]) for row, row_split in zip(all_rows, all_splits) if row_split == split})
            for split in sorted(set(all_splits))
        }
        if args.residualize_nuisance:
            controls_label = "f0_energy_duration_voiced_rms"
            x_all = np.vstack([x_speech, x_sing])
            train_idx = np.flatnonzero(all_splits == "train")
            if len(train_idx) == 0:
                raise ExperimentError("residualization requires at least one train row")
            nuisance_all = raw_nuisance_matrix(all_rows)
            residualizer = fit_nuisance_residualizer(x_all[train_idx], nuisance_all[train_idx])
            nuisance_speech = raw_nuisance_matrix(speech_rows)
            nuisance_sing = raw_nuisance_matrix(singing_rows)
            x_speech = apply_nuisance_residualizer(x_speech, nuisance_speech, residualizer)
            x_sing = apply_nuisance_residualizer(x_sing, nuisance_sing, residualizer)
            print(
                "residualized features: "
                f"controls={controls_label}; train speakers={speaker_split_counts.get('train', 0)}"
            )

        speech_cents, speech_labels = centroids(speech_rows, x_speech)
        sing_cents, sing_labels = centroids(singing_rows, x_sing)
        print(f"speech centroid shape: {speech_cents.shape}")
        print(f"singing centroid shape: {sing_cents.shape}")

        num_speakers = len(set(speech_labels) & set(sing_labels))
        chance = 1.0 / max(num_speakers, 1)
        s2s_r1 = recall_at_k(speech_cents, sing_cents, speech_labels, sing_labels, 1)
        s2s_r5 = recall_at_k(speech_cents, sing_cents, speech_labels, sing_labels, 5)
        s2sp_r1 = recall_at_k(sing_cents, speech_cents, sing_labels, speech_labels, 1)
        s2sp_r5 = recall_at_k(sing_cents, speech_cents, sing_labels, speech_labels, 5)
        s2s_map = mean_average_precision(speech_cents, sing_cents, speech_labels, sing_labels)
        s2sp_map = mean_average_precision(sing_cents, speech_cents, sing_labels, speech_labels)

        metrics = {
            "run_id": args.run_id,
            "stage": "stage_a",
            "manifest_hash": manifest_summary(rows)["manifest_hash"],
            "split_name": "all_speaker_centroids_train_fit_residualizer",
            "seed": args.seed,
            "extractor": args.extractor,
            "checkpoint_hash": args.checkpoint_hash,
            "layer_or_stream": args.layer,
            "representation": "mean_std_centroid",
            "target": "speaker_id",
            "probe_type": "cosine_retrieval",
            "controls": controls_label,
            "speech_to_singing_recall_at_1": s2s_r1,
            "speech_to_singing_recall_at_5": s2s_r5,
            "speech_to_singing_map": s2s_map,
            "singing_to_speech_recall_at_1": s2sp_r1,
            "singing_to_speech_recall_at_5": s2sp_r5,
            "singing_to_speech_map": s2sp_map,
            "num_speakers": num_speakers,
            "chance_recall_at_1": chance,
            "speech_to_singing_recall_at_1_norm": s2s_r1 / chance if chance > 0 else 0.0,
            "singing_to_speech_recall_at_1_norm": s2sp_r1 / chance if chance > 0 else 0.0,
            "train_only_residualization": bool(args.residualize_nuisance),
            "row_split_counts": split_counts,
            "speaker_split_counts": speaker_split_counts,
        }
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_json(metrics, args.metrics_out)
    print(f"R@1 speech->singing: {metrics['speech_to_singing_recall_at_1']:.4f} "
          f"({metrics['speech_to_singing_recall_at_1_norm']:.1f}× chance)")
    print(f"R@1 singing->speech: {metrics['singing_to_speech_recall_at_1']:.4f} "
          f"({metrics['singing_to_speech_recall_at_1_norm']:.1f}× chance)")
    print(f"controls: {controls_label}")
    print(f"wrote metrics -> {args.metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
