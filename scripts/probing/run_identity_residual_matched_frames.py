#!/usr/bin/env python
"""Fixed-frame, voiced-ratio-matched control for the GTSinger same-text subset."""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probing.run_identity_residual_final_validation import mode_probe_centroid  # noqa: E402
from probing.run_identity_residual_robustness import (  # noqa: E402
    MODELS,
    cosine,
    cosine_matrix,
    paired_deltas,
    random_means,
    ranks_from_scores,
    roc_det_points,
    split_indices,
    summary,
    verification_metrics,
    write_roc_det_plots,
)
from probing.run_identity_residual_same_text import (  # noqa: E402
    GTSINGER_SEEDS as SEEDS,
    select_clean_control_pairs,
)
from probing.run_identity_residual_suite import (  # noqa: E402
    MODEL_SPECS,
    feature_path,
    read_jsonl,
    write_csv,
    write_yaml,
)
from research_utils import ExperimentError, current_git_commit  # noqa: E402


PAIRS = Path("/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local/manifests/gtsinger_pairs.jsonl")
UTTERANCES = Path("/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local/manifests/gtsinger_utterances.jsonl")
FEATURE_ROOT = Path("/localdisk/bowen/singing_identity/features/stage1_repaired_200_fresh_local")


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


def load_frame_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise ExperimentError(f"missing feature file: {path}")
    with np.load(path) as data:
        x = np.asarray(data["x"], dtype=np.float64)
        voiced = np.asarray(data["voiced_mask"], dtype=bool) if "voiced_mask" in data.files else np.ones(len(x), dtype=bool)
    if x.ndim != 2 or len(x) == 0 or len(voiced) != len(x):
        raise ExperimentError(f"invalid frame feature: {path}")
    return x, voiced


def load_matched_vectors(
    pairs: list[dict[str, Any]],
    model: str,
    fixed_frames: int,
    crops_per_pair: int,
    candidate_count: int,
    cache_dir: Path,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]], list[dict[str, Any]], str]:
    extractor, checkpoint, layer = MODEL_SPECS[model]
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{model}_frames{fixed_frames}_crops{crops_per_pair}_candidates{candidate_count}_{len(pairs)}"
    array_path = cache_dir / f"{prefix}.npz"
    audit_path = cache_dir / f"{prefix}_audit.json"
    pair_ids = [str(pair["pair_id"]) for pair in pairs]
    if array_path.exists() and audit_path.exists():
        with np.load(array_path) as data:
            saved_ids = json.loads(str(data["pair_ids_json"].item()))
            if saved_ids == pair_ids:
                speakers = json.loads(str(data["speakers_json"].item()))
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                return (
                    {"speech": {s: data[f"matched_speech_{i}"] for i, s in enumerate(speakers)}, "singing": {s: data[f"matched_singing_{i}"] for i, s in enumerate(speakers)}},
                    {"speech": {s: data[f"full_speech_{i}"] for i, s in enumerate(speakers)}, "singing": {s: data[f"full_singing_{i}"] for i, s in enumerate(speakers)}},
                    audit,
                    str(layer),
                )
    matched: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    full: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    audit: list[dict[str, Any]] = []
    for pair in pairs:
        pair_id = str(pair["pair_id"])
        speaker = str(pair["speaker_id"])
        s_path = feature_path(FEATURE_ROOT, extractor, checkpoint, str(layer), str(pair["speech_utt_id"]))
        g_path = feature_path(FEATURE_ROOT, extractor, checkpoint, str(layer), str(pair["singing_utt_id"]))
        speech_x, speech_v = load_frame_file(s_path)
        singing_x, singing_v = load_frame_file(g_path)
        full[speaker]["speech"].append(frame_vector(speech_x))
        full[speaker]["singing"].append(frame_vector(singing_x))
        pair_rows = []
        for crop_draw in range(crops_per_pair):
            seed = zlib.crc32(f"{model}|{pair_id}|{crop_draw}".encode("utf-8"))
            selected = choose_matched_crops(speech_x, speech_v, singing_x, singing_v, fixed_frames, candidate_count, np.random.default_rng(seed))
            if selected is None:
                continue
            s_crop, g_crop, info = selected
            matched[speaker]["speech"].append(frame_vector(s_crop))
            matched[speaker]["singing"].append(frame_vector(g_crop))
            pair_rows.append(info)
        audit.append({
            "pair_id": pair_id, "speaker_id": speaker, "model": model, "layer": layer,
            "speech_frames": len(speech_x), "singing_frames": len(singing_x), "fixed_frames": fixed_frames,
            "requested_crops": crops_per_pair, "accepted_crops": len(pair_rows),
            "status": "accepted" if pair_rows else "excluded_too_short_for_fixed_frames",
            "mean_voiced_ratio_abs_difference": float(np.mean([row["voiced_ratio_abs_difference"] for row in pair_rows])) if pair_rows else float("nan"),
        })
    speakers = sorted(s for s in matched if matched[s]["speech"] and matched[s]["singing"])
    if len(speakers) != 20:
        raise ExperimentError(f"fixed-frame selection retained {len(speakers)} speakers, expected 20")
    matched_out = {mode: {s: np.vstack(matched[s][mode]) for s in speakers} for mode in ["speech", "singing"]}
    full_out = {mode: {s: np.vstack(full[s][mode]) for s in speakers} for mode in ["speech", "singing"]}
    payload: dict[str, Any] = {"pair_ids_json": np.asarray(json.dumps(pair_ids)), "speakers_json": np.asarray(json.dumps(speakers))}
    for i, speaker in enumerate(speakers):
        payload[f"matched_speech_{i}"] = matched_out["speech"][speaker]
        payload[f"matched_singing_{i}"] = matched_out["singing"][speaker]
        payload[f"full_speech_{i}"] = full_out["speech"][speaker]
        payload[f"full_singing_{i}"] = full_out["singing"][speaker]
    np.savez_compressed(array_path, **payload)
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=True), encoding="utf-8")
    return matched_out, full_out, audit, str(layer)


def speaker_centroids(vectors: dict[str, dict[str, np.ndarray]], speakers: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.vstack([vectors["speech"][speaker].mean(axis=0) for speaker in speakers]),
        np.vstack([vectors["singing"][speaker].mean(axis=0) for speaker in speakers]),
    )


def run(args: argparse.Namespace) -> None:
    pairs_all = read_jsonl(args.pairs)
    utterances = {str(row["utt_id"]): row for row in read_jsonl(args.utterances)}
    pairs, selection_audit = select_clean_control_pairs(pairs_all, utterances)
    args.run_root.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    verification: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    mode_rows: list[dict[str, Any]] = []
    cosine_rows: list[dict[str, Any]] = []
    crop_audit: list[dict[str, Any]] = []
    for model in args.models:
        matched, full, audit, layer = load_matched_vectors(pairs, model, args.fixed_frames, args.crops_per_pair, args.candidate_count, args.run_root / model / "cache")
        crop_audit.extend(audit)
        speakers = sorted(matched["speech"])
        speech, singing = speaker_centroids(matched, speakers)
        full_speech, full_singing = speaker_centroids(full, speakers)
        for seed in args.seeds:
            split, train, test = split_indices(speakers, seed, "gtsinger_10_0_10")
            mu = (singing[train] - speech[train]).mean(axis=0)
            original_mu = (full_singing[train] - full_speech[train]).mean(axis=0)
            cosine_rows.append({
                "dataset": "GTSinger_same_text_control_fixed_frames", "model": model, "layer": layer, "split_seed": seed,
                "global_vector_cosine_to_full_utterance_global": cosine(mu, original_mu),
                "matched_global_norm": float(np.linalg.norm(mu)), "full_global_norm": float(np.linalg.norm(original_mu)),
                "train_speakers": ";".join(speakers[i] for i in train),
            })
            variants: list[tuple[str, np.ndarray, np.ndarray, int | str]] = [
                ("raw", speech[test], speech[train], ""),
                ("correct_sign_global", speech[test] + mu, speech[train] + mu, ""),
                ("wrong_sign_global", speech[test] - mu, speech[train] - mu, ""),
            ]
            rng = np.random.default_rng(seed * 1009 + len(model))
            for draw in range(args.control_draws):
                vector = rng.normal(size=mu.shape)
                vector *= np.linalg.norm(mu) / max(np.linalg.norm(vector), 1e-12)
                variants.append(("random_same_norm", speech[test] + vector, speech[train] + vector, draw))
            for variant, query, calibration_query, draw in variants:
                metrics = verification_metrics(query, singing[test], calibration_query, singing[train])
                verification.append({
                    "dataset": "GTSinger_same_text_control_fixed_frames", "model": model, "layer": layer, "split_seed": seed,
                    "variant": variant, "control_draw": draw, "fixed_frames": args.fixed_frames,
                    "crops_per_pair": args.crops_per_pair, "test_speakers": ";".join(speakers[i] for i in test),
                    "trial_construction": "matched-frame speaker centroids; diagonal speech-singing genuine; all off-diagonal impostors; train speakers excluded from test",
                    "threshold_policy": "TMR thresholds are selected from train-speaker trials only; EER is descriptive held-out ROC crossing",
                    **metrics,
                })
                if draw == "":
                    scores = cosine_matrix(query, singing[test])
                    genuine = np.diag(scores)
                    impostor = scores[~np.eye(len(scores), dtype=bool)]
                    curves.extend({"dataset": "GTSinger_same_text_control_fixed_frames", "model": model, "layer": layer, "split_seed": seed, "variant": variant, **point} for point in roc_det_points(genuine, impostor))
            raw_z = np.vstack([speech, singing])
            corrected_z = np.vstack([speech + 0.5 * mu, singing - 0.5 * mu])
            labels = speakers + speakers
            y = np.asarray([0] * len(speakers) + [1] * len(speakers), dtype=int)
            train_probe = np.asarray([i for i, speaker in enumerate(labels) if split[speaker] == "train"])
            test_probe = np.asarray([i for i, speaker in enumerate(labels) if split[speaker] == "test"])
            for variant, z in [("raw", raw_z), ("symmetric_global_corrected", corrected_z)]:
                mode_rows.append({
                    "dataset": "GTSinger_same_text_control_fixed_frames", "model": model, "layer": layer, "split_seed": seed,
                    "variant": variant, "fixed_frames": args.fixed_frames, "mode_probe_granularity": "speaker_mode_centroid",
                    **mode_probe_centroid(z, y, train_probe, test_probe),
                })
    metrics = ["R1", "R5", "mean_rank", "MRR", "EER_test_descriptive", "TMR_at_FMR_1pct", "TMR_at_FMR_0_1pct", "genuine_score_mean", "impostor_score_mean", "genuine_minus_impostor"]
    random_rows = random_means(verification, metrics)
    all_verification = verification + random_rows
    write_csv(selection_audit, args.results_dir / "pair_selection_audit.csv")
    write_csv(crop_audit, args.results_dir / "matched_frame_crop_audit.csv")
    write_csv(verification, args.results_dir / "matched_frame_verification_per_split.csv")
    write_csv(summary(all_verification, ["dataset", "model", "layer", "variant"], metrics), args.results_dir / "matched_frame_verification_summary.csv")
    write_csv(paired_deltas(all_verification, ["R1", "EER_test_descriptive", "TMR_at_FMR_1pct", "genuine_minus_impostor"], args.bootstrap_samples), args.results_dir / "matched_frame_paired_split_deltas.csv")
    write_csv(curves, args.results_dir / "matched_frame_roc_det_per_split.csv")
    write_roc_det_plots(curves, args.results_dir)
    write_csv(mode_rows, args.results_dir / "matched_frame_mode_probe.csv")
    write_csv(cosine_rows, args.results_dir / "matched_frame_global_vector_cosine.csv")
    write_yaml({
        "experiment_name": "identity_residual_matched_frames_2026-07-10", "hostname": socket.gethostname(), "python": sys.executable, "git_commit": current_git_commit(),
        "run_root": str(args.run_root), "feature_root": str(FEATURE_ROOT), "cache_root": os.environ.get("XDG_CACHE_HOME", ""),
        "selection": "technique=control and independent lexical equality after NFKC/boundary normalization", "fixed_frames": args.fixed_frames, "crops_per_pair": args.crops_per_pair, "candidate_count": args.candidate_count,
        "global_vector_fit_population": "train speakers only, using matched-frame samples", "threshold_policy": "TMR train-calibrated; EER descriptive held-out ROC crossing", "interpretation": "reevaluation under matched effective-frame conditions; this does not claim duration removal", "known_limitations": "GTSinger has 20 singers and language is speaker-confounded; voiced-mask matching does not remove every recording or segmentation difference",
    }, args.results_dir / "experiment_card.yaml")
    (args.results_dir / "README_results.md").write_text(
        "# GTSinger Fixed-Frame Global Residual Control\n\n"
        "Each retained same-text/control-technique pair contributes equal-length speech and singing crops. Several candidate crops are sampled on both sides; the selected pair minimizes the difference in voiced-frame proportion. Global vectors, mode probes, and verification thresholds use train speakers only. This is a matched-frame reevaluation, not a claim that duration has been removed.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-frame matched-duration control for the identity residual.")
    parser.add_argument("--pairs", type=Path, default=PAIRS)
    parser.add_argument("--utterances", type=Path, default=UTTERANCES)
    parser.add_argument("--run-root", type=Path, default=Path("/localdisk/bowen/singing_identity/runs/identity_residual_matched_frames_2026-07-10"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/identity_residual_matched_frames_2026-07-10"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--fixed-frames", type=int, default=100)
    parser.add_argument("--crops-per-pair", type=int, default=5)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--control-draws", type=int, default=50)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def main() -> int:
    try:
        run(parse_args())
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
