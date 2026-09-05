#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probing.run_identity_residual_final_validation import (  # noqa: E402
    mode_probe_centroid,
    paired_split_delta_summaries,
    per_query_retrieval_details,
    random_vector_like,
    retrieval_row,
)
from probing.run_identity_residual_suite import (  # noqa: E402
    MODEL_SPECS,
    feature_path,
    feature_vector_from_npz,
    make_speaker_split,
    read_jsonl,
    retrieval,
    write_csv,
    write_yaml,
)
from research_utils import ExperimentError, current_git_commit  # noqa: E402


DEFAULT_PAIRS = Path(
    "/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local/manifests/gtsinger_pairs.jsonl"
)
DEFAULT_UTTERANCES = Path(
    "/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local/manifests/gtsinger_utterances.jsonl"
)
DEFAULT_FEATURE_ROOT = Path("/localdisk/bowen/singing_identity/features/stage1_repaired_200_fresh_local")
DEFAULT_RUN_ROOT = Path("/localdisk/bowen/singing_identity/runs/identity_residual_same_text_2026-07-10")
DEFAULT_RESULTS_DIR = Path("results/identity_residual_same_text_2026-07-10")

GTSINGER_SEEDS = [
    13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
    53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
    151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
    199, 211, 223, 227, 229, 233, 239, 241, 251, 257,
]
PRIMARY_SSL_MODELS = {"wavlm_l12", "mert_l3", "hubert_l6"}


def normalize_lexical_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = re.sub(r"<(?:ap|sp)>", "", text, flags=re.IGNORECASE)
    return "".join(char for char in text if not char.isspace())


def select_clean_control_pairs(
    pairs: list[dict[str, Any]], utterances: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    seen_pair_ids: set[str] = set()
    seen_utt_pairs: set[tuple[str, str]] = set()
    for pair in pairs:
        pair_id = str(pair.get("pair_id", ""))
        speech_id = str(pair.get("speech_utt_id", ""))
        singing_id = str(pair.get("singing_utt_id", ""))
        reason = "accepted"
        speech = utterances.get(speech_id)
        singing = utterances.get(singing_id)
        if str(pair.get("technique", "")).strip().lower() != "control":
            reason = "not_control"
        elif not speech or not singing:
            reason = "missing_utterance"
        elif str(speech.get("speaker_id", "")) != str(pair.get("speaker_id", "")) or str(
            singing.get("speaker_id", "")
        ) != str(pair.get("speaker_id", "")):
            reason = "speaker_mismatch"
        elif normalize_lexical_text(speech.get("text")) != normalize_lexical_text(singing.get("text")):
            reason = "lexical_mismatch"
        elif not normalize_lexical_text(speech.get("text")):
            reason = "empty_text"
        elif pair_id in seen_pair_ids or (speech_id, singing_id) in seen_utt_pairs:
            reason = "duplicate_pair"
        if reason == "accepted":
            seen_pair_ids.add(pair_id)
            seen_utt_pairs.add((speech_id, singing_id))
            accepted.append(pair)
        audit.append(
            {
                "pair_id": pair_id,
                "speaker_id": pair.get("speaker_id", ""),
                "language": pair.get("language", ""),
                "song_id": pair.get("song_id", ""),
                "technique": pair.get("technique", ""),
                "speech_utt_id": speech_id,
                "singing_utt_id": singing_id,
                "manifest_same_text_flag": pair.get("same_text_flag", ""),
                "selection_status": reason,
            }
        )
    if not accepted:
        raise ExperimentError("no clean same-text control pairs")
    speakers = {str(pair["speaker_id"]) for pair in accepted}
    if len(speakers) != 20:
        raise ExperimentError(f"expected 20 clean-pair speakers, got {len(speakers)}")
    return accepted, audit


def load_pair_vectors(
    pairs: list[dict[str, Any]],
    feature_root: Path,
    model: str,
    cache_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    extractor, checkpoint, layer = MODEL_SPECS[model]
    cache_root.mkdir(parents=True, exist_ok=True)
    key = f"{model}_control_lexical_equal_{len(pairs)}"
    speech_path = cache_root / f"{key}_speech.npy"
    singing_path = cache_root / f"{key}_singing.npy"
    meta_path = cache_root / f"{key}.json"
    pair_ids = [str(pair["pair_id"]) for pair in pairs]
    if speech_path.exists() and singing_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("pair_ids") == pair_ids:
            return np.load(speech_path), np.load(singing_path), {
                "extractor": extractor,
                "checkpoint": checkpoint,
                "layer": str(layer),
            }
    speech_vectors = []
    singing_vectors = []
    for pair in pairs:
        speech_feature = feature_path(feature_root, extractor, checkpoint, str(layer), str(pair["speech_utt_id"]))
        singing_feature = feature_path(feature_root, extractor, checkpoint, str(layer), str(pair["singing_utt_id"]))
        speech_vectors.append(feature_vector_from_npz(speech_feature))
        singing_vectors.append(feature_vector_from_npz(singing_feature))
    speech = np.vstack(speech_vectors)
    singing = np.vstack(singing_vectors)
    np.save(speech_path, speech)
    np.save(singing_path, singing)
    meta_path.write_text(json.dumps({"pair_ids": pair_ids}), encoding="utf-8")
    return speech, singing, {"extractor": extractor, "checkpoint": checkpoint, "layer": str(layer)}


def speaker_centroids(
    pairs: list[dict[str, Any]], speech_pair: np.ndarray, singing_pair: np.ndarray
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, pair in enumerate(pairs):
        groups[str(pair["speaker_id"])].append(index)
    speakers = sorted(groups)
    speech = np.vstack([speech_pair[groups[speaker]].mean(axis=0) for speaker in speakers])
    singing = np.vstack([singing_pair[groups[speaker]].mean(axis=0) for speaker in speakers])
    return speakers, speech, singing, {speaker: len(groups[speaker]) for speaker in speakers}


def append_per_speaker_rows(
    rows: list[dict[str, Any]],
    dataset: str,
    model: str,
    layer: str,
    seed: int,
    labels: list[str],
    raw_query: np.ndarray,
    corrected_query: np.ndarray,
    gallery: np.ndarray,
    pair_counts: dict[str, int],
) -> None:
    raw_details = {row["speaker_id"]: row for row in per_query_retrieval_details(raw_query, gallery, labels)}
    corrected_details = {
        row["speaker_id"]: row for row in per_query_retrieval_details(corrected_query, gallery, labels)
    }
    raw_metrics = retrieval(raw_query, gallery, labels, labels)
    corrected_metrics = retrieval(corrected_query, gallery, labels, labels)
    raw_margins = dict(zip(raw_metrics["labels"], raw_metrics["margins"]))
    corrected_margins = dict(zip(corrected_metrics["labels"], corrected_metrics["margins"]))
    for speaker in labels:
        raw = raw_details[speaker]
        corrected = corrected_details[speaker]
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "layer": layer,
                "split_seed": seed,
                "speaker_id": speaker,
                "same_text_control_pairs": pair_counts[speaker],
                "raw_rank": raw["rank"],
                "corrected_rank": corrected["rank"],
                "rank_improvement": int(raw["rank"]) - int(corrected["rank"]),
                "raw_same_minus_impostor_margin": raw_margins[speaker],
                "corrected_same_minus_impostor_margin": corrected_margins[speaker],
                "nearest_impostor_before": raw["nearest_impostor"],
                "nearest_impostor_after": corrected["nearest_impostor"],
            }
        )


def summarize_metric(rows: list[dict[str, Any]], keys: list[str], metric: str) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in keys)].append(float(row[metric]))
    output = []
    for key, values in sorted(groups.items()):
        arr = np.asarray(values, dtype=np.float64)
        output.append(
            {
                **{name: value for name, value in zip(keys, key)},
                "rows": len(arr),
                f"{metric}_mean": float(arr.mean()),
                f"{metric}_std": float(arr.std()),
                f"{metric}_CI_low": float(np.percentile(arr, 2.5)),
                f"{metric}_CI_high": float(np.percentile(arr, 97.5)),
            }
        )
    return output


def run(args: argparse.Namespace) -> None:
    all_pairs = read_jsonl(args.pairs)
    utterance_rows = read_jsonl(args.utterances)
    utterances = {str(row["utt_id"]): row for row in utterance_rows}
    clean_pairs, pair_audit = select_clean_control_pairs(all_pairs, utterances)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.run_root.mkdir(parents=True, exist_ok=True)
    write_csv(pair_audit, args.results_dir / "pair_selection_audit.csv")
    per_split: list[dict[str, Any]] = []
    per_speaker: list[dict[str, Any]] = []
    mode_rows: list[dict[str, Any]] = []
    random_mean_rows: list[dict[str, Any]] = []
    for model in args.models:
        speech_pair, singing_pair, model_meta = load_pair_vectors(
            clean_pairs, args.feature_root, model, args.run_root / "cache"
        )
        speakers, speech, singing, pair_counts = speaker_centroids(clean_pairs, speech_pair, singing_pair)
        for seed in args.seeds:
            split = make_speaker_split(speakers, seed, "gtsinger_10_0_10")
            train_idx = np.asarray([i for i, speaker in enumerate(speakers) if split[speaker] == "train"])
            test_idx = np.asarray([i for i, speaker in enumerate(speakers) if split[speaker] == "test"])
            labels = [speakers[i] for i in test_idx]
            mu = (singing[train_idx] - speech[train_idx]).mean(axis=0)
            s_test = speech[test_idx]
            g_test = singing[test_idx]
            deterministic = [
                ("S->G", "raw", s_test, g_test, "q=C_speech; G=C_singing"),
                ("S->G", "correct_sign_global", s_test + mu, g_test, "q=C_speech+mu_delta_train; G=C_singing"),
                ("S->G", "wrong_sign_global", s_test - mu, g_test, "q=C_speech-mu_delta_train; G=C_singing"),
                ("G->S", "raw", g_test, s_test, "q=C_singing; G=C_speech"),
                ("G->S", "correct_sign_global", g_test - mu, s_test, "q=C_singing-mu_delta_train; G=C_speech"),
                ("G->S", "wrong_sign_global", g_test + mu, s_test, "q=C_singing+mu_delta_train; G=C_speech"),
            ]
            for offset, (direction, variant, query, gallery, formula) in enumerate(deterministic):
                metrics = retrieval(query, gallery, labels, labels)
                row = retrieval_row(
                    "GTSinger_same_text_control",
                    model,
                    model_meta["layer"],
                    seed,
                    split,
                    direction.split("->")[0],
                    direction.split("->")[1],
                    f"{direction}:{variant}",
                    formula,
                    metrics,
                    seed * 1000 + offset,
                    "control_technique;lexical_equal;one_speaker_one_vote",
                )
                row["direction"] = direction
                row["control_draw"] = ""
                per_split.append(row)
            append_per_speaker_rows(
                per_speaker,
                "GTSinger_same_text_control",
                model,
                model_meta["layer"],
                seed,
                labels,
                s_test,
                s_test + mu,
                g_test,
                pair_counts,
            )
            rng = np.random.default_rng(seed * 1009 + len(model))
            random_by_direction: dict[str, list[float]] = {"S->G": [], "G->S": []}
            for draw in range(args.control_draws):
                vector = random_vector_like(mu, rng)
                for direction, query, gallery, formula in [
                    ("S->G", s_test + vector, g_test, "q=C_speech+v_random; ||v_random||=||mu_delta_train||"),
                    ("G->S", g_test - vector, s_test, "q=C_singing-v_random; ||v_random||=||mu_delta_train||"),
                ]:
                    metrics = retrieval(query, gallery, labels, labels)
                    row = retrieval_row(
                        "GTSinger_same_text_control",
                        model,
                        model_meta["layer"],
                        seed,
                        split,
                        direction.split("->")[0],
                        direction.split("->")[1],
                        f"{direction}:random_same_norm",
                        formula,
                        metrics,
                        seed * 100000 + draw * 2 + (direction == "G->S"),
                        f"draw={draw};control_technique;lexical_equal",
                    )
                    row["direction"] = direction
                    row["control_draw"] = draw
                    per_split.append(row)
                    random_by_direction[direction].append(float(metrics["r"]))
            for direction, values in random_by_direction.items():
                random_mean_rows.append(
                    {
                        "dataset": "GTSinger_same_text_control",
                        "model": model,
                        "layer": model_meta["layer"],
                        "split_seed": seed,
                        "variant": f"{direction}:random_same_norm_mean",
                        "R1": float(np.mean(values)),
                    }
                )
            raw_z = np.vstack([speech, singing])
            corrected_z = np.vstack([speech + 0.5 * mu, singing - 0.5 * mu])
            y = np.asarray([0] * len(speakers) + [1] * len(speakers), dtype=int)
            probe_train = np.asarray(
                [i for i, speaker in enumerate(speakers + speakers) if split[speaker] == "train"]
            )
            probe_test = np.asarray(
                [i for i, speaker in enumerate(speakers + speakers) if split[speaker] == "test"]
            )
            for variant, z in [("raw", raw_z), ("symmetric_global_corrected", corrected_z)]:
                probe = mode_probe_centroid(z, y, probe_train, probe_test)
                mode_rows.append(
                    {
                        "dataset": "GTSinger_same_text_control",
                        "model": model,
                        "layer": model_meta["layer"],
                        "split_seed": seed,
                        "variant": variant,
                        **probe,
                        "train_speakers": ";".join(sorted(s for s, part in split.items() if part == "train")),
                        "test_speakers": ";".join(sorted(s for s, part in split.items() if part == "test")),
                        "probe_granularity": "speaker_mode_centroid",
                    }
                )
    write_csv(per_split, args.results_dir / "per_split.csv")
    write_csv(per_speaker, args.results_dir / "per_speaker.csv")
    write_csv(mode_rows, args.results_dir / "mode_probe.csv")
    deterministic_rows = [row for row in per_split if row["control_draw"] == ""]
    summary = summarize_metric(per_split, ["dataset", "model", "layer", "direction", "variant"], "R1")
    write_csv(summary, args.results_dir / "summary.csv")
    comparisons = []
    for direction in ["S->G", "G->S"]:
        comparisons.extend(
            [
                (f"{direction}:correct_sign_global", f"{direction}:raw"),
                (f"{direction}:wrong_sign_global", f"{direction}:raw"),
            ]
        )
    paired = paired_split_delta_summaries(
        deterministic_rows + random_mean_rows,
        comparisons
        + [
            ("S->G:random_same_norm_mean", "S->G:raw"),
            ("G->S:random_same_norm_mean", "G->S:raw"),
        ],
        bootstrap_samples=args.bootstrap_samples,
    )
    write_csv(paired, args.results_dir / "paired_delta_summary.csv")
    ssl_gate = []
    for model in sorted(PRIMARY_SSL_MODELS & set(args.models)):
        correct = next(
            (
                row
                for row in paired
                if row["model"] == model
                and row["target_variant"] == "S->G:correct_sign_global"
                and row["baseline_variant"] == "S->G:raw"
            ),
            None,
        )
        wrong = next(
            (
                row
                for row in paired
                if row["model"] == model
                and row["target_variant"] == "S->G:wrong_sign_global"
            ),
            None,
        )
        random_control = next(
            (
                row
                for row in paired
                if row["model"] == model
                and row["target_variant"] == "S->G:random_same_norm_mean"
            ),
            None,
        )
        if not correct or not wrong or not random_control:
            continue
        gain = float(correct["delta_R1_mean"])
        control_limit = 0.25 * gain
        passed = (
            gain >= 0.05
            and float(correct["delta_R1_CI_low"]) > 0.0
            and float(wrong["delta_R1_mean"]) < control_limit
            and float(random_control["delta_R1_mean"]) < control_limit
        )
        ssl_gate.append(
            {
                "model": model,
                "delta_R1_mean": gain,
                "delta_R1_CI_low": correct["delta_R1_CI_low"],
                "delta_R1_CI_high": correct["delta_R1_CI_high"],
                "wrong_sign_delta_R1": wrong["delta_R1_mean"],
                "random_mean_delta_R1": random_control["delta_R1_mean"],
                "gate_pass": passed,
            }
        )
    overall_pass = sum(bool(row["gate_pass"]) for row in ssl_gate) >= 2
    write_csv(ssl_gate, args.results_dir / "gate_summary.csv")
    write_yaml(
        {
            "experiment_name": "identity_residual_same_text_2026-07-10",
            "dataset": "GTSinger_clean_same_text_control",
            "pairs_manifest": str(args.pairs),
            "utterances_manifest": str(args.utterances),
            "feature_root": str(args.feature_root),
            "run_root": str(args.run_root),
            "hostname": socket.gethostname(),
            "git_commit": current_git_commit(),
            "python": sys.executable,
            "cache_root": os.environ.get("XDG_CACHE_HOME", ""),
            "models": list(args.models),
            "split_seeds": list(args.seeds),
            "clean_pairs": len(clean_pairs),
            "speakers": len({str(pair["speaker_id"]) for pair in clean_pairs}),
            "selection": "technique=control AND NFKC/boundary-insensitive lexical equality",
            "all_preprocessing_fit_population": "train_speakers_only",
            "overall_gate_pass": overall_pass,
            "known_limitations": "20 singers; language is speaker-confounded; same-text flag is not trusted",
        },
        args.results_dir / "experiment_card.yaml",
    )
    report = [
        "# GTSinger Same-Text Global Adapter Validation",
        "",
        f"- clean control pairs: {len(clean_pairs)}",
        f"- speakers: {len({str(pair['speaker_id']) for pair in clean_pairs})}",
        f"- split seeds: {len(args.seeds)}",
        f"- overall gate: {'PASS' if overall_pass else 'FAIL/INCONCLUSIVE'}",
        "",
        "## Selection",
        "",
        "The manifest `same_text_flag` is not trusted. Pairs are retained only when technique is `control` and independently stored speech/singing text agrees after NFKC normalization and removal of `<AP>/<SP>` boundary markers and whitespace.",
        "",
        "## Gate",
        "",
    ]
    for row in ssl_gate:
        report.append(
            f"- {row['model']}: delta R@1={float(row['delta_R1_mean']):.3f}, paired CI=[{float(row['delta_R1_CI_low']):.3f}, {float(row['delta_R1_CI_high']):.3f}], wrong={float(row['wrong_sign_delta_R1']):.3f}, random={float(row['random_mean_delta_R1']):.3f}, pass={row['gate_pass']}"
        )
    report.extend(
        [
            "",
            "## Interpretation",
            "",
            "A pass supports retaining `global speech-to-singing mode residual` as the representation-level term because lexical content is controlled in this subset. A fail does not invalidate the JVS result; it requires narrowing the paper claim to a speech-reference-to-singing-reference domain displacement.",
            "",
            "GTSinger remains supporting evidence because it has only 20 singers and language is strongly coupled to singer identity.",
        ]
    )
    (args.results_dir / "README_results.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Content-controlled GTSinger global residual validation.")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--utterances", type=Path, default=DEFAULT_UTTERANCES)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_SPECS), default=["wavlm_l12", "mert_l3", "hubert_l6", "ecapa"])
    parser.add_argument("--seeds", nargs="+", type=int, default=GTSINGER_SEEDS)
    parser.add_argument("--control-draws", type=int, default=50)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run(args)
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote same-text validation -> {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
