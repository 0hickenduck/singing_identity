#!/usr/bin/env python
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from singing_identity.utils.research_utils import (  # noqa: E402
    ExperimentError,
    cosine_similarity_matrix,
    load_feature_vector,
    read_table,
    validate_pairs,
    validate_utterances,
    write_json,
    write_table,
)


def ridge_predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.column_stack([np.ones(x.shape[0]), x])
    return x_aug @ weights


def write_condition(path: Path, vector: np.ndarray, metadata: dict[str, str | float | bool]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        vector=np.asarray(vector, dtype=np.float32),
        metadata_json=np.asarray([write_json_string(metadata)]),
    )


def write_json_string(value: dict[str, str | float | bool]) -> str:
    import json

    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def run_decoder(command_template: str, condition_path: Path, audio_out: Path, row: dict[str, str]) -> int:
    audio_out.parent.mkdir(parents=True, exist_ok=True)
    values = {
        **row,
        "condition_npz": str(condition_path),
        "audio_out": str(audio_out),
        "output_wav": str(audio_out),
    }
    command = command_template.format(**values)
    return subprocess.run(shlex.split(command), check=False).returncode


def split_pairs_like_mapper(pairs: list[dict], pair_split: str) -> list[dict]:
    if pair_split == "all":
        return pairs
    by_speaker: dict[str, list[dict]] = {}
    for row in pairs:
        by_speaker.setdefault(str(row["speaker_id"]), []).append(row)
    speakers = sorted(by_speaker)
    heldout = set(speakers[-max(1, len(speakers) // 5):])
    return [
        row
        for row in pairs
        if (str(row["speaker_id"]) in heldout) == (pair_split == "test")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize Track 1 Seed-VC prompt-intervention conditions from a trained MicroMapper. "
            "Pass --decoder-command to synthesize audio with an external frozen decoder adapter."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--extractor", required=True)
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--mapper-model", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--decoder-command", default="")
    parser.add_argument("--require-audio", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--pair-split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--include-oracle", action="store_true")
    parser.add_argument("--voiced-only", action="store_true")
    args = parser.parse_args()

    try:
        utterances = read_table(args.manifest)
        validate_utterances(utterances)
        pairs = read_table(args.pairs)
        validate_pairs(pairs, utterances)
        if not args.mapper_model.exists():
            raise ExperimentError(f"Missing mapper model: {args.mapper_model}")
        with np.load(args.mapper_model) as data:
            weights = np.asarray(data["weights"], dtype=np.float64)
        split_pairs = split_pairs_like_mapper(pairs, args.pair_split)
        selected_pairs = split_pairs[: args.max_pairs if args.max_pairs > 0 else len(split_pairs)]
        if not selected_pairs:
            raise ExperimentError(f"No pairs selected for pair_split={args.pair_split}")
        rows = []
        cos_mapped = []
        cos_baseline = []
        failures = []
        for pair in selected_pairs:
            speech_vec = load_feature_vector(
                args.feature_root,
                args.extractor,
                args.checkpoint_hash,
                args.layer,
                pair["speech_utt_id"],
                args.voiced_only,
            )
            singing_vec = load_feature_vector(
                args.feature_root,
                args.extractor,
                args.checkpoint_hash,
                args.layer,
                pair["singing_utt_id"],
                args.voiced_only,
            )
            pred_delta = ridge_predict(speech_vec[None, :], weights)[0]
            mapped_vec = speech_vec + pred_delta
            pair_cos_mapped = float(cosine_similarity_matrix(mapped_vec[None, :], singing_vec[None, :])[0, 0])
            pair_cos_baseline = float(cosine_similarity_matrix(speech_vec[None, :], singing_vec[None, :])[0, 0])
            cos_mapped.append(pair_cos_mapped)
            cos_baseline.append(pair_cos_baseline)
            conditions = [
                ("baseline_speech_prompt", speech_vec, "Source content + target speech reference vector"),
                ("mapped_singing_prompt", mapped_vec, "Source content + MicroMapper-estimated singing reference vector"),
            ]
            if args.include_oracle:
                conditions.append(("oracle_singing_prompt", singing_vec, "Source content + true target singing vector"))
            for condition, vector, description in conditions:
                condition_path = args.out_root / "conditions" / pair["pair_id"] / f"{condition}.npz"
                audio_out = args.out_root / "audio" / pair["pair_id"] / f"{condition}.wav"
                metadata = {
                    "track": "track1",
                    "condition": condition,
                    "description": description,
                    "pair_id": pair["pair_id"],
                    "speaker_id": pair["speaker_id"],
                    "speech_utt_id": pair["speech_utt_id"],
                    "singing_utt_id": pair["singing_utt_id"],
                    "extractor": args.extractor,
                    "checkpoint_hash": args.checkpoint_hash,
                    "layer": args.layer,
                    "mapped_to_oracle_cosine": pair_cos_mapped,
                    "speech_to_oracle_cosine": pair_cos_baseline,
                }
                write_condition(condition_path, vector, metadata)
                decoder_returncode = None
                if args.decoder_command:
                    decoder_returncode = run_decoder(
                        args.decoder_command,
                        condition_path,
                        audio_out,
                        {
                            **pair,
                            "condition": condition,
                            "condition_npz": str(condition_path),
                            "audio_out": str(audio_out),
                        },
                    )
                    if decoder_returncode != 0:
                        failures.append(f"{pair['pair_id']}:{condition}: decoder rc={decoder_returncode}")
                rows.append(
                    {
                        "track": "track1",
                        "condition": condition,
                        "pair_id": pair["pair_id"],
                        "speaker_id": pair["speaker_id"],
                        "speech_utt_id": pair["speech_utt_id"],
                        "singing_utt_id": pair["singing_utt_id"],
                        "condition_npz": str(condition_path),
                        "audio_out": str(audio_out) if args.decoder_command else "",
                        "decoder_returncode": "" if decoder_returncode is None else decoder_returncode,
                        "mapped_to_oracle_cosine": pair_cos_mapped,
                        "speech_to_oracle_cosine": pair_cos_baseline,
                    }
                )
        if args.require_audio and not args.decoder_command:
            raise ExperimentError("--require-audio was set but no --decoder-command was provided")
        if failures:
            raise ExperimentError("; ".join(failures[:10]))
        metrics = {
            "track": "track1",
            "stage": "stage_c_seedvc_condition_materialization",
            "extractor": args.extractor,
            "checkpoint_hash": args.checkpoint_hash,
            "layer": args.layer,
            "pair_split": args.pair_split,
            "pairs": len(selected_pairs),
            "conditions": len(rows),
            "decoder_command_provided": bool(args.decoder_command),
            "mapped_to_oracle_cosine_mean": float(np.mean(cos_mapped)) if cos_mapped else float("nan"),
            "speech_to_oracle_cosine_mean": float(np.mean(cos_baseline)) if cos_baseline else float("nan"),
        }
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    write_table(rows, args.manifest_out)
    write_json(metrics, args.metrics_out)
    print(f"wrote conditions -> {args.out_root}")
    print(f"wrote manifest -> {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
