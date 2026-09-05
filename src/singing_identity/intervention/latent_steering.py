#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from singing_identity.utils.research_utils import (  # noqa: E402
    ExperimentError,
    load_feature_interval_vector,
    load_feature_vector,
    read_table,
    validate_required_columns,
    write_json,
    write_table,
)


DIRECTION_COLUMNS = ["run_id", "phone", "target_technique", "dim", "value"]


def load_direction(path: Path, phone: str, technique: str) -> np.ndarray:
    rows = read_table(path)
    validate_required_columns(rows, DIRECTION_COLUMNS, "technique directions")
    selected = [row for row in rows if row["phone"] == phone and row["target_technique"] == technique]
    if not selected:
        raise ExperimentError(f"No direction rows for phone={phone!r}, technique={technique!r}")
    dims = sorted((int(row["dim"]), float(row["value"])) for row in selected)
    expected = list(range(len(dims)))
    actual = [dim for dim, _ in dims]
    if actual != expected:
        raise ExperimentError(f"Direction dims are not contiguous from zero: first dims={actual[:10]}")
    return np.asarray([value for _, value in dims], dtype=np.float64)


def write_condition(path: Path, vector: np.ndarray, metadata: dict[str, str | float | bool]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        vector=np.asarray(vector, dtype=np.float32),
        metadata_json=np.asarray([json.dumps(metadata, sort_keys=True, ensure_ascii=False)]),
    )


def pair_source_vector(args: argparse.Namespace, row: dict) -> np.ndarray:
    if row.get("base_phone_start_sec") not in {None, ""}:
        return load_feature_interval_vector(
            args.feature_root,
            args.extractor,
            args.checkpoint_hash,
            args.layer,
            row["base_utt_id"],
            float(row["base_phone_start_sec"]),
            float(row["base_phone_end_sec"]),
            args.voiced_only,
        )
    return load_feature_vector(
        args.feature_root,
        args.extractor,
        args.checkpoint_hash,
        args.layer,
        row["base_utt_id"],
        args.voiced_only,
    )


def run_decoder(command_template: str, condition_path: Path, audio_out: Path, row: dict[str, str]) -> int:
    audio_out.parent.mkdir(parents=True, exist_ok=True)
    values = {
        **row,
        "condition_npz": str(condition_path),
        "audio_out": str(audio_out),
        "output_wav": str(audio_out),
    }
    return subprocess.run(shlex.split(command_template.format(**values)), check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize Track 2 three-condition latent-steering comparisons. "
            "Pass --decoder-command to synthesize audio with an external frozen decoder adapter."
        )
    )
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--extractor", required=True)
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--phone", default="<AP>")
    parser.add_argument("--technique", default="breathy")
    parser.add_argument("--lambda-scale", type=float, default=1.0)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--decoder-command", default="")
    parser.add_argument("--require-audio", action="store_true")
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--voiced-only", action="store_true")
    args = parser.parse_args()

    try:
        pairs = read_table(args.pairs)
        required = ["pair_id", "base_utt_id", "technique_utt_id", "speaker_id", "language", "phone", "target_technique"]
        validate_required_columns(pairs, required, "technique pairs")
        selected = [
            row
            for row in pairs
            if row["phone"] == args.phone and row["target_technique"] == args.technique
        ]
        if args.max_examples > 0:
            selected = selected[: args.max_examples]
        if not selected:
            raise ExperimentError(f"No pairs for phone={args.phone!r}, technique={args.technique!r}")
        direction = load_direction(args.directions, args.phone, args.technique)
        direction_norm = float(np.linalg.norm(direction))
        rows = []
        failures = []
        by_condition: dict[str, int] = defaultdict(int)
        for pair in selected:
            source_vec = pair_source_vector(args, pair)
            if len(source_vec) != len(direction):
                raise ExperimentError(
                    f"Direction dim {len(direction)} does not match source vector dim {len(source_vec)}"
                )
            baseline_vec = np.zeros_like(source_vec)
            steered_vec = source_vec + args.lambda_scale * direction
            conditions = [
                ("baseline", baseline_vec, "Source content + target speaker ID"),
                ("baseline_plus_source_vector", source_vec, "Source content + target speaker ID + original source vector"),
                ("steered", steered_vec, "Source content + target speaker ID + technique direction delta Z"),
            ]
            for condition, vector, description in conditions:
                condition_path = args.out_root / "conditions" / pair["pair_id"] / f"{condition}.npz"
                audio_out = args.out_root / "audio" / pair["pair_id"] / f"{condition}.wav"
                metadata = {
                    "track": "track2",
                    "condition": condition,
                    "description": description,
                    "pair_id": pair["pair_id"],
                    "speaker_id": pair["speaker_id"],
                    "base_utt_id": pair["base_utt_id"],
                    "technique_utt_id": pair["technique_utt_id"],
                    "phone": args.phone,
                    "technique": args.technique,
                    "lambda_scale": args.lambda_scale,
                    "direction_norm": direction_norm,
                    "extractor": args.extractor,
                    "checkpoint_hash": args.checkpoint_hash,
                    "layer": args.layer,
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
                        "track": "track2",
                        "condition": condition,
                        "pair_id": pair["pair_id"],
                        "speaker_id": pair["speaker_id"],
                        "base_utt_id": pair["base_utt_id"],
                        "technique_utt_id": pair["technique_utt_id"],
                        "phone": args.phone,
                        "technique": args.technique,
                        "lambda_scale": args.lambda_scale,
                        "condition_npz": str(condition_path),
                        "audio_out": str(audio_out) if args.decoder_command else "",
                        "decoder_returncode": "" if decoder_returncode is None else decoder_returncode,
                    }
                )
                by_condition[condition] += 1
        if args.require_audio and not args.decoder_command:
            raise ExperimentError("--require-audio was set but no --decoder-command was provided")
        if failures:
            raise ExperimentError("; ".join(failures[:10]))
        metrics = {
            "track": "track2",
            "stage": "latent_steering_condition_materialization",
            "extractor": args.extractor,
            "checkpoint_hash": args.checkpoint_hash,
            "layer": args.layer,
            "phone": args.phone,
            "technique": args.technique,
            "lambda_scale": args.lambda_scale,
            "direction_norm": direction_norm,
            "examples": len(selected),
            "conditions": dict(sorted(by_condition.items())),
            "decoder_command_provided": bool(args.decoder_command),
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
