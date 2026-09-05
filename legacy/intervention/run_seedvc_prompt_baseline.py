#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_utils import (  # noqa: E402
    ExperimentError,
    read_table,
    validate_pairs,
    validate_utterances,
    write_json,
    write_table,
)


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


def select_pairs_round_robin(pairs: list[dict], max_pairs: int) -> list[dict]:
    if max_pairs <= 0:
        return pairs
    by_speaker: dict[str, list[dict]] = {}
    for row in pairs:
        by_speaker.setdefault(str(row["speaker_id"]), []).append(row)
    selected: list[dict] = []
    speakers = sorted(by_speaker)
    cursor = 0
    while len(selected) < max_pairs:
        added = False
        for speaker in speakers:
            rows = by_speaker[speaker]
            if cursor < len(rows):
                selected.append(rows[cursor])
                added = True
                if len(selected) >= max_pairs:
                    break
        if not added:
            break
        cursor += 1
    return selected


def find_source_singing(target_pair: dict, pairs: list[dict], utterance_by_id: dict[str, dict]) -> dict:
    """Pick a different-speaker singing source, preferring matched language/song/take."""
    target_speaker = str(target_pair["speaker_id"])
    target_singing = utterance_by_id[str(target_pair["singing_utt_id"])]
    candidate_pairs = [
        row
        for row in pairs
        if str(row["speaker_id"]) != target_speaker
        and str(row.get("language", "")) == str(target_pair.get("language", ""))
    ]
    if not candidate_pairs:
        candidate_pairs = [
            row
            for row in pairs
            if str(row["speaker_id"]) != target_speaker
        ]
    if not candidate_pairs:
        raise ExperimentError(f"No different-speaker source candidates for {target_pair['pair_id']}")

    def score(row: dict) -> tuple[int, str]:
        utt = utterance_by_id[str(row["singing_utt_id"])]
        matches = 0
        for key in ("language", "song_id", "take_id", "technique", "text", "phone_seq"):
            if str(utt.get(key, "")) and str(utt.get(key, "")) == str(target_singing.get(key, "")):
                matches += 1
        return (-matches, str(row["pair_id"]))

    best = sorted(candidate_pairs, key=score)[0]
    return utterance_by_id[str(best["singing_utt_id"])]


def seedvc_command(
    python: str,
    seedvc_root: Path,
    source_wav: Path,
    target_wav: Path,
    output_dir: Path,
    diffusion_steps: int,
    inference_cfg_rate: float,
    fp16: bool,
) -> list[str]:
    return [
        python,
        "inference.py",
        "--source",
        str(source_wav),
        "--target",
        str(target_wav),
        "--output",
        str(output_dir),
        "--diffusion-steps",
        str(diffusion_steps),
        "--length-adjust",
        "1.0",
        "--inference-cfg-rate",
        str(inference_cfg_rate),
        "--f0-condition",
        "True",
        "--auto-f0-adjust",
        "False",
        "--semi-tone-shift",
        "0",
        "--fp16",
        "True" if fp16 else "False",
    ]


def run_command(command: list[str], cwd: Path, env: dict[str, str]) -> int:
    return subprocess.run(command, cwd=cwd, env=env, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build and optionally run the Seed-VC black-box prompt-mode baseline: "
            "same source singing, target speech prompt versus target singing prompt."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--seedvc-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-pairs", type=int, default=10)
    parser.add_argument("--pair-split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    try:
        if not (args.seedvc_root / "inference.py").exists():
            raise ExperimentError(f"Seed-VC inference.py not found in {args.seedvc_root}")
        utterances = read_table(args.manifest)
        validate_utterances(utterances, require_audio_exists=True)
        pairs = read_table(args.pairs)
        validate_pairs(pairs, utterances)
        utterance_by_id = {str(row["utt_id"]): row for row in utterances}
        split_pairs = split_pairs_like_mapper(pairs, args.pair_split)
        selected_pairs = select_pairs_round_robin(split_pairs, args.max_pairs)
        if not selected_pairs:
            raise ExperimentError(f"No pairs selected for pair_split={args.pair_split}")

        env = dict(**__import__("os").environ)
        env.setdefault("TMPDIR", "/tmp")
        env.setdefault("HF_HOME", str(args.out_root / "hf_home"))
        env.setdefault("HF_HUB_CACHE", str(args.out_root / "hf_home" / "hub"))

        rows = []
        failures = []
        for target_pair in selected_pairs:
            source = find_source_singing(target_pair, pairs, utterance_by_id)
            target_speech = utterance_by_id[str(target_pair["speech_utt_id"])]
            target_singing = utterance_by_id[str(target_pair["singing_utt_id"])]
            for condition, target in (
                ("target_speech_prompt", target_speech),
                ("target_singing_prompt", target_singing),
            ):
                condition_out = args.out_root / "audio" / str(target_pair["pair_id"]) / condition
                command = seedvc_command(
                    args.python,
                    args.seedvc_root,
                    Path(str(source["wav_path"])),
                    Path(str(target["wav_path"])),
                    condition_out,
                    args.diffusion_steps,
                    args.inference_cfg_rate,
                    args.fp16,
                )
                rc: int | str = ""
                if args.run:
                    condition_out.mkdir(parents=True, exist_ok=True)
                    rc = run_command(command, args.seedvc_root, env)
                    if rc != 0:
                        failure = f"{target_pair['pair_id']}:{condition}: rc={rc}"
                        failures.append(failure)
                        if args.stop_on_error:
                            raise ExperimentError(failure)
                rows.append(
                    {
                        "track": "track1",
                        "stage": "seedvc_prompt_mode_baseline",
                        "condition": condition,
                        "target_pair_id": target_pair["pair_id"],
                        "target_speaker_id": target_pair["speaker_id"],
                        "source_speaker_id": source["speaker_id"],
                        "source_singing_utt_id": source["utt_id"],
                        "target_speech_utt_id": target_speech["utt_id"],
                        "target_singing_utt_id": target_singing["utt_id"],
                        "source_wav": source["wav_path"],
                        "target_wav": target["wav_path"],
                        "output_dir": str(condition_out),
                        "command": shlex.join(command),
                        "returncode": rc,
                    }
                )
        metrics = {
            "track": "track1",
            "stage": "seedvc_prompt_mode_baseline",
            "seedvc_root": str(args.seedvc_root),
            "pair_split": args.pair_split,
            "pairs": len(selected_pairs),
            "conditions": len(rows),
            "run": args.run,
            "diffusion_steps": args.diffusion_steps,
            "inference_cfg_rate": args.inference_cfg_rate,
            "fp16": args.fp16,
            "failures": failures,
        }
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    write_table(rows, args.manifest_out)
    write_json(metrics, args.metrics_out)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
