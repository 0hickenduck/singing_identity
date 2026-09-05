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

from research_utils import ExperimentError, read_table, write_json, write_table  # noqa: E402


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def mean_or_nan(values: list[float]) -> float:
    clean = [value for value in values if np.isfinite(value)]
    return float(np.mean(clean)) if clean else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate multi-condition Seed-VC outputs with a frozen speaker encoder. "
            "This is an automatic triage proxy, not a substitute for listening."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--baseline-condition", default="baseline_all_speech")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError as exc:
        print(f"error: missing resemblyzer: {exc}", file=sys.stderr)
        return 2

    try:
        rows = read_table(args.manifest)
        if not rows:
            raise ExperimentError("manifest is empty")
        required = {"audio_wav", "source_wav", "target_speech_wav", "target_singing_wav", "target_pair_id", "condition"}
        missing = sorted(required - set(rows[0]))
        if missing:
            raise ExperimentError(f"manifest missing required columns: {', '.join(missing)}")

        encoder = VoiceEncoder(args.device)
        cache: dict[str, np.ndarray] = {}

        def embed(path: str) -> np.ndarray:
            if path not in cache:
                cache[path] = encoder.embed_utterance(preprocess_wav(Path(path)))
            return cache[path]

        score_rows = []
        for row in rows:
            out_emb = embed(str(row["audio_wav"]))
            source_emb = embed(str(row["source_wav"]))
            target_speech_emb = embed(str(row["target_speech_wav"]))
            target_singing_emb = embed(str(row["target_singing_wav"]))
            score_rows.append(
                {
                    **row,
                    "sim_to_source_singing": cosine(out_emb, source_emb),
                    "sim_to_target_speech": cosine(out_emb, target_speech_emb),
                    "sim_to_target_singing": cosine(out_emb, target_singing_emb),
                }
            )

        by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in score_rows:
            by_pair[str(row["target_pair_id"])].append(row)

        delta_rows = []
        for pair_id, group in sorted(by_pair.items()):
            by_condition = {str(row["condition"]): row for row in group}
            if args.baseline_condition not in by_condition:
                raise ExperimentError(f"{pair_id}: missing baseline condition {args.baseline_condition}")
            baseline = by_condition[args.baseline_condition]
            for condition, row in sorted(by_condition.items()):
                delta_rows.append(
                    {
                        "target_pair_id": pair_id,
                        "target_speaker_id": row.get("target_speaker_id"),
                        "source_speaker_id": row.get("source_speaker_id"),
                        "condition": condition,
                        "delta_to_source": float(row["sim_to_source_singing"] - baseline["sim_to_source_singing"]),
                        "delta_to_target_speech": float(row["sim_to_target_speech"] - baseline["sim_to_target_speech"]),
                        "delta_to_target_singing": float(row["sim_to_target_singing"] - baseline["sim_to_target_singing"]),
                        "delta_rms_db": float(row.get("rms_db", float("nan"))) - float(baseline.get("rms_db", float("nan"))),
                        "sim_to_source_singing": float(row["sim_to_source_singing"]),
                        "sim_to_target_speech": float(row["sim_to_target_speech"]),
                        "sim_to_target_singing": float(row["sim_to_target_singing"]),
                    }
                )

        condition_summary = {}
        for condition in sorted({row["condition"] for row in delta_rows}):
            group = [row for row in delta_rows if row["condition"] == condition]
            condition_summary[condition] = {
                "pairs": len(group),
                "mean_delta_to_source": mean_or_nan([float(row["delta_to_source"]) for row in group]),
                "mean_delta_to_target_speech": mean_or_nan([float(row["delta_to_target_speech"]) for row in group]),
                "mean_delta_to_target_singing": mean_or_nan([float(row["delta_to_target_singing"]) for row in group]),
                "mean_delta_rms_db": mean_or_nan([float(row["delta_rms_db"]) for row in group]),
                "positive_delta_to_target_singing": int(sum(float(row["delta_to_target_singing"]) > 0 for row in group)),
            }

        summary = {
            "stage": "seedvc_multicondition_speaker_encoder_eval",
            "encoder": "resemblyzer",
            "baseline_condition": args.baseline_condition,
            "pairs": len(by_pair),
            "conditions": len(score_rows),
            "condition_summary": condition_summary,
            "pair_deltas": delta_rows,
        }
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    write_table(score_rows, args.scores_out)
    write_json(summary, args.summary_out)
    print(json.dumps({k: v for k, v in summary.items() if k != "pair_deltas"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
