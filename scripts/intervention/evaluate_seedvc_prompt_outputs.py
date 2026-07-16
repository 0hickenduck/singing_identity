#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_utils import ExperimentError, read_table, write_json, write_table  # noqa: E402


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Seed-VC prompt-mode outputs with a frozen speaker encoder. "
            "These scores are an automatic triage proxy, not a substitute for listening."
        )
    )
    parser.add_argument("--listening-manifest", type=Path, required=True)
    parser.add_argument("--scores-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError as exc:
        print(f"error: missing resemblyzer: {exc}", file=sys.stderr)
        return 2

    try:
        rows = read_table(args.listening_manifest)
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
            target_speech_emb = embed(str(row["target_wav"] if row["condition"] == "target_speech_prompt" else row["target_wav"]))
            # The listening manifest has one target_wav per condition, so recover both prompt refs from paired rows below.
            score_rows.append(
                {
                    **row,
                    "_output_embedding": out_emb,
                    "_source_embedding": source_emb,
                    "_condition_target_embedding": target_speech_emb,
                }
            )

        by_pair: dict[str, list[dict]] = defaultdict(list)
        for row in score_rows:
            by_pair[str(row["target_pair_id"])].append(row)

        out_rows = []
        pair_rows = []
        for pair_id, group in sorted(by_pair.items()):
            refs = {str(row["condition"]): row for row in group}
            if "target_speech_prompt" not in refs or "target_singing_prompt" not in refs:
                raise ExperimentError(f"{pair_id}: missing paired prompt conditions")
            target_speech_emb = refs["target_speech_prompt"]["_condition_target_embedding"]
            target_singing_emb = refs["target_singing_prompt"]["_condition_target_embedding"]
            source_emb = refs["target_speech_prompt"]["_source_embedding"]
            condition_scores: dict[str, dict[str, float]] = {}
            for row in group:
                out_emb = row["_output_embedding"]
                scores = {
                    "sim_to_source_singing": cosine(out_emb, source_emb),
                    "sim_to_target_speech_prompt": cosine(out_emb, target_speech_emb),
                    "sim_to_target_singing_prompt": cosine(out_emb, target_singing_emb),
                }
                condition_scores[str(row["condition"])] = scores
                clean = {k: v for k, v in row.items() if not k.startswith("_")}
                out_rows.append({**clean, **scores})
            speech_scores = condition_scores["target_speech_prompt"]
            singing_scores = condition_scores["target_singing_prompt"]
            pair_rows.append(
                {
                    "target_pair_id": pair_id,
                    "target_speaker_id": refs["target_speech_prompt"]["target_speaker_id"],
                    "source_speaker_id": refs["target_speech_prompt"]["source_speaker_id"],
                    "singing_prompt_delta_to_target_singing": singing_scores["sim_to_target_singing_prompt"]
                    - speech_scores["sim_to_target_singing_prompt"],
                    "singing_prompt_delta_to_target_speech": singing_scores["sim_to_target_speech_prompt"]
                    - speech_scores["sim_to_target_speech_prompt"],
                    "singing_prompt_delta_to_source": singing_scores["sim_to_source_singing"]
                    - speech_scores["sim_to_source_singing"],
                }
            )

        summary = {
            "encoder": "resemblyzer",
            "conditions": len(out_rows),
            "pairs": len(pair_rows),
            "mean_singing_prompt_delta_to_target_singing": float(
                np.mean([row["singing_prompt_delta_to_target_singing"] for row in pair_rows])
            )
            if pair_rows
            else float("nan"),
            "mean_singing_prompt_delta_to_target_speech": float(
                np.mean([row["singing_prompt_delta_to_target_speech"] for row in pair_rows])
            )
            if pair_rows
            else float("nan"),
            "mean_singing_prompt_delta_to_source": float(
                np.mean([row["singing_prompt_delta_to_source"] for row in pair_rows])
            )
            if pair_rows
            else float("nan"),
            "pair_deltas": pair_rows,
        }
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    write_table(out_rows, args.scores_out)
    write_json(summary, args.summary_out)
    print(json.dumps({k: v for k, v in summary.items() if k != "pair_deltas"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
