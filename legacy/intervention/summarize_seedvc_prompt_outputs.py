#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_utils import ExperimentError, read_table, write_json, write_table  # noqa: E402


def find_audio(output_dir: Path) -> Path:
    wavs = sorted(output_dir.glob("*.wav"))
    if len(wavs) != 1:
        raise ExperimentError(f"Expected one wav in {output_dir}, found {len(wavs)}")
    return wavs[0]


def audio_stats(path: Path) -> dict[str, float | int]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ExperimentError("soundfile is required to summarize generated audio") from exc
    audio, sr = sf.read(str(path), always_2d=False)
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    rms = float(np.sqrt(np.mean(np.square(arr)))) if arr.size else float("nan")
    return {
        "sample_rate": int(sr),
        "num_samples": int(arr.size),
        "duration_sec": float(arr.size / sr) if sr else float("nan"),
        "rms": rms,
        "rms_db": float(20.0 * math.log10(max(rms, 1e-12))),
        "peak": float(np.max(np.abs(arr))) if arr.size else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve and sanity-check Seed-VC prompt baseline outputs.")
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    try:
        rows = read_table(args.conditions)
        out_rows = []
        for row in rows:
            audio_path = find_audio(Path(str(row["output_dir"])))
            out_rows.append(
                {
                    **row,
                    "audio_wav": str(audio_path),
                    **audio_stats(audio_path),
                }
            )
        by_condition = Counter(str(row["condition"]) for row in out_rows)
        summary = {
            "conditions": len(out_rows),
            "pairs": len({str(row["target_pair_id"]) for row in out_rows}),
            "by_condition": dict(sorted(by_condition.items())),
            "missing_or_invalid_audio": 0,
            "duration_sec_mean": float(np.mean([row["duration_sec"] for row in out_rows])) if out_rows else float("nan"),
            "rms_db_mean": float(np.mean([row["rms_db"] for row in out_rows])) if out_rows else float("nan"),
            "peak_max": float(np.max([row["peak"] for row in out_rows])) if out_rows else float("nan"),
        }
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    write_table(out_rows, args.manifest_out)
    write_json(summary, args.summary_out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
