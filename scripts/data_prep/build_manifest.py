#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_utils import (  # noqa: E402
    PAIR_COLUMNS,
    UTTERANCE_COLUMNS,
    ExperimentError,
    coerce_row,
    manifest_summary,
    read_table,
    validate_pairs,
    validate_utterances,
    write_json,
    write_table,
)


def build_manifest(args: argparse.Namespace) -> None:
    source_rows = read_table(args.input)
    rows = [coerce_row(row, UTTERANCE_COLUMNS) for row in source_rows]
    summary = validate_utterances(rows, require_audio_exists=args.require_audio_exists)
    write_table(rows, args.output)
    write_json(summary, args.summary)
    print(f"wrote {len(rows)} utterances -> {args.output}")
    print(f"manifest_hash: {summary['manifest_hash']}")

    if args.pairs_input:
        pair_rows = [coerce_row(row, PAIR_COLUMNS) for row in read_table(args.pairs_input)]
        pair_summary = validate_pairs(pair_rows, rows)
        write_table(pair_rows, args.pairs_output)
        write_json(pair_summary, args.pairs_summary)
        print(f"wrote {len(pair_rows)} pairs -> {args.pairs_output}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize and validate a singing-representation utterance manifest."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL/CSV/Parquet metadata.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL/CSV/Parquet manifest.")
    parser.add_argument("--summary", type=Path, required=True, help="Output manifest summary JSON.")
    parser.add_argument("--pairs-input", type=Path, help="Optional speech/singing pair table.")
    parser.add_argument("--pairs-output", type=Path, help="Optional normalized pair table output.")
    parser.add_argument("--pairs-summary", type=Path, help="Optional pair summary JSON output.")
    parser.add_argument("--require-audio-exists", action="store_true")
    args = parser.parse_args()
    if args.pairs_input and not (args.pairs_output and args.pairs_summary):
        parser.error("--pairs-output and --pairs-summary are required with --pairs-input")
    try:
        build_manifest(args)
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
