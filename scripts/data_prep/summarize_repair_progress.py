#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_time(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize targeted GTSinger repair progress.")
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    args = parser.parse_args()
    status = json.loads(args.status.read_text(encoding="utf-8"))
    first = None
    last = None
    successes = 0
    if args.events.exists():
        with args.events.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("status") != "success":
                    continue
                successes += 1
                when = parse_time(str(event.get("updated_at", "")))
                if when is None:
                    continue
                first = when if first is None or when < first else first
                last = when if last is None or when > last else last
    counters = status.get("counters", {})
    total = int(status.get("total") or status.get("config", {}).get("num_files_to_attempt") or 0)
    index = int(status.get("index") or 0)
    downloaded = int(counters.get("downloaded", successes) or successes)
    already = int(counters.get("already_exists", 0) or 0)
    remaining = max(0, total - index)
    rate = None
    eta_sec = None
    if first and last and last > first and downloaded > 0:
        elapsed = (last - first).total_seconds()
        rate = downloaded / elapsed
        eta_sec = remaining / max(rate, 1e-9)
    summary = {
        "status": status.get("status"),
        "index": index,
        "total": total,
        "already_exists": already,
        "downloaded": downloaded,
        "remaining_index_count": remaining,
        "download_rate_per_min": None if rate is None else rate * 60.0,
        "eta_hours_by_download_rate": None if eta_sec is None else eta_sec / 3600.0,
        "last_path": status.get("last_path"),
        "updated_at": status.get("updated_at"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
