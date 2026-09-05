#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    for attempt in range(3):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            if attempt < 2:
                time.sleep(0.15)
                continue
            return {"status": "invalid_json", "path": str(path)}


def read_latest_jsonl(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            position = handle.tell()
            buffer = bytearray()
            while position > 0:
                position -= 1
                handle.seek(position)
                char = handle.read(1)
                if char == b"\n" and buffer:
                    break
                buffer.extend(char)
            line = bytes(reversed(buffer)).decode("utf-8").strip()
        return json.loads(line) if line else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def read_recent_jsonl(path: Path, limit: int = 200) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: deque[dict[str, Any]] = deque(maxlen=limit)
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return list(rows)


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def eta_line(status: dict[str, Any]) -> str:
    index = status.get("index")
    total = status.get("total")
    updated_at = parse_time(status.get("updated_at"))
    config = status.get("config", {})
    started_at = parse_time(config.get("started_at")) or None
    if not index or not total or not updated_at:
        return "ETA: unavailable"
    progress = float(index) / max(1.0, float(total))
    if progress <= 0:
        return "ETA: unavailable"
    remaining = int(total) - int(index)
    return f"Progress: {index}/{total} ({progress:.1%}); remaining files: {remaining}"


def recent_rate_line(status: dict[str, Any], recent_events: list[dict[str, Any]]) -> str:
    current_total = status.get("total")
    successes = [
        event
        for event in recent_events
        if event.get("status") == "success"
        and event.get("updated_at")
        and (current_total is None or event.get("total") == current_total)
    ]
    if len(successes) < 2:
        return "Recent rate: unavailable"
    first = successes[0]
    last = successes[-1]
    first_time = parse_time(first.get("updated_at"))
    last_time = parse_time(last.get("updated_at"))
    if not first_time or not last_time:
        return "Recent rate: unavailable"
    elapsed_sec = max(1.0, (last_time - first_time).total_seconds())
    completed = max(1, len(successes) - 1)
    files_per_hour = completed / elapsed_sec * 3600.0
    index = int(status.get("index") or 0)
    total = int(status.get("total") or 0)
    remaining = max(0, total - index)
    eta_hours = remaining / files_per_hour if files_per_hour > 0 else 0.0
    return f"Recent rate: {files_per_hour:.1f} files/hour; ETA from recent rate: {eta_hours:.1f} hours"


def read_stage1_jobs(stage1_status: Path) -> list[dict[str, Any]]:
    jobs_root = stage1_status.parent / "jobs"
    if not jobs_root.exists():
        return []
    rows = []
    for path in sorted(jobs_root.glob("*/status.json")):
        payload = read_json(path)
        if payload:
            payload.setdefault("name", path.parent.name)
            rows.append(payload)
    return rows


def stage1_jobs_line(jobs: list[dict[str, Any]]) -> str:
    if not jobs:
        return "stage1_jobs: none yet"
    counts: dict[str, int] = {}
    for job in jobs:
        status = str(job.get("status", "missing"))
        counts[status] = counts.get(status, 0) + 1
    return f"stage1_jobs: {json.dumps(counts, sort_keys=True)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize the repair + Stage 1 supervisor status.")
    parser.add_argument(
        "--supervisor-status",
        type=Path,
        default=Path("/localdisk/bowen/singing_identity/status/stage1_repaired_200_fresh_local/status.json"),
    )
    parser.add_argument(
        "--repair-status",
        type=Path,
        default=Path("/localdisk/bowen/singing_identity/status/stage1_repaired_200_fresh_local/repair_status.json"),
    )
    parser.add_argument(
        "--repair-events",
        type=Path,
        default=Path("/localdisk/bowen/singing_identity/status/stage1_repaired_200_fresh_local/repair_events.jsonl"),
    )
    parser.add_argument(
        "--stage1-status",
        type=Path,
        default=Path("/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local/status.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/stage1_repaired_200_fresh_local_report.md"),
    )
    args = parser.parse_args()

    supervisor = read_json(args.supervisor_status)
    repair = read_json(args.repair_status)
    stage1 = read_json(args.stage1_status)
    latest_event = read_latest_jsonl(args.repair_events)
    recent_events = read_recent_jsonl(args.repair_events)
    stage1_jobs = read_stage1_jobs(args.stage1_status)

    print("Stage 1 supervisor")
    print(f"- supervisor_status: {supervisor.get('status', 'missing')}")
    print(f"- repair_status: {repair.get('status', 'missing')}")
    print(f"- stage1_status: {stage1.get('status', 'missing')}")
    if stage1 or stage1_jobs:
        print(f"- {stage1_jobs_line(stage1_jobs)}")
    print(f"- report_exists: {args.report.exists()} ({args.report})")
    if repair:
        print(f"- {eta_line(repair)}")
        counters = repair.get("counters", {})
        if counters:
            print(f"- repair_counters: {json.dumps(counters, sort_keys=True)}")
        if repair.get("last_path"):
            print(f"- last_path: {repair.get('last_path')}")
        print(f"- {recent_rate_line(repair, recent_events)}")
    if latest_event:
        event_status = latest_event.get("status")
        event_path = latest_event.get("path")
        event_time = latest_event.get("updated_at")
        print(f"- latest_event: {event_status} {event_path} at {event_time}")
    if repair.get("status") == "rate_limited":
        print("- next_action: wait for cooldown or add HF token at /tmp/singing_identity_hf_token, then rerun the same supervisor command.")
    elif supervisor.get("status") == "running_repair":
        print("- next_action: keep repair running; Stage 1 will start after repair exits.")
    elif supervisor.get("status") == "running_stage1":
        failed = [job for job in stage1_jobs if job.get("status") in {"failed", "timeout", "invalid_status"}]
        if failed:
            print("- next_action: inspect failed Stage 1 job logs listed under the configured Stage 1 run root.")
            for job in failed[:10]:
                print(f"  - failed_job: {job.get('name')} status={job.get('status')} log={job.get('log')}")
        else:
            print("- next_action: monitor Stage 1 job statuses under the configured Stage 1 run root.")
    elif supervisor.get("status") == "complete":
        print("- next_action: inspect the final report and human checklist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
