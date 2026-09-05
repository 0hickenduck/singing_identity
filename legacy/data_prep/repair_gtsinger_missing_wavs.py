#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_GTSINGER = "/localdisk/bowen/singing_identity/data/gtsinger_domain_eval"
DEFAULT_REVISION = "4426c862beed558b7e1cb8a4dce7e8c0c83bb208"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc)
    return "429" in message or "Too Many Requests" in message or "rate limit" in message.lower()


def retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("retry-after") or headers.get("Retry-After")
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_token(token_file: Path | None) -> str | None:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token_file:
        return None
    try:
        if token_file.exists():
            token = token_file.read_text(encoding="utf-8").strip()
            return token or None
    except OSError:
        return None
    return None


def select_rows(rows: list[dict[str, Any]], max_rows_per_singer: int) -> list[dict[str, Any]]:
    by_singer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("speech_fn") and row.get("wav_fn") and row.get("singer"):
            by_singer[str(row["singer"])].append(row)
    selected = []
    for _, singer_rows in sorted(by_singer.items()):
        selected.extend(singer_rows[:max_rows_per_singer])
    return selected


def missing_wavs(root: Path, max_rows_per_singer: int, languages: set[str] | None = None) -> list[str]:
    missing: list[str] = []
    for metadata_path in sorted((root / "processed").glob("*/metadata.json")):
        language = metadata_path.parent.name
        if languages and language not in languages:
            continue
        rows = json.loads(metadata_path.read_text(encoding="utf-8"))
        for row in select_rows(rows, max_rows_per_singer):
            for key in ("wav_fn", "speech_fn"):
                rel = str(row.get(key, ""))
                if rel and not (root / rel).exists():
                    missing.append(rel)
    return sorted(set(missing))


def selected_wavs(root: Path, max_rows_per_singer: int, languages: set[str] | None = None) -> list[str]:
    selected: list[str] = []
    for metadata_path in sorted((root / "processed").glob("*/metadata.json")):
        language = metadata_path.parent.name
        if languages and language not in languages:
            continue
        rows = json.loads(metadata_path.read_text(encoding="utf-8"))
        for row in select_rows(rows, max_rows_per_singer):
            for key in ("wav_fn", "speech_fn"):
                rel = str(row.get(key, ""))
                if rel:
                    selected.append(rel)
    return sorted(set(selected))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Targeted repair for missing GTSinger wav files referenced by processed metadata.")
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_GTSINGER))
    parser.add_argument("--repo-id", default="GTSinger/GTSinger")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-rows-per-singer", type=int, default=5000)
    parser.add_argument("--languages", nargs="*")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of missing files to attempt.")
    parser.add_argument("--all-selected", action="store_true", help="Attempt every selected wav path, skipping existing files one by one.")
    parser.add_argument(
        "--resume-existing-queue",
        action="store_true",
        help="Load --queue-out if it already exists, avoiding a metadata/filesystem rescan on restart.",
    )
    parser.add_argument("--skip-final-scan", action="store_true", help="Do not rescan the dataset for remaining missing files at completion.")
    parser.add_argument("--sleep-sec", type=float, default=2.0)
    parser.add_argument("--retry-wait-sec", type=float, default=60.0)
    parser.add_argument("--max-attempts-per-file", type=int, default=4)
    parser.add_argument("--max-consecutive-rate-limits", type=int, default=3)
    parser.add_argument("--max-total-rate-limits", type=int, default=10)
    parser.add_argument("--rate-limit-cooldown-sec", type=float, default=900.0)
    parser.add_argument("--status", type=Path, default=Path("results/data_repair/gtsinger_targeted/status.json"))
    parser.add_argument("--events", type=Path, default=Path("results/data_repair/gtsinger_targeted/events.jsonl"))
    parser.add_argument("--queue-out", type=Path, default=Path("results/data_repair/gtsinger_targeted/missing_queue.json"))
    parser.add_argument("--local-cache-root", type=Path, default=Path("/tmp/singing_identity_hf_targeted"))
    parser.add_argument("--token-file", type=Path, default=Path("/tmp/singing_identity_hf_token"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_existing_queue(path: Path) -> list[str] | None:
    try:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        files = payload.get("files") if isinstance(payload, dict) else payload
        if not isinstance(files, list):
            return None
        return sorted({str(item) for item in files if item})
    except (OSError, json.JSONDecodeError):
        return None


def write_queue(path: Path, root: Path, mode: str, files: list[str]) -> None:
    write_json(
        {
            "updated_at": now(),
            "root": str(root),
            "mode": mode,
            "num_files": len(files),
            "files": files,
        },
        path,
    )


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    args.local_cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HOME", str(args.local_cache_root))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
    token = load_token(args.token_file)
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    existing_queue = load_existing_queue(args.queue_out) if args.resume_existing_queue and not args.force else None
    write_json(
        {
            "status": (
                "loading_existing_queue"
                if existing_queue is not None
                else "collecting_selected_paths"
                if args.all_selected
                else "scanning_metadata"
            ),
            "updated_at": now(),
            "root": str(root),
            "max_rows_per_singer": args.max_rows_per_singer,
            "hf_hub_disable_xet": os.environ.get("HF_HUB_DISABLE_XET"),
            "hf_home": os.environ.get("HF_HOME"),
            "has_hf_token": bool(token),
        },
        args.status,
    )
    languages = set(args.languages) if args.languages else None
    if existing_queue is not None:
        files = existing_queue
    else:
        files = selected_wavs(root, args.max_rows_per_singer, languages) if args.all_selected else missing_wavs(root, args.max_rows_per_singer, languages)
    if args.limit > 0:
        files = files[: args.limit]
    queue_mode = "resume_existing_queue" if existing_queue is not None else "all_selected" if args.all_selected else "missing_only"
    write_queue(args.queue_out, root, queue_mode, files)
    counters: Counter[str] = Counter()
    config = {
        "root": str(root),
        "repo_id": args.repo_id,
        "revision": args.revision,
        "max_rows_per_singer": args.max_rows_per_singer,
        "languages": sorted(languages) if languages else [],
        "limit": args.limit,
        "all_selected": args.all_selected,
        "resume_existing_queue": existing_queue is not None,
        "num_files_to_attempt": len(files),
        "hf_hub_disable_xet": os.environ.get("HF_HUB_DISABLE_XET"),
        "hf_home": os.environ.get("HF_HOME"),
        "has_hf_token": bool(token),
        "queue_out": str(args.queue_out),
        "sleep_sec": args.sleep_sec,
        "max_consecutive_rate_limits": args.max_consecutive_rate_limits,
        "max_total_rate_limits": args.max_total_rate_limits,
    }
    write_json({"status": "running", "updated_at": now(), "config": config, "counters": dict(counters)}, args.status)
    from huggingface_hub import hf_hub_download

    consecutive_rate_limits = 0
    stopped_for_rate_limit = False
    for index, rel in enumerate(files, start=1):
        destination = root / rel
        if destination.exists() and not args.force:
            counters["already_exists"] += 1
            if args.resume_existing_queue and (index == len(files) or index % 50 == 0):
                write_queue(args.queue_out, root, "resume_remaining", files[index:])
            if index == 1 or index % 100 == 0:
                write_json(
                    {
                        "status": "running",
                        "updated_at": now(),
                        "config": config,
                        "last_path": rel,
                        "index": index,
                        "total": len(files),
                        "counters": dict(counters),
                    },
                    args.status,
                )
            continue
        event = {"updated_at": now(), "index": index, "total": len(files), "path": rel}
        for attempt in range(1, args.max_attempts_per_file + 1):
            try:
                event.update({"attempt": attempt, "status": "downloading"})
                append_jsonl(event, args.events)
                path = hf_hub_download(
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    revision=args.revision,
                    filename=rel,
                    local_dir=str(root),
                    force_download=args.force,
                    etag_timeout=30,
                )
                size = Path(path).stat().st_size
                counters["downloaded"] += 1
                consecutive_rate_limits = 0
                if args.resume_existing_queue and (index == len(files) or index % 50 == 0):
                    write_queue(args.queue_out, root, "resume_remaining", files[index:])
                append_jsonl(
                    {
                        "updated_at": now(),
                        "index": index,
                        "total": len(files),
                        "path": rel,
                        "status": "success",
                        "size": size,
                    },
                    args.events,
                )
                break
            except Exception as exc:  # noqa: BLE001
                message = str(exc)
                counters["attempt_failed"] += 1
                rate_limited = is_rate_limit_error(exc)
                if rate_limited:
                    counters["rate_limited_attempts"] += 1
                    consecutive_rate_limits += 1
                append_jsonl(
                    {
                        "updated_at": now(),
                        "index": index,
                        "total": len(files),
                        "path": rel,
                        "attempt": attempt,
                        "status": "failed_attempt",
                        "rate_limited": rate_limited,
                        "error": message,
                        "traceback": traceback.format_exc(),
                    },
                    args.events,
                )
                if (
                    rate_limited
                    and (
                        consecutive_rate_limits >= args.max_consecutive_rate_limits
                        or counters["rate_limited_attempts"] >= args.max_total_rate_limits
                    )
                ):
                    stopped_for_rate_limit = True
                    cooldown = retry_after_seconds(exc) or args.rate_limit_cooldown_sec
                    append_jsonl(
                        {
                            "updated_at": now(),
                            "index": index,
                            "total": len(files),
                            "path": rel,
                            "status": "stopping_for_rate_limit",
                            "cooldown_sec": cooldown,
                            "consecutive_rate_limits": consecutive_rate_limits,
                            "total_rate_limits": counters["rate_limited_attempts"],
                        },
                        args.events,
                    )
                    if args.resume_existing_queue:
                        write_queue(args.queue_out, root, "resume_remaining_after_rate_limit", files[index - 1 :])
                    break
                if attempt >= args.max_attempts_per_file:
                    counters["failed_files"] += 1
                    break
                wait = retry_after_seconds(exc) or (args.retry_wait_sec * (2 ** (attempt - 1)))
                if rate_limited:
                    wait = max(wait, args.retry_wait_sec * 3)
                wait *= random.uniform(0.8, 1.3)
                time.sleep(wait)
        if stopped_for_rate_limit:
            write_json(
                {
                    "status": "rate_limited",
                    "updated_at": now(),
                    "config": config,
                    "last_path": rel,
                    "index": index,
                    "total": len(files),
                    "counters": dict(counters),
                    "resume_command_hint": "rerun this script later; it rebuilds the missing-file queue from current files",
                },
                args.status,
            )
            break
        write_json(
            {
                "status": "running",
                "updated_at": now(),
                "config": config,
                "last_path": rel,
                "index": index,
                "total": len(files),
                "counters": dict(counters),
            },
            args.status,
        )
        time.sleep(args.sleep_sec)
    if stopped_for_rate_limit:
        return 75
    remaining = [] if args.skip_final_scan else missing_wavs(root, args.max_rows_per_singer, languages)
    write_json(
        {
            "status": "complete" if not counters.get("failed_files") else "complete_with_failures",
            "updated_at": now(),
            "config": config,
            "counters": dict(counters),
            "remaining_missing": None if args.skip_final_scan else len(remaining),
            "remaining_sample": [] if args.skip_final_scan else remaining[:20],
        },
        args.status,
    )
    return 0 if not counters.get("failed_files") else 1


if __name__ == "__main__":
    raise SystemExit(main())
