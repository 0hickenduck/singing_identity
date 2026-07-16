#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_REVISION = "4426c862beed558b7e1cb8a4dce7e8c0c83bb208"
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORK_ROOT = Path("/localdisk/bowen/singing_identity")
DEFAULT_PYTHON = str(DEFAULT_REPO_ROOT / ".venv" / "bin" / "python")
DEFAULT_GTSINGER = str(DEFAULT_WORK_ROOT / "data" / "gtsinger_domain_eval")


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def token_available(check_token_files: bool = False, watch_token_file: Path | None = None) -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        return True
    if watch_token_file and watch_token_file.exists() and watch_token_file.stat().st_size > 0:
        token = watch_token_file.read_text(encoding="utf-8").strip()
        if token:
            os.environ["HF_TOKEN"] = token
            return True
    if not check_token_files:
        return False
    home = Path.home()
    token_paths = [
        home / ".cache" / "huggingface" / "token",
        home / ".cache" / "huggingface" / "stored_tokens",
    ]
    return any(path.exists() and path.stat().st_size > 0 for path in token_paths)


def count_local_wavs(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.wav"))


def repair_dataset(args: argparse.Namespace, status_path: Path) -> bool:
    local_dir = args.local_dir.resolve()
    snapshot_download = None
    for attempt in range(1, args.max_download_attempts + 1):
        if args.require_token:
            write_json(
                {
                    "status": "checking_hf_token",
                    "updated_at": now(),
                    "attempt": attempt,
                    "check_token_files": args.check_token_files,
                    "watch_token_file": str(args.watch_token_file) if args.watch_token_file else "",
                },
                status_path,
            )
        if args.require_token and not token_available(args.check_token_files, args.watch_token_file):
            message = "No HF_TOKEN/HUGGING_FACE_HUB_TOKEN was present in this supervisor process."
            if args.watch_token_file:
                message += f" No local token was found at {args.watch_token_file}."
            if args.check_token_files:
                message += " No Hugging Face token file was found either."
            else:
                message += " NFS token-file checks are disabled to avoid stalls."
            write_json(
                {
                    "status": "waiting_for_hf_token",
                    "updated_at": now(),
                    "message": message,
                    "attempt": attempt,
                    "watch_token_file": str(args.watch_token_file) if args.watch_token_file else "",
                    "next_retry_sec": args.retry_wait_sec,
                },
                status_path,
            )
            time.sleep(args.retry_wait_sec)
            continue
        if snapshot_download is None:
            from huggingface_hub import snapshot_download as hf_snapshot_download  # Imported lazily; this can be slow on NFS.

            snapshot_download = hf_snapshot_download
        write_json(
            {
                "status": "downloading",
                "updated_at": now(),
                "attempt": attempt,
                "repo_id": args.repo_id,
                "revision": args.revision,
                "local_dir": str(local_dir),
                "allow_patterns": args.allow_patterns,
                "max_workers": args.max_workers,
            },
            status_path,
        )
        try:
            path = snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                revision=args.revision,
                local_dir=str(local_dir),
                allow_patterns=args.allow_patterns,
                ignore_patterns=args.ignore_patterns,
                max_workers=args.max_workers,
            )
            wav_count = count_local_wavs(local_dir)
            metadata_count = sum(1 for _ in (local_dir / "processed").glob("*/metadata.json"))
            write_json(
                {
                    "status": "download_complete",
                    "updated_at": now(),
                    "local_path": path,
                    "wav_count": wav_count,
                    "metadata_count": metadata_count,
                    "attempt": attempt,
                },
                status_path,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            is_rate_limit = "429" in str(exc) or "Too Many Requests" in str(exc)
            wait_sec = args.retry_wait_sec * min(attempt, 6)
            write_json(
                {
                    "status": "download_retry_wait" if is_rate_limit else "download_failed_retry_wait",
                    "updated_at": now(),
                    "attempt": attempt,
                    "error": str(exc),
                    "traceback": tb,
                    "next_retry_sec": wait_sec,
                },
                status_path,
            )
            if attempt >= args.max_download_attempts:
                write_json(
                    {
                        "status": "failed",
                        "updated_at": now(),
                        "stage": "download",
                        "attempt": attempt,
                        "error": str(exc),
                        "traceback": tb,
                    },
                    status_path,
                )
                return False
            time.sleep(wait_sec)
    return False


def run_repaired_stage1(args: argparse.Namespace, status_path: Path, log_path: Path) -> int:
    cmd = [
        args.python,
        "scripts/run_stage1_overnight.py",
        "--models",
        args.models,
        "--python",
        args.python,
        "--device",
        args.device,
        "--gtsinger-root",
        str(args.local_dir),
        "--run-root",
        str(args.stage1_run_root),
        "--report",
        str(args.stage1_report),
        "--feature-root",
        str(args.stage1_feature_root),
        "--max-rows-per-singer",
        str(args.max_rows_per_singer),
        "--smoke-job-timeout-sec",
        str(args.smoke_job_timeout_sec),
        "--extract-job-timeout-sec",
        str(args.extract_job_timeout_sec),
        "--probe-job-timeout-sec",
        str(args.probe_job_timeout_sec),
        "--local-cache-root",
        str(args.local_cache_root),
    ]
    if args.force_stage1:
        cmd.append("--force")
    if args.allow_home_heavy_io:
        cmd.append("--allow-home-heavy-io")
    write_json(
        {
            "status": "running_repaired_stage1",
            "updated_at": now(),
            "cmd": cmd,
            "stage1_log": str(log_path),
            "stage1_run_root": str(args.stage1_run_root),
            "stage1_report": str(args.stage1_report),
        },
        status_path,
    )
    with log_path.open("ab", buffering=0) as log:
        log.write(("$ " + " ".join(cmd) + "\n\n").encode("utf-8"))
        proc = subprocess.run(cmd, cwd=args.repo_root, stdout=log, stderr=subprocess.STDOUT)
    final_status = "complete" if proc.returncode == 0 else "stage1_failed"
    write_json(
        {
            "status": final_status,
            "updated_at": now(),
            "returncode": proc.returncode,
            "stage1_log": str(log_path),
            "stage1_run_root": str(args.stage1_run_root),
            "stage1_report": str(args.stage1_report),
        },
        status_path,
    )
    return proc.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair GTSinger download, then rerun Stage 1 on the repaired data.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--repo-id", default="GTSinger/GTSinger")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_WORK_ROOT / "data" / "gtsinger_domain_eval")
    parser.add_argument("--status-dir", type=Path, default=DEFAULT_WORK_ROOT / "status" / "gtsinger_redownload")
    parser.add_argument("--allow-patterns", nargs="+", default=["*.wav", "processed/**"])
    parser.add_argument("--ignore-patterns", nargs="+", default=["*.DS_Store"])
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--max-download-attempts", type=int, default=96)
    parser.add_argument("--retry-wait-sec", type=int, default=1800)
    parser.add_argument("--require-token", action="store_true")
    parser.add_argument("--check-token-files", action="store_true")
    parser.add_argument("--watch-token-file", type=Path, default=Path("/tmp/singing_identity_hf_token"))
    parser.add_argument("--models", default="acoustic,wavlm,hubert,mert")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage1-run-root", type=Path, default=DEFAULT_WORK_ROOT / "runs" / "stage1_overnight_repaired_run")
    parser.add_argument("--stage1-report", type=Path, default=Path("results/stage1_overnight_repaired_report.md"))
    parser.add_argument("--stage1-feature-root", type=Path, default=DEFAULT_WORK_ROOT / "features" / "stage1_overnight_repaired")
    parser.add_argument("--max-rows-per-singer", type=int, default=5000)
    parser.add_argument("--smoke-job-timeout-sec", type=int, default=900)
    parser.add_argument("--extract-job-timeout-sec", type=int, default=21600)
    parser.add_argument("--probe-job-timeout-sec", type=int, default=7200)
    parser.add_argument("--local-cache-root", type=Path, default=Path("/localdisk/bowen/.cache/singing_identity"))
    parser.add_argument("--allow-home-heavy-io", action="store_true")
    parser.add_argument("--force-stage1", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.local_dir = args.local_dir.resolve()
    args.status_dir = args.status_dir.resolve()
    args.stage1_run_root = args.stage1_run_root.resolve()
    args.stage1_report = args.stage1_report.resolve()
    args.stage1_feature_root = args.stage1_feature_root.resolve()
    args.local_cache_root = args.local_cache_root.resolve()
    args.watch_token_file = args.watch_token_file.resolve() if args.watch_token_file else None
    for path in [
        args.local_cache_root,
        args.local_cache_root / "huggingface" / "datasets",
        args.local_cache_root / "huggingface" / "hub",
        args.local_cache_root / "huggingface" / "transformers",
        args.local_cache_root / "torch",
        args.local_cache_root / "pip",
        args.local_cache_root / "uv",
        args.local_cache_root / "nv" / "ComputeCache",
        args.local_cache_root / "torch_extensions",
        Path("/localdisk/bowen/tmp"),
    ]:
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(args.local_cache_root))
    os.environ.setdefault("HF_HOME", str(args.local_cache_root / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(args.local_cache_root / "huggingface" / "datasets"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(args.local_cache_root / "huggingface" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(args.local_cache_root / "huggingface" / "transformers"))
    os.environ.setdefault("TORCH_HOME", str(args.local_cache_root / "torch"))
    os.environ.setdefault("PIP_CACHE_DIR", str(args.local_cache_root / "pip"))
    os.environ.setdefault("UV_CACHE_DIR", str(args.local_cache_root / "uv"))
    os.environ.setdefault("UV_LINK_MODE", "copy")
    os.environ.setdefault("CUDA_CACHE_PATH", str(args.local_cache_root / "nv" / "ComputeCache"))
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(args.local_cache_root / "torch_extensions"))
    os.environ.setdefault("TMPDIR", "/localdisk/bowen/tmp")
    args.status_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.status_dir / "status.json"
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    write_json({"status": "started", "updated_at": now(), "config": config}, status_path)
    write_json(config, args.status_dir / "config.json")
    ok = repair_dataset(args, status_path)
    if not ok:
        return 2
    return run_repaired_stage1(args, status_path, args.status_dir / "stage1_repaired_stdout.log")


if __name__ == "__main__":
    raise SystemExit(main())
