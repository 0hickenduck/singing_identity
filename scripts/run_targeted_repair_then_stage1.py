#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORK_ROOT = Path("/localdisk/bowen/singing_identity")
DEFAULT_PYTHON = str(DEFAULT_REPO_ROOT / ".venv" / "bin" / "python")
DEFAULT_GTSINGER = str(DEFAULT_WORK_ROOT / "data" / "gtsinger_domain_eval")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def final_stage1_status(stage1_rc: int, stage1_run_root: Path) -> str:
    if stage1_rc != 0:
        return "stage1_failed"
    stage1_status = read_json(stage1_run_root / "status.json")
    status = stage1_status.get("status")
    if status in {"complete", "complete_with_failures"}:
        return status
    return "complete"


def run_logged(cmd: list[str], cwd: Path, log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        log.write(("\n$ " + " ".join(cmd) + "\n\n").encode("utf-8"))
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
        try:
            returncode = proc.wait()
        except KeyboardInterrupt:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            raise
        log.write((f"\nreturncode={returncode}\n").encode("utf-8"))
        return returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run targeted GTSinger wav repair, then launch Stage 1 on the repaired data.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--gtsinger-root", type=Path, default=Path(DEFAULT_GTSINGER))
    parser.add_argument("--status", type=Path, default=DEFAULT_WORK_ROOT / "status" / "targeted_then_stage1" / "status.json")
    parser.add_argument("--repair-status", type=Path, default=DEFAULT_WORK_ROOT / "status" / "gtsinger_targeted" / "status.json")
    parser.add_argument("--repair-events", type=Path, default=DEFAULT_WORK_ROOT / "status" / "gtsinger_targeted" / "events.jsonl")
    parser.add_argument("--repair-queue", type=Path, default=DEFAULT_WORK_ROOT / "status" / "gtsinger_targeted" / "missing_queue.json")
    parser.add_argument("--log", type=Path, default=DEFAULT_WORK_ROOT / "logs" / "targeted_repair_then_stage1.log")
    parser.add_argument("--max-rows-per-singer", type=int, default=5000)
    parser.add_argument("--repair-limit", type=int, default=0)
    parser.add_argument("--repair-sleep-sec", type=float, default=0.5)
    parser.add_argument("--repair-max-cycles", type=int, default=8)
    parser.add_argument("--repair-rate-limit-wait-sec", type=float, default=1800.0)
    parser.add_argument("--models", default="acoustic,wavlm,hubert,mert")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage1-run-root", type=Path, default=DEFAULT_WORK_ROOT / "runs" / "stage1_overnight_targeted_repaired_run")
    parser.add_argument("--stage1-report", type=Path, default=Path("results/stage1_overnight_targeted_repaired_report.md"))
    parser.add_argument("--stage1-feature-root", type=Path, default=DEFAULT_WORK_ROOT / "features" / "stage1_overnight_targeted_repaired")
    parser.add_argument("--smoke-job-timeout-sec", type=int, default=1200)
    parser.add_argument("--extract-job-timeout-sec", type=int, default=21600)
    parser.add_argument("--probe-job-timeout-sec", type=int, default=7200)
    parser.add_argument("--local-cache-root", type=Path, default=Path("/localdisk/bowen/.cache/singing_identity"))
    parser.add_argument("--allow-home-heavy-io", action="store_true")
    parser.add_argument("--force-stage1", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    cache_root = args.local_cache_root.resolve()
    for path in [
        cache_root,
        cache_root / "huggingface" / "datasets",
        cache_root / "huggingface" / "hub",
        cache_root / "huggingface" / "transformers",
        cache_root / "torch",
        cache_root / "pip",
        cache_root / "uv",
        cache_root / "nv" / "ComputeCache",
        cache_root / "torch_extensions",
        Path("/localdisk/bowen/tmp"),
    ]:
        path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("XDG_CACHE_HOME", str(cache_root))
    env.setdefault("HF_HOME", str(cache_root / "huggingface"))
    env.setdefault("HF_DATASETS_CACHE", str(cache_root / "huggingface" / "datasets"))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "huggingface" / "hub"))
    env.setdefault("TRANSFORMERS_CACHE", str(cache_root / "huggingface" / "transformers"))
    env.setdefault("TORCH_HOME", str(cache_root / "torch"))
    env.setdefault("PIP_CACHE_DIR", str(cache_root / "pip"))
    env.setdefault("UV_CACHE_DIR", str(cache_root / "uv"))
    env.setdefault("UV_LINK_MODE", "copy")
    env.setdefault("CUDA_CACHE_PATH", str(cache_root / "nv" / "ComputeCache"))
    env.setdefault("TORCH_EXTENSIONS_DIR", str(cache_root / "torch_extensions"))
    env.setdefault("TMPDIR", "/localdisk/bowen/tmp")
    config = {
        "repo_root": str(repo_root),
        "gtsinger_root": str(args.gtsinger_root.resolve()),
        "max_rows_per_singer": args.max_rows_per_singer,
        "repair_limit": args.repair_limit,
        "models": args.models,
        "stage1_run_root": str(args.stage1_run_root),
        "stage1_report": str(args.stage1_report),
        "stage1_feature_root": str(args.stage1_feature_root),
        "repair_max_cycles": args.repair_max_cycles,
        "repair_rate_limit_wait_sec": args.repair_rate_limit_wait_sec,
    }
    repair_rc = 0
    repair_status: dict[str, Any] = {}
    for repair_cycle in range(1, max(1, args.repair_max_cycles) + 1):
        write_json(
            {
                "status": "running_repair",
                "updated_at": now(),
                "config": config,
                "repair_cycle": repair_cycle,
            },
            args.status,
        )
        repair_cmd = [
            args.python,
            "scripts/data_prep/repair_gtsinger_missing_wavs.py",
            "--root",
            str(args.gtsinger_root),
            "--max-rows-per-singer",
            str(args.max_rows_per_singer),
            "--sleep-sec",
            str(args.repair_sleep_sec),
            "--status",
            str(args.repair_status),
            "--events",
            str(args.repair_events),
            "--queue-out",
            str(args.repair_queue),
            "--local-cache-root",
            str(args.local_cache_root.resolve() / "hf_targeted"),
            "--resume-existing-queue",
            "--skip-final-scan",
        ]
        if args.repair_limit > 0:
            repair_cmd.extend(["--limit", str(args.repair_limit)])
        repair_rc = run_logged(repair_cmd, repo_root, args.log, env)
        repair_status = read_json(args.repair_status)
        if repair_status.get("status") != "rate_limited" and repair_rc != 75:
            break
        if repair_cycle >= args.repair_max_cycles:
            break
        write_json(
            {
                "status": "waiting_after_rate_limit",
                "updated_at": now(),
                "config": config,
                "repair_cycle": repair_cycle,
                "repair_returncode": repair_rc,
                "repair_status": repair_status,
                "next_retry_sec": args.repair_rate_limit_wait_sec,
                "note": "Repair will rebuild the missing-file queue and resume after cooldown.",
            },
            args.status,
        )
        time.sleep(args.repair_rate_limit_wait_sec)
    write_json(
        {
            "status": "running_stage1",
            "updated_at": now(),
            "config": config,
            "repair_returncode": repair_rc,
            "repair_status": repair_status,
            "note": "Stage 1 starts even if targeted repair had partial file failures; the manifest records remaining missing wavs.",
        },
        args.status,
    )
    stage1_cmd = [
        args.python,
        "scripts/run_stage1_overnight.py",
        "--models",
        args.models,
        "--python",
        args.python,
        "--device",
        args.device,
        "--gtsinger-root",
        str(args.gtsinger_root),
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
        "--fast-no-wav-stats",
    ]
    if args.force_stage1:
        stage1_cmd.append("--force")
    if args.allow_home_heavy_io:
        stage1_cmd.append("--allow-home-heavy-io")
    stage1_rc = run_logged(stage1_cmd, repo_root, args.log, env)
    final_status = final_stage1_status(stage1_rc, args.stage1_run_root)
    write_json(
        {
            "status": final_status,
            "updated_at": now(),
            "config": config,
            "repair_returncode": repair_rc,
            "repair_status": repair_status,
            "stage1_returncode": stage1_rc,
            "stage1_report": str(args.stage1_report.resolve()),
        },
        args.status,
    )
    return stage1_rc


if __name__ == "__main__":
    raise SystemExit(main())
