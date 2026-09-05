"""Experiment provenance, Git state tracking, and reproducibility helpers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 hex digest of a file in streaming chunks."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    hasher = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_provenance(repo_root: Optional[Path] = None, strict: bool = False) -> Dict[str, Any]:
    """Capture full Git provenance: 40-char SHA, branch, dirty status, diff patch, and remote."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]

    git_dir = repo_root / ".git"
    if not git_dir.exists() and not (repo_root / ".git").is_file():
        # Check if parent has git
        res = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "--git-dir"],
                             capture_output=True, text=True)
        if res.returncode != 0:
            return {
                "commit": "unknown",
                "branch": "unknown",
                "dirty": True,
                "diff": None,
                "remote_url": None,
                "status": "no_git_repository",
            }

    # 1. Full 40-character commit SHA
    commit_res = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=False)
    commit = commit_res.stdout.strip() if commit_res.returncode == 0 else "unknown"

    # 2. Branch or detached-HEAD status
    branch_res = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True, check=False)
    branch = branch_res.stdout.strip() if branch_res.returncode == 0 else "unknown"

    # 3. Dirty working tree status
    status_res = subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain"],
                                capture_output=True, text=True, check=False)
    dirty_lines = status_res.stdout.strip()
    is_dirty = bool(dirty_lines)

    # 4. Remote URL
    remote_res = subprocess.run(["git", "-C", str(repo_root), "config", "--get", "remote.origin.url"],
                                capture_output=True, text=True, check=False)
    remote_url = remote_res.stdout.strip() if remote_res.returncode == 0 else None

    # 5. Diff / patch capture
    diff_text = None
    if is_dirty:
        diff_res = subprocess.run(["git", "-C", str(repo_root), "diff", "HEAD"],
                                  capture_output=True, text=True, check=False)
        diff_text = diff_res.stdout if diff_res.returncode == 0 else ""
        # Also append untracked files list if any
        untracked = [line[3:] for line in dirty_lines.splitlines() if line.startswith("?? ")]
        if untracked:
            diff_text += "\n\n# Untracked files:\n" + "\n".join(f"#   {f}" for f in untracked)

        if strict:
            raise RuntimeError(
                f"Strict reproducibility mode: repository at '{repo_root}' has uncommitted changes. "
                "Commit your changes or disable strict reproducibility to proceed."
            )
        print(f"[PROVENANCE WARNING] Repository has uncommitted changes! Working tree dirty status recorded and diff captured.", file=sys.stderr)

    return {
        "commit": commit,
        "branch": branch,
        "dirty": is_dirty,
        "diff": diff_text,
        "remote_url": remote_url,
    }


def resolve_feature_root(feature_root: Optional[Path | str] = None) -> Path:
    """Resolve active feature root using CLI arg, env var, or local scratch fallback."""
    if feature_root:
        return Path(feature_root).resolve()
    env_root = os.environ.get("SINGING_IDENTITY_FEATURE_ROOT")
    if env_root:
        return Path(env_root).resolve()
    # Check default node-local scratch
    local_scratch = Path("/localdisk/bowen/singing_identity/features")
    if local_scratch.exists():
        return local_scratch.resolve()
    # Check in-repo fallback
    repo_features = Path(__file__).resolve().parents[3] / "experiments" / "shared_features"
    if repo_features.exists():
        return repo_features.resolve()
    return local_scratch


def get_data_provenance(
    dataset_name: str,
    manifest_path: Path | str,
    feature_set: Optional[str] = None,
    feature_root: Optional[Path | str] = None,
    extractor: Optional[str] = None,
    checkpoint: Optional[str] = None,
    layer: Optional[str | int] = None,
) -> Dict[str, Any]:
    """Capture data and feature provenance separately from Git."""
    p = Path(manifest_path)
    m_path = p.resolve()
    sha = compute_file_sha256(m_path)
    manifest_sha256 = sha if sha else None

    resolved_feat_root = resolve_feature_root(feature_root)
    feature_manifest_sha256: Optional[str] = None
    feature_provenance = "not_applicable"
    feature_version = "default"

    if feature_set or extractor:
        extractor_name = extractor or feature_set or ""
        ckpt_name = checkpoint or ""
        layer_str = str(layer) if layer is not None else ""

        # Search for metadata.json in feature directory
        candidate_meta = resolved_feat_root / extractor_name
        if ckpt_name:
            candidate_meta = candidate_meta / ckpt_name
        if layer_str:
            candidate_meta = candidate_meta / layer_str
        meta_file = candidate_meta / "metadata.json"

        if meta_file.exists():
            feature_manifest_sha256 = compute_file_sha256(meta_file)
            feature_provenance = "cached"
            try:
                meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
                feature_version = str(meta_data.get("checkpoint_hash", meta_data.get("feature_dim", "1.0")))
            except Exception:
                feature_version = "1.0"
        else:
            feature_provenance = "not_applicable"
            # Check if directory exists at all
            if not candidate_meta.exists() and not (resolved_feat_root / extractor_name).exists():
                reconstruct_cmd = (
                    f"uv run python scripts/data/extract_features.py "
                    f"--approach configs/approaches/{extractor_name}.json "
                    f"--manifest {manifest_path} --feature-root {resolved_feat_root}"
                )
                print(
                    f"[PROVENANCE NOTICE] Feature cache at '{candidate_meta}' is not pre-cached. "
                    f"To reconstruct it, execute:\n  {reconstruct_cmd}",
                    file=sys.stderr,
                )

    return {
        "dataset": dataset_name,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256 if manifest_sha256 else None,
        "feature_set": feature_set or extractor or "raw",
        "feature_version": feature_version or "default",
        "feature_provenance": feature_provenance,
        "feature_manifest_sha256": feature_manifest_sha256,
        "feature_root": str(resolved_feat_root),
    }


def get_execution_context(seed: Optional[int] = None, command_str: Optional[str] = None) -> Dict[str, Any]:
    """Capture execution context: environment, packages, machine, command."""
    # Capture relevant package versions without bloat
    package_versions: Dict[str, str] = {}
    for pkg in ("numpy", "torch", "scipy", "sklearn", "transformers"):
        try:
            mod = __import__(pkg)
            package_versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    # Capture CUDA details
    cuda_info: Dict[str, Any] = {"available": False}
    try:
        import torch
        if torch.cuda.is_available():
            cuda_info = {
                "available": True,
                "version": getattr(torch.version, "cuda", "unknown"),
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            }
    except Exception:
        pass

    cmd = command_str if command_str else (sys.executable + " " + " ".join(sys.argv))
    entrypoint = sys.argv[0] if sys.argv else "interactive"

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": cmd,
        "entrypoint": entrypoint,
        "seed": seed,
        "hostname": socket.gethostname(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": package_versions,
        "cuda": cuda_info,
    }


def init_run_directory(
    base_runs_dir: Path | str,
    experiment_name: str,
    approach_name: str,
    run_id: Optional[str] = None,
    seed: int = 42,
    resolved_config: Optional[Dict[str, Any]] = None,
    data_provenance: Optional[Dict[str, Any]] = None,
    strict_git: bool = False,
    command_str: Optional[str] = None,
    repo_root: Optional[Path] = None,
    target_run_dir: Optional[Path | str] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Initialize a standardized run folder: config.json, metadata.json, command.txt, logs/, artifacts/."""
    now_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if target_run_dir is not None:
        run_dir = Path(target_run_dir)
        if not run_id:
            run_id = run_dir.name
    else:
        if not run_id:
            run_id = f"{now_str}_seed{seed}"
        run_dir = Path(base_runs_dir) / experiment_name / approach_name / run_id
    logs_dir = run_dir / "logs"
    artifacts_dir = run_dir / "artifacts"
    logs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    git_prov = get_git_provenance(repo_root=repo_root, strict=strict_git)
    exec_ctx = get_execution_context(seed=seed, command_str=command_str)

    # If git was dirty, write git patch into artifacts
    if git_prov.get("diff"):
        patch_path = artifacts_dir / "uncommitted.patch"
        patch_path.write_text(git_prov["diff"], encoding="utf-8")

    # Write command.txt
    (run_dir / "command.txt").write_text(exec_ctx["command"] + "\n", encoding="utf-8")

    # Write fully resolved config.json
    config_dict = resolved_config or {}
    (run_dir / "config.json").write_text(json.dumps(config_dict, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Build metadata
    metadata = {
        "run_id": run_id,
        "experiment": experiment_name,
        "approach": approach_name,
        "seed": seed,
        "created_at": exec_ctx["created_at"],
        "provenance_status": "complete",
        "git": {
            "commit": git_prov["commit"],
            "branch": git_prov["branch"],
            "dirty": git_prov["dirty"],
            "remote_url": git_prov["remote_url"],
            "patch_artifact": "artifacts/uncommitted.patch" if git_prov.get("diff") else None,
        },
        "data": data_provenance or {},
        "execution": exec_ctx,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return run_dir, metadata


def record_run_metrics(
    run_dir: Path | str,
    metrics: Dict[str, Any],
    evaluation_context: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save metrics.json with stable metric schema."""
    r_dir = Path(run_dir)
    payload = {
        "metrics": metrics,
        "evaluation_context": evaluation_context or {},
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = r_dir / "metrics.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def validate_run_provenance(run_dir: Path | str) -> Dict[str, Any]:
    """Validate that a run satisfies the provenance invariant."""
    r_dir = Path(run_dir)
    report: Dict[str, Any] = {
        "run_dir": str(r_dir),
        "run_id": r_dir.name,
        "status": "complete",
        "valid": True,
        "errors": [],
        "warnings": [],
    }

    # 1. config.json
    cfg_file = r_dir / "config.json"
    if not cfg_file.exists():
        # Check if resolved_config.json exists (historical stage1)
        if (r_dir / "resolved_config.json").exists():
            report["warnings"].append("Uses legacy 'resolved_config.json' instead of 'config.json'")
        else:
            report["errors"].append("Missing config.json")

    # 2. metadata.json
    meta_file = r_dir / "metadata.json"
    if not meta_file.exists():
        # Check if experiment_card.yaml exists (historical)
        if (r_dir / "experiment_card.yaml").exists():
            report["status"] = "partial"
            report["warnings"].append("Historical run: uses 'experiment_card.yaml' (missing dirty status, exact command, and seed)")
            return report
        else:
            report["errors"].append("Missing metadata.json")
            report["status"] = "unknown"
            report["valid"] = False
            return report

    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception as exc:
        report["errors"].append(f"Invalid metadata.json JSON: {exc}")
        report["status"] = "unknown"
        report["valid"] = False
        return report

    # Check Git provenance
    git_info = meta.get("git", {})
    commit = git_info.get("commit", "")
    if not commit or commit == "unknown" or len(commit) < 7:
        report["errors"].append("Missing or invalid git commit SHA in metadata")
    if "dirty" not in git_info:
        report["errors"].append("Missing git 'dirty' boolean in metadata")
    elif git_info["dirty"]:
        patch_file = r_dir / "artifacts" / "uncommitted.patch"
        legacy_patch = r_dir / "artifacts" / "git_patch.diff"
        if not patch_file.exists() and not legacy_patch.exists() and not git_info.get("diff"):
            report["warnings"].append("Working tree was dirty, but no diff/patch artifact was saved")

    # Check Data provenance
    data_info = meta.get("data", {})
    if not data_info.get("manifest"):
        report["warnings"].append("No dataset manifest recorded in data provenance")
    if not data_info.get("manifest_sha256"):
        report["warnings"].append("No manifest SHA-256 fingerprint recorded")

    # Check Metrics
    metric_file = r_dir / "metrics.json"
    if not metric_file.exists():
        # Check for alternative metric files
        found_metrics = list(r_dir.glob("*metrics*.json")) + list(r_dir.glob("*.csv"))
        if not found_metrics:
            report["errors"].append("Missing metrics.json")
            report["valid"] = False
        else:
            report["warnings"].append(f"Uses non-standard metric files: {[f.name for f in found_metrics]}")

    if report["errors"]:
        report["valid"] = False
        report["status"] = "invalid"

    return report
