#!/usr/bin/env python3
"""Standardized benchmark entrypoint for Stage 1 experiments with automatic provenance."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from singing_identity.utils.provenance import (
    init_run_directory,
    get_data_provenance,
    record_run_metrics,
    validate_run_provenance,
)
from singing_identity.runner import Stage1Runner, parse_args as parse_stage1_args


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Stage 1 benchmark experiments with full machine-readable provenance."
    )
    # Re-declare arguments compatible with Stage1Runner
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-root", type=Path, default=None, help="Explicit run root. Defaults to runs/<exp>/<approach>/<run_id>")
    parser.add_argument("--experiment", default="stage1_multimodel")
    parser.add_argument("--approach", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-reproducibility", action="store_true", help="Refuse execution if working directory is dirty")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument("--gtsinger-root", type=Path, default=Path("/localdisk/bowen/singing_identity/data/GTSinger"))
    parser.add_argument("--existing-manifest", type=Path)
    parser.add_argument("--existing-pairs", type=Path)
    parser.add_argument("--existing-phone-examples", type=Path)
    parser.add_argument("--existing-technique-pairs", type=Path)
    parser.add_argument("--existing-manifest-report", type=Path)
    parser.add_argument("--models", default="acoustic,wavlm,hubert,mert")
    parser.add_argument("--contentvec-model")
    parser.add_argument("--contentvec-layer", default="12")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--max-rows-per-singer", type=int, default=5000)
    parser.add_argument("--smoke-max-rows-per-singer", type=int, default=200)
    parser.add_argument("--max-phone-pairs-per-group", type=int, default=2000)
    parser.add_argument("--fast-no-wav-stats", action="store_true")
    parser.add_argument("--job-timeout-sec", type=int, default=7200)
    parser.add_argument("--smoke-job-timeout-sec", type=int, default=1800)
    parser.add_argument("--extract-job-timeout-sec", type=int, default=21600)
    parser.add_argument("--probe-job-timeout-sec", type=int, default=7200)
    parser.add_argument("--local-cache-root", type=Path, default=Path("/localdisk/bowen/.cache"))
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-home-heavy-io", action="store_true")
    args = parser.parse_args()

    # Determine approach name
    approach_name = args.approach or (args.models.replace(",", "_") if len(args.models.split(",")) <= 2 else "multimodel_suite")

    # Resolve manifest path
    manifest_path = args.existing_manifest or Path("data/processed/gtsinger_utterances.jsonl")

    # Capture data provenance
    data_prov = get_data_provenance(
        dataset_name="synthetic_gtsinger" if args.synthetic else "gtsinger",
        manifest_path=manifest_path,
        feature_set=approach_name,
        feature_root=args.feature_root,
        extractor=args.models.split(",")[0],
    )

    # Resolve config
    resolved_config = {
        "experiment": args.experiment,
        "approach": approach_name,
        "seed": args.seed,
        "models": args.models.split(","),
        "synthetic": args.synthetic,
        "smoke_only": args.smoke_only,
        "device": args.device,
        "folds": args.folds,
        "bootstrap": args.bootstrap,
        "max_rows_per_singer": args.max_rows_per_singer,
        "manifest_path": str(manifest_path),
        "strict_reproducibility": args.strict_reproducibility,
    }

    # Initialize standardized run directory
    base_runs = ROOT / "runs"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}_seed{args.seed}"

    if args.run_root is None:
        run_dir, metadata = init_run_directory(
            base_runs_dir=base_runs,
            experiment_name=args.experiment,
            approach_name=approach_name,
            run_id=run_id,
            seed=args.seed,
            resolved_config=resolved_config,
            data_provenance=data_prov,
            strict_git=args.strict_reproducibility,
            repo_root=ROOT,
        )
        args.run_root = run_dir
    else:
        run_dir = args.run_root.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        exp_name = args.experiment
        appr_name = approach_name
        if len(run_dir.parts) >= 3 and run_dir.parent.name and run_dir.parent.parent.name:
            appr_name = run_dir.parent.name
            exp_name = run_dir.parent.parent.name
        _, metadata = init_run_directory(
            base_runs_dir=base_runs,
            experiment_name=exp_name,
            approach_name=appr_name,
            run_id=run_dir.name,
            seed=args.seed,
            resolved_config=resolved_config,
            data_provenance=data_prov,
            strict_git=args.strict_reproducibility,
            repo_root=ROOT,
            target_run_dir=run_dir,
        )

    if args.report is None:
        args.report = run_dir / "report.md"

    if args.feature_root is None:
        args.feature_root = run_dir / "features"

    print(f"=== Initialized Provenance Run: {run_id} ===")
    print(f"  Run Directory: {run_dir}")
    print(f"  Git Commit:    {metadata['git']['commit']} (dirty={metadata['git']['dirty']})")
    print(f"  Seed:          {args.seed}")
    print(f"  Data Manifest: {data_prov['manifest']} (sha256={data_prov['manifest_sha256'][:10]}...)")

    # Run Stage 1 execution
    runner = Stage1Runner(args)
    rc = runner.run()

    # For synthetic runs, record the actual consumed synthetic manifest and its actual SHA-256
    if args.synthetic:
        consumed_manifest = runner.manifest if runner.manifest.exists() else (run_dir / "manifests" / "gtsinger_utterances.jsonl")
        if not consumed_manifest.exists():
            synth_candidates = list(run_dir.glob("synthetic/**/utterances.jsonl"))
            if synth_candidates:
                consumed_manifest = synth_candidates[0]
        try:
            manifest_to_record = consumed_manifest.relative_to(ROOT)
        except ValueError:
            manifest_to_record = consumed_manifest
        data_prov = get_data_provenance(
            dataset_name="synthetic_gtsinger",
            manifest_path=str(manifest_to_record),
            feature_set=approach_name,
            feature_root=args.feature_root,
            extractor=args.models.split(",")[0],
        )
        metadata["data"] = data_prov
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        resolved_config["manifest_path"] = str(manifest_to_record)
        (run_dir / "config.json").write_text(json.dumps(resolved_config, indent=2) + "\n", encoding="utf-8")

    # Clean up duplicate legacy config if present
    legacy_cfg = run_dir / "resolved_config.json"
    if legacy_cfg.exists():
        legacy_cfg.unlink()

    # Extract metrics from run outputs and record structured metrics.json
    metrics: dict[str, Any] = {}
    track1_metrics: list[dict[str, Any]] = []
    for metrics_file in sorted(run_dir.glob("track1/*/*/mode_controlled_metrics.json")):
        parent = metrics_file.parent
        try:
            m_controlled = json.loads(metrics_file.read_text(encoding="utf-8"))
            m_raw = json.loads((parent / "mode_raw_metrics.json").read_text(encoding="utf-8")) if (parent / "mode_raw_metrics.json").exists() else {}
            m_retrieval = json.loads((parent / "retrieval_metrics.json").read_text(encoding="utf-8")) if (parent / "retrieval_metrics.json").exists() else {}
            m_mapper = json.loads((parent / "mapper_metrics.json").read_text(encoding="utf-8")) if (parent / "mapper_metrics.json").exists() else {}

            entry = {
                "model": metrics_file.parts[-3],
                "layer": metrics_file.parts[-2],
                "controlled_sep_auc": m_controlled.get("metrics", {}).get("separability_auc_mean"),
                "raw_sep_auc": m_raw.get("metrics", {}).get("separability_auc_mean"),
                "r1": m_retrieval.get("singing_to_speech_recall_at_1"),
                "r5": m_retrieval.get("singing_to_speech_recall_at_5"),
                "mrr": m_retrieval.get("singing_to_speech_mrr"),
                "mapper_cosine": m_mapper.get("cosine_mean"),
                "global_residual_cosine": m_mapper.get("global_mean_residual_cosine_mean"),
            }
            track1_metrics.append(entry)
        except Exception:
            pass

    if track1_metrics:
        # Take primary model metrics (e.g. wavlm or first model)
        primary = track1_metrics[0]
        metrics["verification"] = {
            "primary_model": primary["model"],
            "primary_layer": primary["layer"],
            "r1": primary["r1"],
            "r5": primary["r5"],
            "mrr": primary["mrr"],
            "controlled_sep_auc": primary["controlled_sep_auc"],
            "all_track1_models": track1_metrics,
        }
    else:
        # Check for status
        metrics["execution_status"] = {"returncode": rc, "success": rc == 0}

    record_run_metrics(run_dir, metrics, evaluation_context={"synthetic": args.synthetic, "smoke_only": args.smoke_only})

    # Validate provenance
    val_report = validate_run_provenance(run_dir)
    print(f"=== Provenance Validation: {val_report['status'].upper()} ===")
    if val_report["warnings"]:
        for w in val_report["warnings"]:
            print(f"  [WARNING] {w}")
    if val_report["errors"]:
        for e in val_report["errors"]:
            print(f"  [ERROR] {e}")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
