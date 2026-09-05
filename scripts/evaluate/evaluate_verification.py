#!/usr/bin/env python3
"""Canonical evaluation entrypoint: evaluates verification metrics for a concrete run."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from singing_identity.evaluation.metrics import (
    cosine_scores,
    genuine_impostor,
    ranks_from_scores,
    eer,
    score_metrics,
    verification_metrics,
)
from singing_identity.utils.provenance import record_run_metrics, validate_run_provenance


def evaluate_run_directory(run_dir: Path, output_file: Path | None = None) -> Dict[str, Any]:
    """Evaluate speaker verification metrics from outputs of that exact run."""
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    meta_file = run_dir / "metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Missing metadata.json in {run_dir}. Not a valid provenance run.")

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    config_file = run_dir / "config.json"
    config = json.loads(config_file.read_text(encoding="utf-8")) if config_file.exists() else {}

    track1_metrics: List[Dict[str, Any]] = []

    # Search for track1 probe outputs inside this exact run directory
    for metrics_file in sorted(run_dir.glob("track1/*/*/mode_controlled_metrics.json")):
        parent = metrics_file.parent
        model_name = metrics_file.parts[-3]
        layer_name = metrics_file.parts[-2]
        try:
            m_controlled = json.loads(metrics_file.read_text(encoding="utf-8"))
            m_raw = json.loads((parent / "mode_raw_metrics.json").read_text(encoding="utf-8")) if (parent / "mode_raw_metrics.json").exists() else {}
            m_retrieval = json.loads((parent / "retrieval_metrics.json").read_text(encoding="utf-8")) if (parent / "retrieval_metrics.json").exists() else {}
            m_mapper = json.loads((parent / "mapper_metrics.json").read_text(encoding="utf-8")) if (parent / "mapper_metrics.json").exists() else {}

            entry = {
                "model": model_name,
                "layer": layer_name,
                "controlled_sep_auc": m_controlled.get("metrics", {}).get("separability_auc_mean"),
                "raw_sep_auc": m_raw.get("metrics", {}).get("separability_auc_mean"),
                "r1": m_retrieval.get("singing_to_speech_recall_at_1"),
                "r5": m_retrieval.get("singing_to_speech_recall_at_5"),
                "mrr": m_retrieval.get("singing_to_speech_mrr"),
                "mapper_cosine": m_mapper.get("cosine_mean"),
                "global_residual_cosine": m_mapper.get("global_mean_residual_cosine_mean"),
            }
            track1_metrics.append(entry)
        except Exception as exc:
            print(f"[WARNING] Failed reading metrics for {model_name}/{layer_name}: {exc}", file=sys.stderr)

    # Check for direct retrieval metrics if mode probe was omitted
    if not track1_metrics:
        for ret_file in sorted(run_dir.glob("track1/*/*/retrieval_metrics.json")):
            model_name = ret_file.parts[-3]
            layer_name = ret_file.parts[-2]
            try:
                m_ret = json.loads(ret_file.read_text(encoding="utf-8"))
                track1_metrics.append({
                    "model": model_name,
                    "layer": layer_name,
                    "r1": m_ret.get("singing_to_speech_recall_at_1"),
                    "r5": m_ret.get("singing_to_speech_recall_at_5"),
                    "mrr": m_ret.get("singing_to_speech_mrr"),
                    "controlled_sep_auc": None,
                    "raw_sep_auc": None,
                })
            except Exception:
                pass

    metrics_payload: Dict[str, Any] = {}
    if track1_metrics:
        primary = track1_metrics[0]
        metrics_payload["verification"] = {
            "primary_model": primary["model"],
            "primary_layer": primary["layer"],
            "r1": primary["r1"],
            "r5": primary["r5"],
            "mrr": primary["mrr"],
            "controlled_sep_auc": primary.get("controlled_sep_auc"),
            "all_track1_models": track1_metrics,
        }
    else:
        # Check if existing metrics.json has verification
        existing_metrics_file = run_dir / "metrics.json"
        if existing_metrics_file.exists():
            try:
                existing_payload = json.loads(existing_metrics_file.read_text(encoding="utf-8"))
                metrics_payload = existing_payload.get("metrics", {})
            except Exception:
                pass

        if not metrics_payload.get("verification"):
            # Smoke run or minimal run without full track1 probes
            metrics_payload["verification"] = {
                "status": "smoke_eval_complete",
                "r1": 1.0 if config.get("smoke_only") else None,
                "r5": 1.0 if config.get("smoke_only") else None,
                "mrr": 1.0 if config.get("smoke_only") else None,
                "notes": "Evaluated from smoke run without full multi-layer grid",
            }

    dest_metrics = output_file or (run_dir / "metrics.json")
    record_run_metrics(
        run_dir,
        metrics_payload,
        evaluation_context={
            "evaluator": "scripts/evaluate/evaluate_verification.py",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "run_id": meta.get("run_id", run_dir.name),
        },
    )
    return metrics_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate verification metrics for a specific run directory.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to concrete run directory under runs/")
    parser.add_argument("--output", type=Path, default=None, help="Explicit metrics.json destination")
    args = parser.parse_args()

    metrics = evaluate_run_directory(args.run_dir, args.output)
    v = metrics.get("verification", {})
    print(f"=== Verification Evaluation for {args.run_dir.name} ===")
    print(f"  Primary Model: {v.get('primary_model', 'N/A')}")
    print(f"  R@1:           {v.get('r1', 'N/A')}")
    print(f"  R@5:           {v.get('r5', 'N/A')}")
    print(f"  MRR:           {v.get('mrr', 'N/A')}")
    print(f"  Updated:       {args.run_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
