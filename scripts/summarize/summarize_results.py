#!/usr/bin/env python3
"""Aggregates benchmark results directly from run-level provenance records.

Canonical workflow:
    config + code + data -> run -> run-level metrics -> summarizer -> summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from singing_identity.utils.provenance import validate_run_provenance


def parse_historical_card(card_path: Path) -> Dict[str, Any]:
    """Parse historical experiment_card.yaml without inventing missing fields."""
    info: Dict[str, Any] = {}
    if not card_path.exists():
        return info
    for line in card_path.read_text(encoding="utf-8").splitlines():
        if ":" in line and not line.strip().startswith("-"):
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip().strip('"').strip("'")
    return info


def discover_and_aggregate_runs(
    runs_dir: Path,
    results_dir: Path,
    strict_provenance: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Scan runs and results, validate provenance, and extract benchmark metrics."""
    rows: List[Dict[str, Any]] = []
    companion_provenance: Dict[str, Any] = {}

    # 1. Discover modern runs under runs/
    for meta_file in sorted(runs_dir.glob("*/*/*/metadata.json")):
        run_folder = meta_file.parent
        val = validate_run_provenance(run_folder)

        if strict_provenance and not val["valid"]:
            print(f"[REJECTED STRICT RUN] {run_folder.name} failed provenance: {val['errors']}", file=sys.stderr)
            continue

        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            metrics_file = run_folder / "metrics.json"
            metrics_payload = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else {}
            metrics = metrics_payload.get("metrics", {})

            verif = metrics.get("verification", {})
            prob = metrics.get("probing", {})

            run_id = meta.get("run_id", run_folder.name)
            experiment = meta.get("experiment", run_folder.parents[1].name)
            approach = meta.get("approach", run_folder.parent.name)
            git_info = meta.get("git", {})
            data_info = meta.get("data", {})
            exec_info = meta.get("execution", {})

            r1 = verif.get("r1")
            r5 = verif.get("r5")
            mrr = verif.get("mrr")
            margin = verif.get("margin", verif.get("global_residual_cosine"))

            row = {
                "experiment": experiment,
                "approach": approach,
                "dataset": data_info.get("dataset", "unknown"),
                "run_id": run_id,
                "git_commit": git_info.get("commit", "unknown"),
                "git_dirty": str(git_info.get("dirty", "unknown")),
                "seed": str(meta.get("seed", "unknown")),
                "provenance_status": "complete" if val["valid"] else "invalid",
                "r1": f"{r1:.4f}" if isinstance(r1, (int, float)) else "N/A",
                "r5": f"{r5:.4f}" if isinstance(r5, (int, float)) else "N/A",
                "mrr": f"{mrr:.4f}" if isinstance(mrr, (int, float)) else "N/A",
                "margin": f"{margin:.4f}" if isinstance(margin, (int, float)) else "N/A",
                "notes": f"Generated from modern run {run_id}",
            }
            rows.append(row)

            companion_provenance[run_id] = {
                "provenance_status": row["provenance_status"],
                "run_directory": str(run_folder.relative_to(ROOT)),
                "config_snapshot": str((run_folder / "config.json").relative_to(ROOT)) if (run_folder / "config.json").exists() else None,
                "command": exec_info.get("command"),
                "git": git_info,
                "data": data_info,
                "validation": val,
            }
        except Exception as exc:
            print(f"[ERROR] Failed to aggregate modern run {run_folder}: {exc}", file=sys.stderr)

    # 2. Discover historical runs in results/
    for exp_dir in sorted(results_dir.glob("identity_residual_*")):
        if not exp_dir.is_dir():
            continue
        card_file = exp_dir / "experiment_card.yaml"
        if not card_file.exists():
            continue

        card = parse_historical_card(card_file)
        run_name = exp_dir.name
        commit = card.get("git_commit", "unknown")

        # Historical runs did not record dirty status, exact CLI command, or manifest sha256
        prov_status = "partial"
        if strict_provenance:
            print(f"[REJECTED STRICT HISTORICAL] {run_name} has only partial provenance", file=sys.stderr)
            continue

        # Look for existing benchmark metrics in CSV
        matched_csv = exp_dir / "speaker_balanced_mode_vector_results.csv"
        alt_csv = exp_dir / "restricted_gallery_results.csv"

        r1_val, r5_val, mrr_val, margin_val = "N/A", "N/A", "N/A", "N/A"
        target_csv = matched_csv if matched_csv.exists() else (alt_csv if alt_csv.exists() else None)
        if target_csv:
            try:
                with target_csv.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        # Extract top row or centroid row
                        if "centroid" in r.get("variant", "") or not r1_val != "N/A":
                            r1_val = r.get("R1", "N/A")
                            r5_val = r.get("R5", "N/A")
                            mrr_val = r.get("MRR", "N/A")
                            margin_val = r.get("same_minus_impostor_margin", "N/A")
                            if "centroid" in r.get("variant", ""):
                                break
            except Exception:
                pass

        row = {
            "experiment": "identity_residual_historical",
            "approach": "wavlm_l12_residualized",
            "dataset": card.get("datasets", "JVS_JVSMuSiC"),
            "run_id": run_name,
            "git_commit": commit,
            "git_dirty": "unknown (historical)",
            "seed": "unknown (historical)",
            "provenance_status": prov_status,
            "r1": r1_val,
            "r5": r5_val,
            "mrr": mrr_val,
            "margin": margin_val,
            "notes": "Historical run backfilled from experiment_card.yaml (dirty state and command unrecorded)",
        }
        rows.append(row)

        companion_provenance[run_name] = {
            "provenance_status": prov_status,
            "run_directory": str(exp_dir.relative_to(ROOT)),
            "git_commit": commit,
            "hostname": card.get("hostname"),
            "datasets": card.get("datasets"),
            "unrecoverable_fields": ["git.dirty", "execution.command", "data.manifest_sha256", "seed"],
        }

    return rows, companion_provenance


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate benchmark summary.csv directly from verified run metrics.")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs", help="Directory containing modern runs")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results", help="Directory containing historical results")
    parser.add_argument("--output-csv", type=Path, default=ROOT / "results" / "summary.csv", help="Destination summary.csv")
    parser.add_argument("--output-json", type=Path, default=ROOT / "results" / "summary_provenance.json", help="Companion provenance JSON")
    parser.add_argument("--strict", action="store_true", help="Reject runs with partial or invalid provenance")
    args = parser.parse_args()

    rows, companion = discover_and_aggregate_runs(args.runs_dir, args.results_dir, strict_provenance=args.strict)

    if not rows:
        print("warning: no runs matched aggregation criteria", file=sys.stderr)
        return 0

    fieldnames = [
        "experiment",
        "approach",
        "dataset",
        "run_id",
        "git_commit",
        "git_dirty",
        "seed",
        "provenance_status",
        "r1",
        "r5",
        "mrr",
        "margin",
        "notes",
    ]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(companion, f, indent=2, sort_keys=True)

    print(f"Aggregated {len(rows)} run(s) -> {args.output_csv}")
    print(f"Wrote companion provenance -> {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
