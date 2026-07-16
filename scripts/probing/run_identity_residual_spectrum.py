#!/usr/bin/env python
"""Train-speaker residual spectrum analysis without a test-target adapter."""
from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probing.run_identity_residual_robustness import (  # noqa: E402
    JVS_SEEDS,
    MODELS,
    GTSINGER_SEEDS,
    load_gtsinger,
    load_jvs,
    split_indices,
)
from probing.run_identity_residual_suite import write_csv, write_yaml  # noqa: E402
from research_utils import current_git_commit  # noqa: E402


def run(args: argparse.Namespace) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    spectrum: list[dict[str, Any]] = []
    energy: list[dict[str, Any]] = []
    for dataset in args.datasets:
        for model in args.models:
            if dataset == "JVS_JVSMuSiC":
                speakers, vectors, layer = load_jvs(model, args.robustness_run_root / dataset / model / "cache")
                seeds, protocol = JVS_SEEDS, "jvs_60_20_20"
            else:
                speakers, vectors, layer, _pairs = load_gtsinger(model, args.robustness_run_root / dataset / model / "cache")
                seeds, protocol = GTSINGER_SEEDS, "gtsinger_10_0_10"
            speech = np.vstack([vectors[s]["speech"].mean(axis=0) for s in speakers])
            singing = np.vstack([vectors[s]["singing"].mean(axis=0) for s in speakers])
            for seed in seeds:
                split, train, _test = split_indices(speakers, seed, protocol)
                residual = singing[train] - speech[train]
                mean = residual.mean(axis=0)
                centered = residual - mean
                # The non-zero feature-space covariance eigenvalues equal those
                # of this much smaller speaker Gram matrix.
                eigenvalue = np.linalg.eigvalsh(centered @ centered.T)[::-1] / max(len(train) - 1, 1)
                eigenvalue = np.maximum(eigenvalue, 0.0)
                total = max(float(eigenvalue.sum()), 1e-12)
                fraction = eigenvalue / total
                cumulative = np.cumsum(fraction)
                explained_energy = 1.0 - float(np.sum(centered ** 2)) / max(float(np.sum(residual ** 2)), 1e-12)
                positive = fraction[fraction > 0]
                effective_rank = float(np.exp(-np.sum(positive * np.log(positive)))) if len(positive) else 0.0
                energy.append({
                    "dataset": dataset, "model": model, "layer": layer, "split_seed": seed,
                    "train_speakers": ";".join(speakers[i] for i in train),
                    "shared_mean_explained_energy": explained_energy,
                    "mean_residual_norm": float(np.linalg.norm(residual, axis=1).mean()),
                    "mean_centered_residual_norm": float(np.linalg.norm(centered, axis=1).mean()),
                    "global_mean_norm": float(np.linalg.norm(mean)),
                    "effective_rank_centered": effective_rank,
                })
                for component, (value, share, cum) in enumerate(zip(eigenvalue, fraction, cumulative), start=1):
                    spectrum.append({
                        "dataset": dataset, "model": model, "layer": layer, "split_seed": seed,
                        "component": component, "centered_residual_eigenvalue": float(value),
                        "centered_variance_fraction": float(share), "centered_variance_cumulative": float(cum),
                        "analysis_population": "train_speakers_only",
                    })
    summary: list[dict[str, Any]] = []
    for key in sorted({(r["dataset"], r["model"], r["layer"], r["component"]) for r in spectrum}):
        rows = [r for r in spectrum if (r["dataset"], r["model"], r["layer"], r["component"]) == key]
        summary.append({
            "dataset": key[0], "model": key[1], "layer": key[2], "component": key[3],
            "eigenvalue_mean": float(np.mean([r["centered_residual_eigenvalue"] for r in rows])),
            "variance_fraction_mean": float(np.mean([r["centered_variance_fraction"] for r in rows])),
            "variance_cumulative_mean": float(np.mean([r["centered_variance_cumulative"] for r in rows])),
            "splits": len(rows),
        })
    write_csv(spectrum, args.results_dir / "centered_residual_spectrum_per_split.csv")
    write_csv(summary, args.results_dir / "centered_residual_spectrum_summary.csv")
    write_csv(energy, args.results_dir / "global_mean_energy_per_split.csv")
    write_yaml({
        "experiment_name": "identity_residual_spectrum_2026-07-10", "hostname": socket.gethostname(), "python": sys.executable, "git_commit": current_git_commit(),
        "robustness_run_root": str(args.robustness_run_root), "results_dir": str(args.results_dir), "cache_root": os.environ.get("XDG_CACHE_HOME", ""),
        "fit_population": "train speakers only", "adapter": "not_run", "reason": "A test-speech-only non-oracle coefficient rule was not pre-specified; fitting one would resume the stopped individualized mapper line.",
    }, args.results_dir / "experiment_card.yaml")
    (args.results_dir / "README_results.md").write_text(
        "# Global Residual Spectrum\n\nThis is a descriptive train-speaker PCA spectrum. It does not apply PCA components to held-out speakers, reconstruct test singing, or claim a one-dimensional residual.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Residual spectrum analysis for the global displacement.")
    parser.add_argument("--robustness-run-root", type=Path, default=Path("/localdisk/bowen/singing_identity/runs/identity_residual_protocol_robustness_2026-07-10"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/identity_residual_spectrum_2026-07-10"))
    parser.add_argument("--datasets", nargs="+", choices=["JVS_JVSMuSiC", "GTSinger_same_text_control"], default=["JVS_JVSMuSiC", "GTSinger_same_text_control"])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
