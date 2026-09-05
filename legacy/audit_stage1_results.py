#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def collect_jobs(run_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((run_root / "jobs").glob("*/status.json")):
        data = read_json(path)
        if data:
            data.setdefault("name", path.parent.name)
            rows.append(data)
    return rows


def track1_rows(run_root: Path) -> list[dict[str, Any]]:
    rows = []
    for controlled_path in sorted(run_root.glob("track1/*/*/mode_controlled_metrics.json")):
        parent = controlled_path.parent
        controlled = read_json(controlled_path)
        raw = read_json(parent / "mode_raw_metrics.json")
        shuffled = read_json(parent / "mode_shuffled_metrics.json")
        retrieval = read_json(parent / "retrieval_metrics.json")
        mapper = read_json(parent / "mapper_metrics.json")
        rows.append(
            {
                "model": controlled_path.parts[-3],
                "layer": controlled_path.parts[-2],
                "raw_sep_auc": safe_float(raw.get("metrics", {}).get("separability_auc_mean")),
                "controlled_sep_auc": safe_float(controlled.get("metrics", {}).get("separability_auc_mean")),
                "shuffled_sep_auc": safe_float(shuffled.get("metrics", {}).get("separability_auc_mean")),
                "r1": safe_float(retrieval.get("singing_to_speech_recall_at_1")),
                "r5": safe_float(retrieval.get("singing_to_speech_recall_at_5")),
                "mapper": safe_float(mapper.get("cosine_mean")),
                "global_residual": safe_float(mapper.get("global_mean_residual_cosine_mean")),
            }
        )
    return rows


def track2_rows(run_root: Path) -> list[dict[str, Any]]:
    rows = []
    for metrics_path in sorted(run_root.glob("track2/*/*/technique_metrics.json")):
        data = read_json(metrics_path)
        groups = data.get("groups", [])
        reliable = [g for g in groups if g.get("num_pairs", 0) >= 3]
        passed = [
            g
            for g in reliable
            if g.get("bootstrap_angle_mean_deg", 999) <= 30
            and g.get("analogy_top1", 0) > g.get("wrong_technique_top1", 0)
            and g.get("analogy_top1", 0) > g.get("shuffled_direction_top1", 0)
        ]
        rows.append(
            {
                "model": metrics_path.parts[-3],
                "layer": metrics_path.parts[-2],
                "reliable_groups": len(reliable),
                "passed_groups": len(passed),
            }
        )
    return rows


def format_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Stage 1 run status and completed metrics.")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local"),
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    jobs = collect_jobs(run_root)
    counts: dict[str, int] = {}
    for job in jobs:
        counts[str(job.get("status", "missing"))] = counts.get(str(job.get("status", "missing")), 0) + 1
    bad_jobs = [job for job in jobs if job.get("status") not in {"success", "skipped"}]

    lines = [
        "# Stage 1 Audit",
        "",
        f"- Run root: `{run_root}`",
        f"- Job counts: `{json.dumps(counts, sort_keys=True)}`",
    ]
    if bad_jobs:
        lines.extend(["", "## Jobs To Check", ""])
        for job in bad_jobs:
            lines.append(f"- `{job.get('name')}` status=`{job.get('status')}` log=`{job.get('log')}`")

    t1 = track1_rows(run_root)
    lines.extend(["", "## Track 1", ""])
    if t1:
        lines.extend(
            format_table(
                ["model", "layer", "raw_sep", "controlled_sep", "shuffled_sep", "R@1", "R@5", "mapper", "global"],
                [
                    [
                        row["model"],
                        row["layer"],
                        f"{row['raw_sep_auc']:.3f}",
                        f"{row['controlled_sep_auc']:.3f}",
                        f"{row['shuffled_sep_auc']:.3f}",
                        f"{row['r1']:.3f}",
                        f"{row['r5']:.3f}",
                        f"{row['mapper']:.3f}",
                        f"{row['global_residual']:.3f}",
                    ]
                    for row in t1
                ],
            )
        )
    else:
        lines.append("No completed Track 1 controlled metrics found.")

    t2 = track2_rows(run_root)
    lines.extend(["", "## Track 2", ""])
    if t2:
        lines.extend(
            format_table(
                ["model", "layer", "reliable_groups", "passed_groups"],
                [[row["model"], row["layer"], row["reliable_groups"], row["passed_groups"]] for row in t2],
            )
        )
    else:
        lines.append("No completed Track 2 metrics found.")

    output = "\n".join(lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
