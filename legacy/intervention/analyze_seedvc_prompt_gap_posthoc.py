#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_utils import ExperimentError, read_table, write_json, write_table  # noqa: E402


GAP_FIELDS = {
    "target_singing_gain": "singing_prompt_delta_to_target_singing",
    "target_speech_gain": "singing_prompt_delta_to_target_speech",
    "source_gain": "singing_prompt_delta_to_source",
    "singing_domain_advantage": "singing_domain_advantage",
    "target_singing_gain_resid_rms": "target_singing_gain_resid_rms",
    "target_singing_gain_resid_language": "target_singing_gain_resid_language",
    "target_singing_gain_resid_rms_language": "target_singing_gain_resid_rms_language",
    "target_singing_gain_resid_rms_target": "target_singing_gain_resid_rms_target",
}

PREDICTOR_FIELDS = ("pred_delta_norm", "true_delta_norm", "residual_norm", "delta_cosine")


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"could not read JSON {path}: {exc}") from exc


def rankdata(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def corr_pair(x: list[float], y: list[float]) -> dict[str, Any]:
    pairs = [(a, b) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 3:
        return {"n": len(pairs), "pearson": None, "spearman": None}
    xv = np.asarray([p[0] for p in pairs], dtype=np.float64)
    yv = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if float(np.std(xv)) < 1e-12 or float(np.std(yv)) < 1e-12:
        pearson = None
    else:
        pearson = float(np.corrcoef(xv, yv)[0, 1])
    xr = rankdata(xv.tolist())
    yr = rankdata(yv.tolist())
    if float(np.std(xr)) < 1e-12 or float(np.std(yr)) < 1e-12:
        spearman = None
    else:
        spearman = float(np.corrcoef(xr, yr)[0, 1])
    return {"n": len(pairs), "pearson": pearson, "spearman": spearman}


def sign_test(values: list[float]) -> dict[str, Any]:
    nonzero = [v for v in values if math.isfinite(v) and abs(v) > 1e-12]
    positives = sum(v > 0 for v in nonzero)
    n = len(nonzero)
    if n == 0:
        return {"n": 0, "positive": 0, "negative": 0, "two_sided_p": None}
    lower_tail = sum(math.comb(n, i) for i in range(0, positives + 1)) / (2**n)
    upper_tail = sum(math.comb(n, i) for i in range(positives, n + 1)) / (2**n)
    return {
        "n": n,
        "positive": positives,
        "negative": n - positives,
        "two_sided_p": float(min(1.0, 2.0 * min(lower_tail, upper_tail))),
    }


def bootstrap_mean_ci(values: list[float], seed: int, draws: int = 20000) -> list[float | None]:
    clean = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if len(clean) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        sample = clean[rng.integers(0, len(clean), size=len(clean))]
        means[i] = float(np.mean(sample))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return [float(lo), float(hi)]


def numeric_summary(values: list[float], seed: int) -> dict[str, Any]:
    clean = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if len(clean) == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "mean_ci95_bootstrap": [None, None],
            "sign_test": sign_test([]),
        }
    return {
        "n": int(len(clean)),
        "mean": float(np.mean(clean)),
        "median": float(np.median(clean)),
        "std": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "mean_ci95_bootstrap": bootstrap_mean_ci(clean.tolist(), seed=seed),
        "sign_test": sign_test(clean.tolist()),
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def read_manifest_pairs(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_table(path)
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in rows:
        pair_id = str(row["target_pair_id"])
        condition = str(row["condition"])
        grouped[pair_id][condition] = row
    return grouped


def pair_records(seedvc_summary: dict[str, Any], manifest_by_pair: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in seedvc_summary.get("pair_deltas", []):
        pair_id = str(row["target_pair_id"])
        conds = manifest_by_pair.get(pair_id, {})
        speech = conds.get("target_speech_prompt", {})
        singing = conds.get("target_singing_prompt", {})
        target_singing_gain = finite_float(row.get("singing_prompt_delta_to_target_singing"))
        target_speech_gain = finite_float(row.get("singing_prompt_delta_to_target_speech"))
        source_gain = finite_float(row.get("singing_prompt_delta_to_source"))
        speech_rms = finite_float(speech.get("rms_db"))
        singing_rms = finite_float(singing.get("rms_db"))
        record = {
            "pair_id": pair_id,
            "language": str((singing or speech).get("target_singing_utt_id", pair_id)).split("__", 1)[0],
            "source_speaker_id": row.get("source_speaker_id") or (singing or speech).get("source_speaker_id"),
            "target_speaker_id": row.get("target_speaker_id") or (singing or speech).get("target_speaker_id"),
            "target_singing_gain": target_singing_gain,
            "target_speech_gain": target_speech_gain,
            "source_gain": source_gain,
            "singing_domain_advantage": (
                target_singing_gain - target_speech_gain
                if target_singing_gain is not None and target_speech_gain is not None
                else None
            ),
            "speech_prompt_rms_db": speech_rms,
            "singing_prompt_rms_db": singing_rms,
            "rms_delta_db": singing_rms - speech_rms if speech_rms is not None and singing_rms is not None else None,
            "speech_prompt_duration_sec": finite_float(speech.get("duration_sec")),
            "singing_prompt_duration_sec": finite_float(singing.get("duration_sec")),
        }
        out.append(record)
    return out


def one_hot(values: list[str]) -> np.ndarray:
    categories = sorted(set(values))
    if not categories:
        return np.zeros((len(values), 0), dtype=np.float64)
    return np.asarray([[1.0 if value == cat else 0.0 for cat in categories] for value in values], dtype=np.float64)


def add_residual_field(records: list[dict[str, Any]], output: str, y_field: str, covariates: list[str]) -> None:
    valid_idx = []
    y = []
    numeric_cols: list[list[float]] = []
    categorical_cols: list[list[str]] = []
    for i, row in enumerate(records):
        y_value = finite_float(row.get(y_field))
        if y_value is None:
            continue
        numeric_values = []
        categorical_values = []
        ok = True
        for cov in covariates:
            if cov in {"language", "target_speaker_id", "source_speaker_id"}:
                categorical_values.append(str(row.get(cov, "")))
            else:
                value = finite_float(row.get(cov))
                if value is None:
                    ok = False
                    break
                numeric_values.append(value)
        if not ok:
            continue
        valid_idx.append(i)
        y.append(y_value)
        numeric_cols.append(numeric_values)
        categorical_cols.append(categorical_values)
    for row in records:
        row[output] = None
    if len(valid_idx) < 3:
        return
    x_parts = [np.ones((len(valid_idx), 1), dtype=np.float64)]
    if numeric_cols and numeric_cols[0]:
        x_parts.append(np.asarray(numeric_cols, dtype=np.float64))
    if categorical_cols and categorical_cols[0]:
        cats = np.asarray(categorical_cols, dtype=object)
        for j in range(cats.shape[1]):
            x_parts.append(one_hot([str(value) for value in cats[:, j].tolist()]))
    x = np.hstack(x_parts)
    y_arr = np.asarray(y, dtype=np.float64)
    beta, *_ = np.linalg.lstsq(x, y_arr, rcond=None)
    resid = y_arr - x @ beta
    for idx, value in zip(valid_idx, resid):
        records[idx][output] = float(value)


def add_residualized_gaps(records: list[dict[str, Any]]) -> None:
    add_residual_field(records, "target_singing_gain_resid_rms", "target_singing_gain", ["rms_delta_db"])
    add_residual_field(records, "target_singing_gain_resid_language", "target_singing_gain", ["language"])
    add_residual_field(
        records,
        "target_singing_gain_resid_rms_language",
        "target_singing_gain",
        ["rms_delta_db", "language"],
    )
    add_residual_field(
        records,
        "target_singing_gain_resid_rms_target",
        "target_singing_gain",
        ["rms_delta_db", "target_speaker_id"],
    )


def grouped_summary(records: list[dict[str, Any]], field: str, seed: int) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in records:
        value = finite_float(row.get(field))
        if value is None:
            continue
        groups[str(row.get("language", ""))].append(value)
    return {key: numeric_summary(vals, seed=seed) for key, vals in sorted(groups.items())}


def grouped_summary_by_key(records: list[dict[str, Any]], key: str, field: str, seed: int) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in records:
        value = finite_float(row.get(field))
        if value is None:
            continue
        groups[str(row.get(key, ""))].append(value)
    return {name: numeric_summary(vals, seed=seed) for name, vals in sorted(groups.items())}


def path_label(predictions_path: Path) -> str:
    if predictions_path.name == "predictions.jsonl":
        return f"{predictions_path.parent.parent.name}/{predictions_path.parent.name}"
    return predictions_path.stem


def prediction_paths(paths: list[Path]) -> list[Path]:
    out = []
    for path in paths:
        if path.is_dir():
            found = sorted(path.glob("*/predictions.jsonl"))
            out.extend(found)
        elif path.exists():
            out.append(path)
    if not out:
        raise ExperimentError("no prediction files found")
    return out


def track1_join_summary(records: list[dict[str, Any]], paths: list[Path]) -> list[dict[str, Any]]:
    seedvc_by_pair = {str(row["pair_id"]): row for row in records}
    summaries = []
    for path in prediction_paths(paths):
        rep = path_label(path)
        rows = [row for row in read_table(path) if str(row.get("pair_id")) in seedvc_by_pair]
        by_model: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_model[(str(row["model"]), int(row["target_components"]))].append(row)
        for (model, k), group in sorted(by_model.items(), key=lambda item: (item[0][1], item[0][0])):
            joined = []
            for row in group:
                seedvc = seedvc_by_pair[str(row["pair_id"])]
                joined.append({**row, **{f"seedvc_{key}": value for key, value in seedvc.items()}})
            for gap_name in GAP_FIELDS:
                y = [finite_float(row.get(f"seedvc_{gap_name}")) for row in joined]
                y_clean = [float(v) if v is not None else float("nan") for v in y]
                for predictor in PREDICTOR_FIELDS:
                    x = [finite_float(row.get(predictor)) for row in joined]
                    x_clean = [float(v) if v is not None else float("nan") for v in x]
                    corr = corr_pair(x_clean, y_clean)
                    summaries.append(
                        {
                            "representation": rep,
                            "target_components": k,
                            "model": model,
                            "gap": gap_name,
                            "predictor": predictor,
                            **corr,
                        }
                    )
    return summaries


def qa_correlations(records: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for gap in GAP_FIELDS:
        y = [finite_float(row.get(gap)) for row in records]
        y_clean = [float(v) if v is not None else float("nan") for v in y]
        for x_name in ("rms_delta_db", "source_gain", "singing_domain_advantage"):
            if x_name == gap:
                continue
            x = [finite_float(row.get(x_name)) for row in records]
            x_clean = [float(v) if v is not None else float("nan") for v in x]
            out[f"{x_name}_vs_{gap}"] = corr_pair(x_clean, y_clean)
    return out


def best_track1_rows(rows: list[dict[str, Any]], gap: str = "target_singing_gain", min_n: int = 30) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if row["gap"] == gap and row["spearman"] is not None and row["n"] >= min_n
    ]
    candidates.sort(key=lambda row: abs(float(row["spearman"])), reverse=True)
    return candidates[:20]


def write_markdown_report(payload: dict[str, Any], path: Path) -> None:
    seedvc = payload["seedvc"]
    target = seedvc["gap_summaries"]["target_singing_gain"]
    speech = seedvc["gap_summaries"]["target_speech_gain"]
    source = seedvc["gap_summaries"]["source_gain"]
    domain = seedvc["gap_summaries"]["singing_domain_advantage"]
    qa = payload["qa_correlations"]
    top = payload["track1_top_target_singing_correlations"][:10]
    top_resid = payload["track1_top_resid_rms_language_correlations"][:10]

    lines = [
        "# SeedVC 30-pair Prompt Gap Posthoc",
        "",
        "## 结论",
        "",
        (
            f"- SeedVC 自动 proxy 上，singing prompt 相比 speech prompt 更接近 target singing："
            f"mean={fmt(target['mean'], 4)}, 95% bootstrap CI="
            f"[{fmt(target['mean_ci95_bootstrap'][0], 4)}, {fmt(target['mean_ci95_bootstrap'][1], 4)}], "
            f"sign={target['sign_test']['positive']}/{target['sign_test']['n']} positive。"
        ),
        (
            f"- 同时它更不接近 target speech：mean={fmt(speech['mean'], 4)}, "
            f"sign={speech['sign_test']['positive']}/{speech['sign_test']['n']} positive。"
            " 这提示 Resemblyzer 差值混有 singing/speech 域匹配，不是纯身份判断。"
        ),
        (
            f"- 对 source singing 的平均变化很小：mean={fmt(source['mean'], 4)}。"
            " 说明这次 prompt 模式主要改变 target-domain 相似度，不像是简单把输出整体推向 source。"
        ),
        (
            f"- target-singing gain 减 target-speech gain 的域优势很大：mean={fmt(domain['mean'], 4)}。"
            " 这更像“目标 singing reference 的域/唱法匹配优势”，需要听评确认它是不是人耳身份优势。"
        ),
        "",
        "## 自动 QA",
        "",
        "| check | n | pearson | spearman |",
        "|---|---:|---:|---:|",
    ]
    for key in sorted(qa):
        item = qa[key]
        lines.append(
            f"| {key} | {item['n']} | {fmt(item['pearson'], 3)} | {fmt(item['spearman'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## Track 1 Accounting Join",
            "",
            "这些是 SeedVC target-singing gain 与 Track 1 v2 prediction rows 的最高绝对 Spearman 相关。它们是探索性检查，不应当单独当显著性结论。",
            "",
            "| rep | K | model | predictor | n | pearson | spearman |",
            "|---|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in top:
        lines.append(
            f"| {row['representation']} | {row['target_components']} | {row['model']} | "
            f"{row['predictor']} | {row['n']} | {fmt(row['pearson'], 3)} | {fmt(row['spearman'], 3)} |"
        )
    lines.extend(
        [
            "",
            "去掉 RMS delta 和 language 后，target-singing gain residual 与 Track 1 predictors 的最高相关：",
            "",
            "| rep | K | model | predictor | n | pearson | spearman |",
            "|---|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in top_resid:
        lines.append(
            f"| {row['representation']} | {row['target_components']} | {row['model']} | "
            f"{row['predictor']} | {row['n']} | {fmt(row['pearson'], 3)} | {fmt(row['spearman'], 3)} |"
        )
    lines.extend(
        [
            "",
            "## 人需要介入的位置",
            "",
            "- 现在需要人工听评的不是“是否有自动差值”，而是自动差值究竟对应人耳身份、唱法/气声、还是 speech-vs-singing 域匹配。",
            "- 优先听 target-singing gain 高、但 target-speech gain 低的样本；这类最容易让 Resemblyzer 把域匹配当身份。",
            "- 本轮听评页面：`listening_review.html`，每个 pair 包含 source、target speech、target singing、speech prompt generation、singing prompt generation。"
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    seedvc_summary = load_json(args.seedvc_summary)
    manifest = read_manifest_pairs(args.listening_manifest)
    records = pair_records(seedvc_summary, manifest)
    if not records:
        raise ExperimentError("no pair-level SeedVC records found")
    add_residualized_gaps(records)
    gap_summaries = {
        name: numeric_summary(
            [float(v) for v in (finite_float(row.get(name)) for row in records) if v is not None],
            seed=args.seed,
        )
        for name in GAP_FIELDS
    }
    track1 = track1_join_summary(records, args.predictions)
    return {
        "stage": "seedvc_prompt_gap_posthoc",
        "seedvc_summary": str(args.seedvc_summary),
        "listening_manifest": str(args.listening_manifest),
        "pairs": len(records),
        "coverage": {
            "languages": dict(Counter(str(row.get("language", "")) for row in records)),
            "target_speakers": dict(Counter(str(row.get("target_speaker_id", "")) for row in records)),
            "source_speakers": dict(Counter(str(row.get("source_speaker_id", "")) for row in records)),
        },
        "seedvc": {
            "gap_summaries": gap_summaries,
            "by_language_target_singing_gain": grouped_summary(records, "target_singing_gain", seed=args.seed),
            "by_target_speaker_target_singing_gain": grouped_summary_by_key(
                records, "target_speaker_id", "target_singing_gain", seed=args.seed
            ),
        },
        "qa_correlations": qa_correlations(records),
        "track1_correlations": track1,
        "track1_top_target_singing_correlations": best_track1_rows(track1, gap="target_singing_gain", min_n=len(records)),
        "track1_top_resid_rms_language_correlations": best_track1_rows(
            track1, gap="target_singing_gain_resid_rms_language", min_n=len(records)
        ),
        "pair_records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Posthoc QA for SeedVC prompt gap and Track 1 accounting joins.")
    parser.add_argument("--seedvc-summary", type=Path, required=True)
    parser.add_argument("--listening-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--pairs-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    try:
        payload = run(args)
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_json(payload, args.json_out)
    if args.pairs_out:
        write_table(payload["pair_records"], args.pairs_out)
    if args.report_out:
        write_markdown_report(payload, args.report_out)
    print(json.dumps({k: payload[k] for k in ("stage", "pairs", "coverage")}, indent=2, sort_keys=True))
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
