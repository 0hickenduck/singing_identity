#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


CASES = (
    "no_nuisance_mode_dominance",
    "mode_nuisance_dominance",
    "mode_proxy",
    "within_mode_nuisance",
    "random_nuisance",
    "leakage_trap",
)

VARIANT_FORMULAS = {
    "raw": "z",
    "mode_dummy_only": "z - pred_train(z ~ 1 + mode_dummy) + mean_train(pred)",
    "full_nuisance_mean_preserving": (
        "z - pred_train(z ~ 1 + mode_dummy + nuisance) + mean_train(pred)"
    ),
    "full_nuisance_classic": "z - pred_train(z ~ 1 + mode_dummy + nuisance)",
    "random_nuisance_mean_preserving": (
        "z - pred_train(z ~ 1 + random_nuisance) + mean_train(pred)"
    ),
}


@dataclass(frozen=True)
class SyntheticData:
    z: np.ndarray
    speaker: np.ndarray
    mode: np.ndarray
    nuisance: np.ndarray
    random_nuisance: np.ndarray
    train_mask: np.ndarray
    test_speakers: np.ndarray
    case_notes: str


def _unit(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm <= 0:
        raise ValueError("cannot normalize zero vector")
    return x / norm


def _orthogonal_components(rng: np.random.Generator, count: int, dim: int) -> np.ndarray:
    basis, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    return basis[:count]


def _case_params(case: str) -> Dict[str, float]:
    params: Dict[str, Dict[str, float]] = {
        "no_nuisance_mode_dominance": {
            "speaker": 1.8,
            "mode": 0.10,
            "nuisance": 0.0,
            "noise": 0.03,
            "utterances": 4,
        },
        "mode_nuisance_dominance": {
            "speaker": 0.55,
            "mode": 9.0,
            "nuisance": 0.0,
            "noise": 0.22,
            "utterances": 5,
        },
        "mode_proxy": {
            "speaker": 0.90,
            "mode": 0.0,
            "nuisance": 5.5,
            "noise": 0.06,
            "utterances": 4,
        },
        "within_mode_nuisance": {
            "speaker": 1.20,
            "mode": 0.0,
            "nuisance": 5.0,
            "noise": 0.08,
            "utterances": 4,
        },
        "random_nuisance": {
            "speaker": 1.8,
            "mode": 0.10,
            "nuisance": 0.0,
            "noise": 0.04,
            "utterances": 4,
        },
        "leakage_trap": {
            "speaker": 1.05,
            "mode": 0.0,
            "nuisance": 5.0,
            "noise": 0.06,
            "utterances": 4,
        },
    }
    return params[case]


def _make_nuisance(
    case: str,
    rng: np.random.Generator,
    speaker_index: int,
    mode_sign: int,
    is_train: bool,
) -> Tuple[float, float]:
    if case == "mode_proxy":
        return mode_sign + rng.normal(), 1.0
    if case == "within_mode_nuisance":
        return rng.normal(), 1.0
    if case == "leakage_trap":
        # The nuisance-to-embedding relationship exists only for held-out
        # speakers. A correct train-only residualizer cannot estimate it.
        return rng.normal(), 0.0 if is_train else 1.0
    if case == "random_nuisance":
        return rng.normal(), 0.0
    if case == "mode_nuisance_dominance":
        return 0.0, 0.0
    if case == "no_nuisance_mode_dominance":
        return 0.0, 0.0
    raise ValueError(f"unknown case: {case}")


def generate_case(case: str, speakers: int, seed: int) -> SyntheticData:
    if speakers < 8:
        raise ValueError("--speakers must be at least 8 for a speaker-disjoint split")

    params = _case_params(case)
    rng = np.random.default_rng(seed)
    dim = max(96, speakers + 8)
    components = _orthogonal_components(rng, 3, dim)
    q_mode = components[0]
    v_nuisance = components[1]

    speaker_vectors = rng.normal(size=(speakers, dim))
    speaker_vectors = speaker_vectors / np.linalg.norm(speaker_vectors, axis=1, keepdims=True)

    train_count = max(4, int(round(speakers * 0.65)))
    train_count = min(train_count, speakers - 4)
    test_speakers = np.arange(train_count, speakers)
    utterances = int(params["utterances"])

    z_rows: List[np.ndarray] = []
    speaker_rows: List[int] = []
    mode_rows: List[int] = []
    nuisance_rows: List[float] = []
    random_rows: List[float] = []
    train_rows: List[bool] = []

    for speaker_index in range(speakers):
        is_train = speaker_index < train_count
        for mode in (0, 1):
            mode_sign = -1 if mode == 0 else 1
            for _ in range(utterances):
                nuisance, case_nuisance_multiplier = _make_nuisance(
                    case, rng, speaker_index, mode_sign, is_train
                )
                noise = params["noise"] * rng.normal(size=dim)
                z = (
                    params["speaker"] * speaker_vectors[speaker_index]
                    + params["mode"] * mode_sign * q_mode
                    + params["nuisance"] * case_nuisance_multiplier * nuisance * v_nuisance
                    + noise
                )
                z_rows.append(z)
                speaker_rows.append(speaker_index)
                mode_rows.append(mode)
                nuisance_rows.append(float(nuisance))
                random_rows.append(float(rng.normal()))
                train_rows.append(is_train)

    notes = {
        "no_nuisance_mode_dominance": "identity signal dominates; residualization should preserve retrieval",
        "mode_nuisance_dominance": "large additive mode offset; mode dummy residualization should help",
        "mode_proxy": "measured nuisance is a noisy proxy for mode; full nuisance should help most",
        "within_mode_nuisance": "within-mode linear nuisance corrupts centroids; full nuisance should remove it",
        "random_nuisance": "measured nuisance is unrelated; random control should not improve",
        "leakage_trap": (
            "nuisance relationship is test-only; train-only residualizer should not exploit it"
        ),
    }[case]

    return SyntheticData(
        z=np.asarray(z_rows, dtype=float),
        speaker=np.asarray(speaker_rows, dtype=int),
        mode=np.asarray(mode_rows, dtype=int),
        nuisance=np.asarray(nuisance_rows, dtype=float),
        random_nuisance=np.asarray(random_rows, dtype=float),
        train_mask=np.asarray(train_rows, dtype=bool),
        test_speakers=test_speakers,
        case_notes=notes,
    )


def _standardize_train_only(x: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    mean = x[train_mask].mean(axis=0)
    std = x[train_mask].std(axis=0)
    std[std < 1e-12] = 1.0
    return (x - mean) / std


def residualize(
    z: np.ndarray,
    design: np.ndarray,
    train_mask: np.ndarray,
    *,
    mean_preserving: bool,
) -> np.ndarray:
    x = _standardize_train_only(design, train_mask)
    x_with_intercept = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(x_with_intercept[train_mask], z[train_mask], rcond=None)[0]
    pred = x_with_intercept @ beta
    if mean_preserving:
        return z - pred + pred[train_mask].mean(axis=0)
    return z - pred


def transformed_variants(data: SyntheticData) -> Dict[str, np.ndarray]:
    mode_dummy = data.mode.astype(float)[:, None]
    full_nuisance = np.column_stack([data.mode.astype(float), data.nuisance])
    random_nuisance = data.random_nuisance[:, None]
    return {
        "raw": data.z,
        "mode_dummy_only": residualize(
            data.z, mode_dummy, data.train_mask, mean_preserving=True
        ),
        "full_nuisance_mean_preserving": residualize(
            data.z, full_nuisance, data.train_mask, mean_preserving=True
        ),
        "full_nuisance_classic": residualize(
            data.z, full_nuisance, data.train_mask, mean_preserving=False
        ),
        "random_nuisance_mean_preserving": residualize(
            data.z, random_nuisance, data.train_mask, mean_preserving=True
        ),
    }


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def retrieval_counts(data: SyntheticData, z: np.ndarray) -> Dict[str, Tuple[int, int, float]]:
    speech_centroids = []
    singing_centroids = []
    for speaker in data.test_speakers:
        speech_mask = (data.speaker == speaker) & (data.mode == 0)
        singing_mask = (data.speaker == speaker) & (data.mode == 1)
        speech_centroids.append(z[speech_mask].mean(axis=0))
        singing_centroids.append(z[singing_mask].mean(axis=0))

    speech = _normalize_rows(np.asarray(speech_centroids))
    singing = _normalize_rows(np.asarray(singing_centroids))
    sim = speech @ singing.T
    target = np.arange(len(data.test_speakers))
    s_to_g = int(np.sum(np.argmax(sim, axis=1) == target))
    g_to_s = int(np.sum(np.argmax(sim, axis=0) == target))
    den = len(data.test_speakers)
    chance = 1.0 / den
    return {"S->G": (s_to_g, den, chance), "G->S": (g_to_s, den, chance)}


def _avg_variant(rows: Iterable[Dict[str, object]], case: str, variant: str) -> float:
    vals = [
        float(row["r1"])
        for row in rows
        if row["case"] == case and row["variant"] == variant
    ]
    if not vals:
        raise ValueError(f"missing rows for {case}/{variant}")
    return float(np.mean(vals))


def case_passes(rows: List[Dict[str, object]], case: str) -> Tuple[bool, str]:
    raw = _avg_variant(rows, case, "raw")
    mode = _avg_variant(rows, case, "mode_dummy_only")
    full = _avg_variant(rows, case, "full_nuisance_mean_preserving")
    classic = _avg_variant(rows, case, "full_nuisance_classic")
    random_control = _avg_variant(rows, case, "random_nuisance_mean_preserving")

    if case == "no_nuisance_mode_dominance":
        ok = raw >= 0.95 and full >= 0.95 and mode >= 0.95
        return ok, "raw and residualized variants should stay near ceiling"
    if case == "mode_nuisance_dominance":
        ok = mode >= raw + 0.10 and full >= raw + 0.10
        return ok, "mode-aware residualization should beat raw by at least 0.10 R@1"
    if case == "mode_proxy":
        ok = full >= raw + 0.20 and full >= mode - 0.05 and classic >= 0.85
        return ok, "full nuisance should remove the noisy mode proxy"
    if case == "within_mode_nuisance":
        ok = full >= raw + 0.35 and full >= mode + 0.30 and classic >= 0.90
        return ok, "full nuisance should beat mode-only on within-mode nuisance"
    if case == "random_nuisance":
        ok = abs(full - raw) <= 0.10 and abs(random_control - raw) <= 0.10
        return ok, "unrelated nuisance and random control should not materially change R@1"
    if case == "leakage_trap":
        ok = (full - raw) <= 0.20 and (classic - raw) <= 0.20
        return ok, "train-only residualizer must not exploit test-only nuisance relationship"
    raise ValueError(f"unknown case: {case}")


def run_synthetic_gate(out_dir: Path, seed: int, speakers: int) -> Tuple[List[Dict[str, object]], bool]:
    rows: List[Dict[str, object]] = []
    for case_index, case in enumerate(CASES):
        data = generate_case(case, speakers=speakers, seed=seed + 1009 * case_index)
        variants = transformed_variants(data)
        raw_counts = retrieval_counts(data, variants["raw"])
        raw_by_direction = {
            direction: num / den for direction, (num, den, _chance) in raw_counts.items()
        }

        case_start = len(rows)
        for variant, z_variant in variants.items():
            counts = retrieval_counts(data, z_variant)
            for direction, (num, den, chance) in counts.items():
                r1 = num / den
                rows.append(
                    {
                        "case": case,
                        "variant": variant,
                        "formula": VARIANT_FORMULAS[variant],
                        "direction": direction,
                        "r1_num": num,
                        "r1_den": den,
                        "r1": r1,
                        "chance_r1": chance,
                        "delta_vs_raw": r1 - raw_by_direction[direction],
                        "pass_expected": "",
                        "notes": data.case_notes,
                    }
                )

        case_ok, expectation = case_passes(rows, case)
        for row in rows[case_start:]:
            row["pass_expected"] = "true" if case_ok else "false"
            row["notes"] = f"{row['notes']}; expectation: {expectation}"

    all_ok = all(row["pass_expected"] == "true" for row in rows)
    write_outputs(out_dir, rows, seed, speakers, all_ok)
    return rows, all_ok


def _format_float(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_outputs(
    out_dir: Path,
    rows: List[Dict[str, object]],
    seed: int,
    speakers: int,
    all_ok: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "case",
        "variant",
        "formula",
        "direction",
        "r1_num",
        "r1_den",
        "r1",
        "chance_r1",
        "delta_vs_raw",
        "pass_expected",
        "notes",
    ]
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_float(row[column]) for column in columns})

    case_status = {
        case: all(row["pass_expected"] == "true" for row in rows if row["case"] == case)
        for case in CASES
    }
    card = {
        "name": "identity_residual_synthetic_gate",
        "seed": seed,
        "speakers": speakers,
        "cases": list(CASES),
        "variants": list(VARIANT_FORMULAS),
        "speaker_split": "speaker-disjoint; first 65% train speakers, remainder test speakers",
        "fit_scope": "residualizers and scalers are fit only on train speaker utterances",
        "outputs": ["summary.csv", "README_results.md", "experiment_card.yaml"],
        "passed": all_ok,
        "case_status": case_status,
    }
    write_yaml(out_dir / "experiment_card.yaml", card)

    lines = [
        "# Identity Residual Synthetic Gate",
        "",
        f"- seed: `{seed}`",
        f"- speakers: `{speakers}`",
        f"- passed: `{str(all_ok).lower()}`",
        "- split: speaker-disjoint, train speakers only for scalers and residualizers",
        "",
        "## Case Verdicts",
        "",
    ]
    for case in CASES:
        status = "PASS" if case_status[case] else "FAIL"
        raw = _avg_variant(rows, case, "raw")
        mode = _avg_variant(rows, case, "mode_dummy_only")
        full = _avg_variant(rows, case, "full_nuisance_mean_preserving")
        classic = _avg_variant(rows, case, "full_nuisance_classic")
        random_control = _avg_variant(rows, case, "random_nuisance_mean_preserving")
        lines.append(
            f"- `{case}`: {status}; raw={raw:.3f}, mode={mode:.3f}, "
            f"full_mean={full:.3f}, full_classic={classic:.3f}, random={random_control:.3f}"
        )
    lines.extend(
        [
            "",
            "## Leakage Trap",
            "",
            (
                "The leakage case puts the nuisance-to-embedding relationship only in held-out "
                "test speakers. Because the design scaler and OLS residualizer are fit on train "
                "speaker utterances only, the full nuisance variants are expected not to improve "
                "over raw by more than +0.20 absolute R@1."
            ),
        ]
    )
    (out_dir / "README_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_yaml(path: Path, data: Dict[str, object]) -> None:
    try:
        import yaml  # type: ignore

        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return
    except Exception:
        pass

    def scalar(value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        return json.dumps(str(value))

    def emit(value: object, indent: int = 0) -> List[str]:
        pad = " " * indent
        if isinstance(value, dict):
            lines: List[str] = []
            for key, item in value.items():
                if isinstance(item, (dict, list)):
                    lines.append(f"{pad}{key}:")
                    lines.extend(emit(item, indent + 2))
                else:
                    lines.append(f"{pad}{key}: {scalar(item)}")
            return lines
        if isinstance(value, list):
            lines = []
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f"{pad}-")
                    lines.extend(emit(item, indent + 2))
                else:
                    lines.append(f"{pad}- {scalar(item)}")
            return lines
        return [f"{pad}{scalar(value)}"]

    path.write_text("\n".join(emit(data)) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CPU-only synthetic gates for identity residualized retrieval."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--speakers", type=int, default=48)
    args = parser.parse_args()

    try:
        _rows, all_ok = run_synthetic_gate(args.out_dir, args.seed, args.speakers)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"wrote synthetic gate outputs -> {args.out_dir}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
