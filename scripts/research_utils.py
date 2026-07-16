from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


UTTERANCE_COLUMNS = [
    "utt_id",
    "speaker_id",
    "language",
    "vocal_range",
    "mode",
    "technique",
    "song_id",
    "phrase_id",
    "take_id",
    "wav_path",
    "start_sec",
    "end_sec",
    "duration_sec",
    "sample_rate",
    "num_samples",
    "text",
    "phone_seq",
    "alignment_path",
    "paired_utt_id",
    "control_utt_id",
    "split_group_key",
    "snr_db",
    "rms_db",
    "f0_mean_hz",
    "f0_std_hz",
    "f0_min_hz",
    "f0_max_hz",
    "f0_voiced_pct",
    "energy_mean",
    "energy_std",
    "alignment_quality_flag",
]

PAIR_COLUMNS = [
    "pair_id",
    "speech_utt_id",
    "singing_utt_id",
    "speaker_id",
    "language",
    "song_id",
    "phrase_id",
    "technique",
    "pair_type",
    "same_text_flag",
    "same_song_flag",
]

PHONE_EXAMPLE_COLUMNS = [
    "phone_ex_id",
    "utt_id",
    "speaker_id",
    "language",
    "mode",
    "technique",
    "song_id",
    "phrase_id",
    "phone",
    "phone_start_sec",
    "phone_end_sec",
    "phone_duration_sec",
]

TECHNIQUE_PAIR_COLUMNS = [
    "pair_id",
    "base_phone_ex_id",
    "technique_phone_ex_id",
    "base_utt_id",
    "technique_utt_id",
    "speaker_id",
    "language",
    "phone",
    "base_technique",
    "target_technique",
    "song_id",
    "phrase_id",
    "base_phone_start_sec",
    "base_phone_end_sec",
    "technique_phone_start_sec",
    "technique_phone_end_sec",
]

FEATURE_KEYS = ["x", "times_sec", "voiced_mask", "phone_id", "f0_hz", "energy"]

NUMERIC_DEFAULTS = {
    "start_sec": 0.0,
    "end_sec": 0.0,
    "duration_sec": 0.0,
    "sample_rate": 0,
    "num_samples": 0,
    "snr_db": float("nan"),
    "rms_db": float("nan"),
    "f0_mean_hz": float("nan"),
    "f0_std_hz": float("nan"),
    "f0_min_hz": float("nan"),
    "f0_max_hz": float("nan"),
    "f0_voiced_pct": float("nan"),
    "energy_mean": float("nan"),
    "energy_std": float("nan"),
}


class ExperimentError(RuntimeError):
    pass


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=13)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_table(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise ExperimentError(
                "Reading Parquet requires pandas and pyarrow. Install them or use JSONL/CSV."
            ) from exc
        return pd.read_parquet(path).to_dict("records")
    raise ExperimentError(f"Unsupported table format: {path}")


def write_table(rows: Iterable[dict[str, Any]], path: Path) -> None:
    rows = list(rows)
    ensure_parent(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=True) + "\n")
        return
    if suffix == ".csv":
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return
    if suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise ExperimentError(
                "Writing Parquet requires pandas and pyarrow. Install them or use JSONL/CSV."
            ) from exc
        pd.DataFrame(rows).to_parquet(path, index=False)
        return
    raise ExperimentError(f"Unsupported table format: {path}")


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coerce_row(row: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "speaker_id" in columns and not row.get("speaker_id") and row.get("singer_id"):
        row = {**row, "speaker_id": row["singer_id"]}
    for col in columns:
        value = row.get(col, NUMERIC_DEFAULTS.get(col, ""))
        if col in NUMERIC_DEFAULTS:
            try:
                out[col] = float(value) if "." in str(value) or "nan" in str(value).lower() else int(value)
            except (TypeError, ValueError):
                out[col] = NUMERIC_DEFAULTS[col]
        else:
            out[col] = "" if value is None else str(value)
    return out


def validate_required_columns(rows: list[dict[str, Any]], columns: list[str], name: str) -> None:
    if not rows:
        raise ExperimentError(f"{name} has no rows")
    missing = sorted(set(columns) - set(rows[0]))
    if missing:
        raise ExperimentError(f"{name} missing required columns: {', '.join(missing)}")


def validate_utterances(rows: list[dict[str, Any]], require_audio_exists: bool = False) -> dict[str, Any]:
    validate_required_columns(rows, UTTERANCE_COLUMNS, "utterance manifest")
    seen: set[str] = set()
    errors: list[str] = []
    for i, row in enumerate(rows):
        utt_id = str(row["utt_id"])
        if not utt_id:
            errors.append(f"row {i}: empty utt_id")
        if utt_id in seen:
            errors.append(f"duplicate utt_id: {utt_id}")
        seen.add(utt_id)
        if not row["speaker_id"]:
            errors.append(f"{utt_id}: missing speaker_id")
        if row["mode"] not in {"speech", "singing", "singing_control", "singing_technique"}:
            errors.append(f"{utt_id}: invalid mode {row['mode']!r}")
        if require_audio_exists and row["wav_path"] and not Path(row["wav_path"]).exists():
            errors.append(f"{utt_id}: wav_path does not exist: {row['wav_path']}")
    if errors:
        raise ExperimentError("; ".join(errors[:20]))
    return manifest_summary(rows)


def validate_pairs(rows: list[dict[str, Any]], utterances: list[dict[str, Any]]) -> dict[str, Any]:
    validate_required_columns(rows, PAIR_COLUMNS, "pair manifest")
    utt_ids = {row["utt_id"] for row in utterances}
    errors = []
    for row in rows:
        for col in ("speech_utt_id", "singing_utt_id"):
            if row[col] not in utt_ids:
                errors.append(f"{row['pair_id']}: missing {col}={row[col]}")
    if errors:
        raise ExperimentError("; ".join(errors[:20]))
    return {"pairs": len(rows), "pair_types": dict(Counter(row["pair_type"] for row in rows))}


def manifest_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    speakers = sorted({row["speaker_id"] for row in rows})
    modes = Counter(row["mode"] for row in rows)
    techniques = Counter(row["technique"] for row in rows)
    languages = Counter(row["language"] for row in rows)
    return {
        "num_utterances": len(rows),
        "num_speakers": len(speakers),
        "modes": dict(modes),
        "techniques": dict(techniques),
        "languages": dict(languages),
        "manifest_hash": stable_json_hash(rows),
    }


def validate_feature_npz(path: Path) -> tuple[int, int]:
    with np.load(path) as data:
        missing = [key for key in FEATURE_KEYS if key not in data]
        if missing:
            raise ExperimentError(f"{path} missing feature keys: {missing}")
        x = np.asarray(data["x"])
        if x.ndim != 2:
            raise ExperimentError(f"{path} x must be [T, D], got shape {x.shape}")
        if not np.isfinite(x).all():
            raise ExperimentError(f"{path} x contains NaN or inf")
        t = x.shape[0]
        for key in FEATURE_KEYS[1:]:
            arr = np.asarray(data[key])
            if arr.shape[0] != t:
                raise ExperimentError(f"{path} {key} length {arr.shape[0]} != x frames {t}")
    return int(x.shape[0]), int(x.shape[1])


def feature_path(root: Path, extractor: str, checkpoint_hash: str, layer: str, utt_id: str) -> Path:
    return root / extractor / checkpoint_hash / str(layer) / f"{utt_id}.npz"


def load_feature_vector(
    feature_root: Path,
    extractor: str,
    checkpoint_hash: str,
    layer: str,
    utt_id: str,
    voiced_only: bool = False,
) -> np.ndarray:
    path = feature_path(feature_root, extractor, checkpoint_hash, layer, utt_id)
    if not path.exists():
        raise ExperimentError(f"Missing feature file: {path}")
    validate_feature_npz(path)
    with np.load(path) as data:
        x = np.asarray(data["x"], dtype=np.float64)
        if voiced_only:
            mask = np.asarray(data["voiced_mask"], dtype=bool)
            if mask.any():
                x = x[mask]
        mean = x.mean(axis=0)
        std = x.std(axis=0)
    return np.concatenate([mean, std]).astype(np.float64)


def load_feature_interval_vector(
    feature_root: Path,
    extractor: str,
    checkpoint_hash: str,
    layer: str,
    utt_id: str,
    start_sec: float,
    end_sec: float,
    voiced_only: bool = False,
) -> np.ndarray:
    path = feature_path(feature_root, extractor, checkpoint_hash, layer, utt_id)
    if not path.exists():
        raise ExperimentError(f"Missing feature file: {path}")
    validate_feature_npz(path)
    with np.load(path) as data:
        x = np.asarray(data["x"], dtype=np.float64)
        times = np.asarray(data["times_sec"], dtype=np.float64)
        mask = (times >= float(start_sec)) & (times <= float(end_sec))
        if voiced_only:
            mask = mask & np.asarray(data["voiced_mask"], dtype=bool)
        if not mask.any():
            center = 0.5 * (float(start_sec) + float(end_sec))
            nearest = int(np.argmin(np.abs(times - center)))
            mask = np.zeros_like(times, dtype=bool)
            mask[nearest] = True
        x = x[mask]
        mean = x.mean(axis=0)
        std = x.std(axis=0)
    return np.concatenate([mean, std]).astype(np.float64)


def build_matrix(
    rows: list[dict[str, Any]],
    feature_root: Path,
    extractor: str,
    checkpoint_hash: str,
    layer: str,
    voiced_only: bool = False,
) -> tuple[np.ndarray, list[str]]:
    vectors = []
    utt_ids = []
    for row in rows:
        vectors.append(load_feature_vector(feature_root, extractor, checkpoint_hash, layer, row["utt_id"], voiced_only))
        utt_ids.append(row["utt_id"])
    return np.vstack(vectors), utt_ids


def nuisance_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return zscore(raw_nuisance_matrix(rows))


def raw_nuisance_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    cols = ["f0_mean_hz", "f0_std_hz", "f0_voiced_pct", "energy_mean", "energy_std", "duration_sec", "rms_db"]
    mat = []
    for row in rows:
        vals = []
        for col in cols:
            try:
                vals.append(float(row[col]))
            except (TypeError, ValueError):
                vals.append(float("nan"))
        mat.append(vals)
    arr = np.asarray(mat, dtype=np.float64)
    col_means = np.nanmean(arr, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    inds = np.where(~np.isfinite(arr))
    arr[inds] = np.take(col_means, inds[1])
    return arr


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / np.where(std < 1e-8, 1.0, std)


def residualize(x: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    z = np.column_stack([np.ones(nuisance.shape[0]), nuisance])
    beta = np.linalg.pinv(z) @ x
    return x - z @ beta


def fit_nuisance_residualizer(x_train: np.ndarray, nuisance_train: np.ndarray) -> dict[str, np.ndarray]:
    nuisance_train = np.asarray(nuisance_train, dtype=np.float64)
    mean = nuisance_train.mean(axis=0, keepdims=True)
    std = nuisance_train.std(axis=0, keepdims=True)
    z_train = (nuisance_train - mean) / np.where(std < 1e-8, 1.0, std)
    z_aug = np.column_stack([np.ones(z_train.shape[0]), z_train])
    beta = np.linalg.pinv(z_aug) @ np.asarray(x_train, dtype=np.float64)
    return {"mean": mean, "std": std, "beta": beta}


def apply_nuisance_residualizer(x: np.ndarray, nuisance: np.ndarray, residualizer: dict[str, np.ndarray]) -> np.ndarray:
    nuisance = np.asarray(nuisance, dtype=np.float64)
    z = (nuisance - residualizer["mean"]) / np.where(residualizer["std"] < 1e-8, 1.0, residualizer["std"])
    z_aug = np.column_stack([np.ones(z.shape[0]), z])
    return np.asarray(x, dtype=np.float64) - z_aug @ residualizer["beta"]


def deterministic_speaker_split(
    rows: list[dict[str, Any]],
    seed: int,
    train_frac: float = 0.7,
    dev_frac: float = 0.15,
) -> dict[str, str]:
    speakers = sorted({row["speaker_id"] for row in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(speakers)
    n = len(speakers)
    n_train = max(1, int(round(n * train_frac)))
    n_dev = max(1, int(round(n * dev_frac))) if n >= 3 else 0
    split = {}
    for i, speaker in enumerate(speakers):
        if i < n_train:
            split[speaker] = "train"
        elif i < n_train + n_dev:
            split[speaker] = "dev"
        else:
            split[speaker] = "test"
    if "test" not in split.values() and n > 1:
        split[speakers[-1]] = "test"
    return split


def kfold_speaker_splits(rows: list[dict[str, Any]], folds: int, seed: int) -> list[dict[str, str]]:
    speakers = sorted({row["speaker_id"] for row in rows})
    if folds < 2:
        return [deterministic_speaker_split(rows, seed)]
    folds = min(folds, len(speakers))
    rng = np.random.default_rng(seed)
    rng.shuffle(speakers)
    buckets = [list(bucket) for bucket in np.array_split(np.asarray(speakers, dtype=object), folds)]
    splits = []
    for fold_idx, test_speakers in enumerate(buckets):
        dev_speakers = set(buckets[(fold_idx + 1) % folds]) if folds > 2 else set()
        test_set = set(test_speakers)
        fold_split = {}
        for speaker in speakers:
            if speaker in test_set:
                fold_split[speaker] = "test"
            elif speaker in dev_speakers:
                fold_split[speaker] = "dev"
            else:
                fold_split[speaker] = "train"
        splits.append(fold_split)
    return splits


def assert_no_speaker_leakage(rows: list[dict[str, Any]], splits: np.ndarray) -> None:
    by_split: dict[str, set[str]] = defaultdict(set)
    for row, split in zip(rows, splits):
        by_split[str(split)].add(str(row["speaker_id"]))
    train = by_split.get("train", set())
    dev = by_split.get("dev", set())
    test = by_split.get("test", set())
    if train & test:
        raise ExperimentError(f"speaker leakage train/test: {sorted(train & test)}")
    if train & dev:
        raise ExperimentError(f"speaker leakage train/dev: {sorted(train & dev)}")


def deterministic_group_split(
    rows: list[dict[str, Any]],
    group_key: str,
    seed: int,
    train_frac: float = 0.7,
    dev_frac: float = 0.15,
) -> dict[str, str]:
    groups = sorted({str(row[group_key]) for row in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_train = max(1, int(round(n * train_frac)))
    n_dev = max(1, int(round(n * dev_frac))) if n >= 3 else 0
    split = {}
    for i, group in enumerate(groups):
        if i < n_train:
            split[group] = "train"
        elif i < n_train + n_dev:
            split[group] = "dev"
        else:
            split[group] = "test"
    if "test" not in split.values() and n > 1:
        split[groups[-1]] = "test"
    return split


def assert_no_group_leakage(rows: list[dict[str, Any]], splits: np.ndarray, group_key: str) -> None:
    by_split: dict[str, set[str]] = defaultdict(set)
    for row, split in zip(rows, splits):
        by_split[str(split)].add(str(row[group_key]))
    train = by_split.get("train", set())
    dev = by_split.get("dev", set())
    test = by_split.get("test", set())
    if train & test:
        raise ExperimentError(f"{group_key} leakage train/test: {sorted(train & test)}")
    if train & dev:
        raise ExperimentError(f"{group_key} leakage train/dev: {sorted(train & dev)}")


def binary_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=np.float64)
    pos = score[y_true == 1]
    neg = score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for value in pos:
        wins += np.sum(value > neg) + 0.5 * np.sum(value == neg)
    return float(wins / (len(pos) * len(neg)))


def auc_summary(y_true: np.ndarray, score: np.ndarray) -> dict[str, float | str]:
    signed = binary_auc(y_true, score)
    if not np.isfinite(signed):
        return {"signed_auc": signed, "separability_auc": signed, "orientation": "undefined"}
    if signed < 0.5:
        return {"signed_auc": signed, "separability_auc": 1.0 - signed, "orientation": "reversed"}
    return {"signed_auc": signed, "separability_auc": signed, "orientation": "aligned"}


def fit_logistic_regression(x: np.ndarray, y: np.ndarray, seed: int = 13, steps: int = 1200, lr: float = 0.05) -> tuple[np.ndarray, float]:
    x = zscore(x)
    y = y.astype(np.float64)
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore

        clf = LogisticRegression(
            C=10.0,
            solver="liblinear",
            max_iter=max(100, min(500, steps)),
            tol=1e-4,
            random_state=seed,
        )
        clf.fit(x, y.astype(int))
        return clf.coef_.reshape(-1).astype(np.float64), float(clf.intercept_[0])
    except Exception:
        pass

    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.01, size=x.shape[1])
    b = 0.0
    for _ in range(steps):
        logits = np.clip(x @ w + b, -40.0, 40.0)
        p = 1.0 / (1.0 + np.exp(-logits))
        err = p - y
        w -= lr * ((x.T @ err) / len(y) + 1e-4 * w)
        b -= lr * float(err.mean())
    return w, b


def predict_logistic(x_train: np.ndarray, x_test: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    x = (x_test - mean) / np.where(std < 1e-8, 1.0, std)
    return 1.0 / (1.0 + np.exp(-np.clip(x @ w + b, -40.0, 40.0)))


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b_norm = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a_norm @ b_norm.T


def recall_at_k(query: np.ndarray, gallery: np.ndarray, query_labels: list[str], gallery_labels: list[str], k: int) -> float:
    sims = cosine_similarity_matrix(query, gallery)
    hits = 0
    for i, label in enumerate(query_labels):
        top = np.argsort(-sims[i])[:k]
        hits += int(label in {gallery_labels[j] for j in top})
    return hits / max(1, len(query_labels))


def mean_average_precision(query: np.ndarray, gallery: np.ndarray, query_labels: list[str], gallery_labels: list[str]) -> float:
    sims = cosine_similarity_matrix(query, gallery)
    ap_values = []
    for i, label in enumerate(query_labels):
        order = np.argsort(-sims[i])
        hits = 0
        precision_sum = 0.0
        for rank, gallery_idx in enumerate(order, start=1):
            if gallery_labels[gallery_idx] == label:
                hits += 1
                precision_sum += hits / rank
        if hits:
            ap_values.append(precision_sum / hits)
        else:
            ap_values.append(0.0)
    return float(np.mean(ap_values)) if ap_values else 0.0


def angular_degrees(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return float("nan")
    value = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(math.degrees(math.acos(value)))


def write_json(value: Any, path: Path) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return dict(groups)


def parse_layer(value: str) -> str:
    return str(value).strip()


def current_git_commit() -> str:
    head = Path(".git/HEAD")
    if not head.exists():
        return "unknown"
    content = head.read_text(encoding="utf-8").strip()
    if content.startswith("ref:"):
        ref = Path(".git") / content.split(" ", 1)[1]
        return ref.read_text(encoding="utf-8").strip() if ref.exists() else "unknown"
    return content


def require_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed
