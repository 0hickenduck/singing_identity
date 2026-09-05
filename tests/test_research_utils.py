from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from singing_identity.utils.research_utils import (
    ExperimentError,
    FEATURE_KEYS,
    UTTERANCE_COLUMNS,
    apply_nuisance_residualizer,
    assert_no_group_leakage,
    assert_no_speaker_leakage,
    auc_summary,
    binary_auc,
    deterministic_group_split,
    fit_nuisance_residualizer,
    kfold_speaker_splits,
    load_feature_interval_vector,
    mean_average_precision,
    residualize,
    validate_feature_npz,
    validate_utterances,
)


class ResearchUtilsTest(unittest.TestCase):
    def test_manifest_required_columns(self) -> None:
        row = {col: "" for col in UTTERANCE_COLUMNS}
        row.update({"utt_id": "u1", "speaker_id": "s1", "mode": "speech"})
        summary = validate_utterances([row])
        self.assertEqual(summary["num_utterances"], 1)

    def test_manifest_missing_column_fails(self) -> None:
        with self.assertRaises(ExperimentError):
            validate_utterances([{"utt_id": "u1"}])

    def test_feature_cache_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "u1.npz"
            frames = 5
            np.savez(
                path,
                x=np.ones((frames, 3), dtype=np.float32),
                times_sec=np.arange(frames, dtype=np.float32),
                voiced_mask=np.ones(frames, dtype=bool),
                phone_id=np.ones(frames, dtype=np.int32),
                f0_hz=np.ones(frames, dtype=np.float32),
                energy=np.ones(frames, dtype=np.float32),
            )
            self.assertEqual(validate_feature_npz(path), (5, 3))

    def test_feature_cache_missing_key_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.npz"
            np.savez(path, x=np.ones((2, 2), dtype=np.float32))
            with self.assertRaises(ExperimentError):
                validate_feature_npz(path)

    def test_binary_auc(self) -> None:
        auc = binary_auc(np.asarray([0, 0, 1, 1]), np.asarray([0.1, 0.2, 0.8, 0.9]))
        self.assertEqual(auc, 1.0)

    def test_mean_average_precision(self) -> None:
        query = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        gallery = np.asarray([[0.0, 1.0], [1.0, 0.0]])
        self.assertEqual(mean_average_precision(query, gallery, ["a", "b"], ["b", "a"]), 1.0)

    def test_auc_summary_tracks_reversed_orientation(self) -> None:
        summary = auc_summary(np.asarray([0, 0, 1, 1]), np.asarray([0.9, 0.8, 0.2, 0.1]))
        self.assertEqual(summary["signed_auc"], 0.0)
        self.assertEqual(summary["separability_auc"], 1.0)
        self.assertEqual(summary["orientation"], "reversed")

    def test_residualize_removes_linear_nuisance(self) -> None:
        z = np.arange(10, dtype=np.float64)[:, None]
        x = np.column_stack([2.0 * z[:, 0] + 1.0, np.ones(10)])
        r = residualize(x, z)
        self.assertLess(float(np.linalg.norm(r)), 1e-10)

    def test_train_only_nuisance_residualizer_removes_train_component(self) -> None:
        z_train = np.arange(10, dtype=np.float64)[:, None]
        x_train = np.column_stack([2.0 * z_train[:, 0] + 1.0])
        residualizer = fit_nuisance_residualizer(x_train, z_train)
        r_train = apply_nuisance_residualizer(x_train, z_train, residualizer)
        self.assertLess(float(np.linalg.norm(r_train)), 1e-10)

    def test_kfold_split_has_no_speaker_leakage(self) -> None:
        rows = []
        for speaker in range(6):
            for item in range(2):
                rows.append({"speaker_id": f"s{speaker}", "utt_id": f"s{speaker}_{item}"})
        for split_map in kfold_speaker_splits(rows, folds=3, seed=1):
            splits = np.asarray([split_map[row["speaker_id"]] for row in rows])
            assert_no_speaker_leakage(rows, splits)

    def test_group_split_has_no_song_leakage(self) -> None:
        rows = []
        for song in range(8):
            for item in range(2):
                rows.append({"song_id": f"song{song}", "utt_id": f"song{song}_{item}"})
        split_map = deterministic_group_split(rows, "song_id", seed=2)
        splits = np.asarray([split_map[row["song_id"]] for row in rows])
        assert_no_group_leakage(rows, splits, "song_id")

    def test_feature_interval_uses_phone_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "toy" / "ckpt" / "l0" / "u1.npz"
            path.parent.mkdir(parents=True)
            x = np.vstack([np.zeros((5, 2), dtype=np.float32), np.ones((5, 2), dtype=np.float32) * 10])
            np.savez(
                path,
                x=x,
                times_sec=np.linspace(0.0, 0.9, 10, dtype=np.float32),
                voiced_mask=np.ones(10, dtype=bool),
                phone_id=np.ones(10, dtype=np.int32),
                f0_hz=np.ones(10, dtype=np.float32),
                energy=np.ones(10, dtype=np.float32),
            )
            first = load_feature_interval_vector(root, "toy", "ckpt", "l0", "u1", 0.0, 0.35)
            second = load_feature_interval_vector(root, "toy", "ckpt", "l0", "u1", 0.55, 0.9)
            self.assertLess(float(first[0]), 1.0)
            self.assertGreater(float(second[0]), 9.0)


if __name__ == "__main__":
    unittest.main()
