from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from probing.run_identity_residual_matched_frames import choose_matched_crops, crop_starts, frame_vector


class MatchedFrameControlTests(unittest.TestCase):
    def test_crop_starts_respect_fixed_frame_length(self) -> None:
        starts = crop_starts(17, 5, 8, np.random.default_rng(7))
        self.assertTrue(len(starts))
        self.assertTrue(np.all(starts >= 0))
        self.assertTrue(np.all(starts + 5 <= 17))

    def test_crop_selection_matches_voiced_ratios_and_frame_count(self) -> None:
        speech = np.arange(20, dtype=float).reshape(10, 2)
        singing = speech + 100.0
        speech_voiced = np.asarray([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
        singing_voiced = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 0], dtype=bool)
        selected = choose_matched_crops(speech, speech_voiced, singing, singing_voiced, 4, 7, np.random.default_rng(3))
        self.assertIsNotNone(selected)
        s_crop, g_crop, info = selected  # type: ignore[misc]
        self.assertEqual(s_crop.shape, (4, 2))
        self.assertEqual(g_crop.shape, (4, 2))
        self.assertLessEqual(float(info["voiced_ratio_abs_difference"]), 0.25)

    def test_too_short_pair_is_excluded(self) -> None:
        x = np.zeros((3, 2))
        selected = choose_matched_crops(x, np.ones(3, dtype=bool), x, np.ones(3, dtype=bool), 4, 3, np.random.default_rng(1))
        self.assertIsNone(selected)

    def test_frame_vector_contains_mean_and_standard_deviation(self) -> None:
        x = np.asarray([[1.0, 3.0], [3.0, 7.0]])
        np.testing.assert_allclose(frame_vector(x), np.asarray([2.0, 5.0, 1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
