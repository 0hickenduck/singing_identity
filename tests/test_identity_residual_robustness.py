from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

from singing_identity.evaluation.metrics import (
    eer,
    roc_det_points,
    train_threshold_at_fmr,
    verification_metrics,
)


class IdentityResidualRobustnessTests(unittest.TestCase):
    def test_train_threshold_respects_requested_fmr(self) -> None:
        impostor = np.asarray([0.1, 0.2, 0.3, 0.9])
        threshold, observed = train_threshold_at_fmr(impostor, 0.25)
        self.assertLessEqual(observed, 0.25)
        self.assertGreaterEqual(threshold, 0.9)

    def test_verification_tmr_is_calibrated_from_train_trials(self) -> None:
        query = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        gallery = query.copy()
        # Train impostors force the 1% threshold above 0.9. Test labels are not
        # consulted when selecting that operating threshold.
        calibration_query = np.asarray([[1.0, 0.0], [0.9, 0.1]])
        calibration_gallery = np.asarray([[1.0, 0.0], [0.9, 0.1]])
        metrics = verification_metrics(query, gallery, calibration_query, calibration_gallery)
        self.assertEqual(metrics["TMR_at_FMR_1pct_status"], "train_calibrated")
        self.assertGreater(float(metrics["TMR_at_FMR_1pct_train_threshold"]), 0.9)
        self.assertEqual(metrics["R1"], 1.0)

    def test_eer_and_roc_points_are_well_formed(self) -> None:
        genuine = np.asarray([0.9, 0.8, 0.95])
        impostor = np.asarray([0.1, 0.2, 0.3, 0.4])
        self.assertLess(eer(genuine, impostor), 0.1)
        points = roc_det_points(genuine, impostor)
        self.assertTrue(points)
        self.assertTrue(all(0.0 <= point["FMR"] <= 1.0 for point in points))
        self.assertTrue(all(0.0 <= point["TMR"] <= 1.0 for point in points))


if __name__ == "__main__":
    unittest.main()
