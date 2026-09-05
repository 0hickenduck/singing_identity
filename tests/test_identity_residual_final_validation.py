from __future__ import annotations

import unittest

import numpy as np

from singing_identity.methods.identity_residual import (
    add_global_rows,
    mode_probe_centroid,
    paired_split_delta_summaries,
    restricted_retrieval,
)
from singing_identity.evaluation.metrics import (
    retrieval_chance_summary,
)


class IdentityResidualFinalValidationTest(unittest.TestCase):
    def test_logistic_probe_threshold_is_train_only(self) -> None:
        train = np.asarray([[-2.0, 0.0], [-1.0, 0.2], [1.0, -0.1], [2.0, 0.0]])
        y = np.asarray([0, 0, 1, 1, 0, 1])
        train_idx = np.arange(4)
        test_idx = np.asarray([4, 5])
        first = mode_probe_centroid(
            np.vstack([train, [[-1.5, 10.0], [1.5, -10.0]]]), y, train_idx, test_idx
        )
        second = mode_probe_centroid(
            np.vstack([train, [[1000.0, -500.0], [-1000.0, 500.0]]]), y, train_idx, test_idx
        )

        self.assertAlmostEqual(first["decision_threshold"], second["decision_threshold"])
        self.assertAlmostEqual(first["coefficient_norm"], second["coefficient_norm"])
        self.assertEqual(first["AUC"], 1.0)

    def test_restricted_chance_uses_each_query_gallery_size(self) -> None:
        labels = ["a", "b", "c", "d", "e"]
        attributes = {"a": "small", "b": "small", "c": "large", "d": "large", "e": "large"}
        metrics = restricted_retrieval(np.eye(5), np.eye(5), labels, attributes, "group")
        chance = retrieval_chance_summary(metrics)

        self.assertEqual(metrics["gallery_sizes"], [2, 2, 3, 3, 3])
        self.assertAlmostEqual(chance["chance_R1_mean_per_query"], 0.4)
        self.assertNotAlmostEqual(chance["chance_R1"], 1.0 / metrics["den"])
        self.assertEqual(chance["gallery_size_distribution"], "2:2;3:3")

    def test_per_speaker_rows_pair_raw_and_corrected_rank_details(self) -> None:
        rows = []
        per_speaker = []
        speakers = ["train_a", "train_b", "test_a", "test_b"]
        speech = np.asarray([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
        singing = np.asarray([[1.0, 1.0], [-1.0, 1.0], [1.0, 1.0], [-1.0, 1.0]])
        split = {speaker: ("train" if speaker.startswith("train") else "test") for speaker in speakers}

        add_global_rows(
            rows,
            per_speaker,
            "toy",
            "toy_model",
            "0",
            13,
            split,
            speakers,
            speech,
            singing,
            np.asarray([0, 1]),
            np.asarray([2, 3]),
            np.random.default_rng(13),
            0,
        )

        self.assertEqual(len(per_speaker), 2)
        for row in per_speaker:
            self.assertIn("raw_rank", row)
            self.assertIn("corrected_rank", row)
            self.assertEqual(
                row["rank_improvement_raw_minus_corrected"], row["raw_rank"] - row["corrected_rank"]
            )
            self.assertTrue(row["nearest_impostor_before"])
            self.assertTrue(row["nearest_impostor_after"])

    def test_paired_delta_summary_uses_matching_split_seeds(self) -> None:
        common = {"dataset": "toy", "model": "model", "layer": "0"}
        rows = [
            {**common, "variant": "raw", "split_seed": 1, "R1": 0.2},
            {**common, "variant": "corrected", "split_seed": 1, "R1": 0.4},
            {**common, "variant": "raw", "split_seed": 2, "R1": 0.5},
            {**common, "variant": "corrected", "split_seed": 2, "R1": 0.4},
            {**common, "variant": "corrected", "split_seed": 3, "R1": 1.0},
        ]

        summary = paired_split_delta_summaries(
            rows, [("corrected", "raw")], bootstrap_samples=1000, random_seed=7
        )[0]

        self.assertEqual(summary["paired_splits"], 2)
        self.assertEqual(summary["paired_split_seeds"], "1;2")
        self.assertAlmostEqual(summary["delta_R1_mean"], 0.05)
        self.assertLessEqual(summary["delta_R1_CI_low"], summary["delta_R1_mean"])
        self.assertGreaterEqual(summary["delta_R1_CI_high"], summary["delta_R1_mean"])


if __name__ == "__main__":
    unittest.main()
