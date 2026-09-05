from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class IdentityResidualSyntheticTest(unittest.TestCase):
    def test_synthetic_gate_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "identity_residual_synthetic"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "singing_identity.probing.synthetic_gate",
                    "--out-dir",
                    str(out_dir),
                    "--seed",
                    "13",
                    "--speakers",
                    "24",
                ],
                cwd=ROOT,
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

            summary_path = out_dir / "summary.csv"
            self.assertTrue(summary_path.exists())
            self.assertTrue((out_dir / "README_results.md").exists())
            self.assertTrue((out_dir / "experiment_card.yaml").exists())

            with summary_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(
                {
                    "no_nuisance_mode_dominance",
                    "mode_nuisance_dominance",
                    "mode_proxy",
                    "within_mode_nuisance",
                    "random_nuisance",
                    "leakage_trap",
                },
                {row["case"] for row in rows},
            )
            self.assertTrue(rows)
            self.assertTrue(all(row["pass_expected"] == "true" for row in rows))


if __name__ == "__main__":
    unittest.main()
