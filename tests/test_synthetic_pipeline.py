from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SyntheticPipelineTest(unittest.TestCase):
    def run_cmd(self, *args: str) -> None:
        subprocess.run([sys.executable, *args], cwd=ROOT, check=True)

    def test_track1_and_track2_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "synthetic"
            self.run_cmd(
                 "-m",
                "singing_identity.data.synthetic",
                "--root",
                str(root),
                "--speakers",
                "8",
                "--items-per-mode",
                "2",
            )
            config = json.loads((root / "synthetic_run_config.json").read_text(encoding="utf-8"))
            self.run_cmd(
                 "-m",
                "singing_identity.data.features",
                "--manifest",
                config["track1_manifest"],
                "--feature-root",
                config["feature_root"],
                "--extractor",
                config["extractor"],
                "--checkpoint-hash",
                config["checkpoint_hash"],
                "--layer",
                config["layer"],
                "--report",
                str(root / "validate_features.json"),
            )
            self.run_cmd(
                 "-m",
                "singing_identity.probing.mode_probe",
                "--manifest",
                config["track1_manifest"],
                "--feature-root",
                config["feature_root"],
                "--extractor",
                config["extractor"],
                "--checkpoint-hash",
                config["checkpoint_hash"],
                "--layer",
                config["layer"],
                "--metrics-out",
                str(root / "mode_metrics.json"),
                "--predictions-out",
                str(root / "mode_predictions.jsonl"),
            )
            self.run_cmd(
                 "-m",
                "singing_identity.probing.residual_control",
                "--manifest",
                config["track1_manifest"],
                "--feature-root",
                config["feature_root"],
                "--extractor",
                config["extractor"],
                "--checkpoint-hash",
                config["checkpoint_hash"],
                "--layer",
                config["layer"],
                "--metrics-out",
                str(root / "residual_mode_metrics.json"),
                "--predictions-out",
                str(root / "residual_mode_predictions.jsonl"),
            )
            self.run_cmd(
                 "-m",
                "singing_identity.probing.speaker_retrieval",
                "--manifest",
                config["track1_manifest"],
                "--feature-root",
                config["feature_root"],
                "--extractor",
                config["extractor"],
                "--checkpoint-hash",
                config["checkpoint_hash"],
                "--layer",
                config["layer"],
                "--metrics-out",
                str(root / "retrieval_metrics.json"),
            )
            self.run_cmd(
                 "-m",
                "singing_identity.probing.technique_directions",
                "--pairs",
                config["track2_pairs"],
                "--feature-root",
                config["feature_root"],
                "--extractor",
                config["extractor"],
                "--checkpoint-hash",
                config["checkpoint_hash"],
                "--layer",
                config["layer"],
                "--metrics-out",
                str(root / "technique_metrics.json"),
                "--directions-out",
                str(root / "technique_directions.jsonl"),
                "--bootstrap",
                "10",
            )
            self.run_cmd(
                 "-m",
                "singing_identity.intervention.micro_mapper",
                "--manifest",
                config["track1_manifest"],
                "--pairs",
                config["track1_pairs"],
                "--feature-root",
                config["feature_root"],
                "--extractor",
                config["extractor"],
                "--checkpoint-hash",
                config["checkpoint_hash"],
                "--layer",
                config["layer"],
                "--metrics-out",
                str(root / "mapper_metrics.json"),
                "--predictions-out",
                str(root / "mapper_predictions.jsonl"),
                "--model-out",
                str(root / "mapper_model.npz"),
            )
            self.run_cmd(
                 "-m",
                "singing_identity.intervention.seedvc_inject",
                "--manifest",
                config["track1_manifest"],
                "--pairs",
                config["track1_pairs"],
                "--feature-root",
                config["feature_root"],
                "--extractor",
                config["extractor"],
                "--checkpoint-hash",
                config["checkpoint_hash"],
                "--layer",
                config["layer"],
                "--mapper-model",
                str(root / "mapper_model.npz"),
                "--out-root",
                str(root / "track1_intervention"),
                "--manifest-out",
                str(root / "track1_intervention_manifest.jsonl"),
                "--metrics-out",
                str(root / "track1_intervention_metrics.json"),
                "--max-pairs",
                "2",
            )
            self.run_cmd(
                 "-m",
                "singing_identity.intervention.latent_steering",
                "--pairs",
                config["track2_pairs"],
                "--directions",
                str(root / "technique_directions.jsonl"),
                "--feature-root",
                config["feature_root"],
                "--extractor",
                config["extractor"],
                "--checkpoint-hash",
                config["checkpoint_hash"],
                "--layer",
                config["layer"],
                "--phone",
                "a",
                "--technique",
                "vibrato",
                "--out-root",
                str(root / "track2_steering"),
                "--manifest-out",
                str(root / "track2_steering_manifest.jsonl"),
                "--metrics-out",
                str(root / "track2_steering_metrics.json"),
                "--max-examples",
                "2",
            )


if __name__ == "__main__":
    unittest.main()
