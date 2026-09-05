"""Unit tests for the experiment provenance and reproducibility module."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from singing_identity.utils.provenance import (
    compute_file_sha256,
    get_data_provenance,
    get_execution_context,
    get_git_provenance,
    init_run_directory,
    record_run_metrics,
    resolve_feature_root,
    validate_run_provenance,
)


class ProvenanceTest(unittest.TestCase):
    def test_compute_file_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.txt"
            p.write_bytes(b"hello world\n")
            # echo -n "hello world\n" | sha256sum -> d9014c4624844aa5bac314773d6b689ad467fa4e1d1a50a1b8a99d5a95f72ff5
            digest = compute_file_sha256(p)
            self.assertEqual(digest, "a948904f2f0f479b8f8197694b30184b0d2ed1c1cd2a1ec0fb85d299a192a447")

            non_existent = Path(tmp) / "does_not_exist.txt"
            self.assertEqual(compute_file_sha256(non_existent), "")

    def test_get_git_provenance_in_repo(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        prov = get_git_provenance(repo_root=repo_root, strict=False)
        self.assertIn("commit", prov)
        self.assertIn("branch", prov)
        self.assertIn("dirty", prov)
        self.assertIsInstance(prov["dirty"], bool)
        # Commit should be 40 chars hex or unknown
        self.assertTrue(len(prov["commit"]) == 40 or prov["commit"] == "unknown")

    def test_get_git_provenance_non_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prov = get_git_provenance(repo_root=Path(tmp), strict=False)
            self.assertEqual(prov["commit"], "unknown")
            self.assertEqual(prov["branch"], "unknown")
            self.assertTrue(prov["dirty"])
            self.assertEqual(prov["status"], "no_git_repository")

    def test_resolve_feature_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            custom_path = Path(tmp) / "custom_features"
            resolved = resolve_feature_root(custom_path)
            self.assertEqual(resolved, custom_path.resolve())

        # Test env var override
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / "env_features"
            env_path.mkdir()
            old_val = os.environ.get("SINGING_IDENTITY_FEATURE_ROOT")
            try:
                os.environ["SINGING_IDENTITY_FEATURE_ROOT"] = str(env_path)
                self.assertEqual(resolve_feature_root(None), env_path.resolve())
            finally:
                if old_val is not None:
                    os.environ["SINGING_IDENTITY_FEATURE_ROOT"] = old_val
                else:
                    os.environ.pop("SINGING_IDENTITY_FEATURE_ROOT", None)

    def test_get_data_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            m_path = Path(tmp) / "manifest.jsonl"
            m_path.write_text('{"utt_id": "u1"}\n', encoding="utf-8")
            data_prov = get_data_provenance(
                dataset_name="test_dataset",
                manifest_path=m_path,
                feature_set="test_feature",
                feature_root=tmp,
            )
            self.assertEqual(data_prov["dataset"], "test_dataset")
            self.assertEqual(data_prov["manifest"], str(m_path))
            self.assertEqual(len(data_prov["manifest_sha256"]), 64)
            self.assertEqual(data_prov["feature_set"], "test_feature")

    def test_get_execution_context(self) -> None:
        ctx = get_execution_context(seed=1234, command_str="python test.py --foo bar")
        self.assertEqual(ctx["seed"], 1234)
        self.assertEqual(ctx["command"], "python test.py --foo bar")
        self.assertIn("hostname", ctx)
        self.assertIn("python_version", ctx)
        self.assertIn("platform", ctx)
        self.assertIn("package_versions", ctx)
        self.assertIn("cuda", ctx)

    def test_init_run_directory_and_record_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_runs = Path(tmp) / "runs"
            cfg = {"model": "acoustic", "batch_size": 32}
            data_prov = {"dataset": "gtsinger", "manifest_sha256": "abcdef"}

            run_dir, metadata = init_run_directory(
                base_runs_dir=base_runs,
                experiment_name="test_exp",
                approach_name="acoustic",
                run_id="run_001",
                seed=42,
                resolved_config=cfg,
                data_provenance=data_prov,
                strict_git=False,
                command_str="python test_run.py",
            )

            self.assertTrue(run_dir.exists())
            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "metadata.json").exists())
            self.assertTrue((run_dir / "command.txt").exists())
            self.assertTrue((run_dir / "logs").is_dir())
            self.assertTrue((run_dir / "artifacts").is_dir())

            saved_cfg = json.loads((run_dir / "config.json").read_text())
            self.assertEqual(saved_cfg["model"], "acoustic")

            self.assertEqual(metadata["run_id"], "run_001")
            self.assertEqual(metadata["seed"], 42)
            self.assertEqual(metadata["provenance_status"], "complete")

            # Record metrics
            metrics_payload = {"val_eer": 0.05, "val_min_dcf": 0.02}
            metrics_file = record_run_metrics(run_dir, metrics_payload, evaluation_context={"split": "val"})
            self.assertTrue(metrics_file.exists())
            saved_metrics = json.loads(metrics_file.read_text())
            self.assertEqual(saved_metrics["metrics"]["val_eer"], 0.05)

            # Validate complete run
            val_res = validate_run_provenance(run_dir)
            self.assertEqual(val_res["status"], "complete")
            self.assertTrue(val_res["valid"])
            self.assertEqual(val_res["errors"], [])

    def test_validate_run_provenance_failure_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad_dir = Path(tmp) / "bad_run"
            bad_dir.mkdir()

            # Empty run directory
            val_empty = validate_run_provenance(bad_dir)
            self.assertFalse(val_empty["valid"])
            self.assertEqual(val_empty["status"], "unknown")
            self.assertTrue(any("Missing config.json" in e for e in val_empty["errors"]))

            # Historical run with experiment_card.yaml
            hist_dir = Path(tmp) / "hist_run"
            hist_dir.mkdir()
            (hist_dir / "config.json").write_text("{}", encoding="utf-8")
            (hist_dir / "experiment_card.yaml").write_text("git_commit: '1234567'\n", encoding="utf-8")
            (hist_dir / "metrics.json").write_text('{"metrics": {}}', encoding="utf-8")

            val_hist = validate_run_provenance(hist_dir)
            self.assertEqual(val_hist["status"], "partial")
            self.assertTrue(val_hist["valid"])
            self.assertTrue(any("Historical run" in w for w in val_hist["warnings"]))


if __name__ == "__main__":
    unittest.main()
