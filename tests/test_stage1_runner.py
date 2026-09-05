from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from singing_identity.runner import Stage1Runner


ROOT = Path(__file__).resolve().parents[1]


class Stage1RunnerTest(unittest.TestCase):
    def test_synthetic_smoke_report_is_generated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            report = Path(tmp) / "report.md"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/run/run_stage1.py",
                    "--synthetic",
                    "--smoke-only",
                    "--run-root",
                    str(run_root),
                    "--report",
                    str(report),
                    "--python",
                    sys.executable,
                    "--models",
                    "acoustic",
                ],
                cwd=ROOT,
                check=True,
            )
            self.assertTrue(report.exists())
            text = report.read_text(encoding="utf-8")
            self.assertIn("Stage 1 Overnight Report", text)
            self.assertIn("Successful jobs", text)

    def test_failed_optional_job_records_error_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                run_root=Path(tmp) / "run",
                report=Path(tmp) / "report.md",
                feature_root=Path(tmp) / "features",
                gtsinger_root=Path(tmp) / "missing",
                models="acoustic",
                contentvec_model=None,
                contentvec_layer="12",
                python=sys.executable,
                device="cpu",
                force=False,
                smoke_only=True,
                synthetic=True,
                folds=2,
                bootstrap=10,
                max_rows_per_singer=2,
                max_phone_pairs_per_group=10,
                job_timeout_sec=30,
            )
            runner = Stage1Runner(args)
            ok = runner.run_job(
                "optional_failure",
                [sys.executable, "-c", "import sys; sys.exit(3)"],
                [Path(tmp) / "missing.out"],
                optional=True,
            )
            self.assertFalse(ok)
            job_dir = args.run_root / "jobs" / "optional_failure"
            status = (job_dir / "status.json").read_text(encoding="utf-8")
            summary = (job_dir / "summary.md").read_text(encoding="utf-8")
            self.assertIn('"status": "failed"', status)
            self.assertIn("Status: `failed`", summary)
            self.assertIn("Missing outputs:", summary)

    def test_timed_out_job_records_timeout_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                run_root=Path(tmp) / "run",
                report=Path(tmp) / "report.md",
                feature_root=Path(tmp) / "features",
                gtsinger_root=Path(tmp) / "missing",
                models="acoustic",
                contentvec_model=None,
                contentvec_layer="12",
                python=sys.executable,
                device="cpu",
                force=False,
                smoke_only=True,
                synthetic=True,
                folds=2,
                bootstrap=10,
                max_rows_per_singer=2,
                max_phone_pairs_per_group=10,
                job_timeout_sec=30,
            )
            runner = Stage1Runner(args)
            ok = runner.run_job(
                "optional_timeout",
                [sys.executable, "-c", "import time; time.sleep(30)"],
                [Path(tmp) / "missing.out"],
                optional=True,
                timeout_sec=1,
            )
            self.assertFalse(ok)
            job_dir = args.run_root / "jobs" / "optional_timeout"
            status = (job_dir / "status.json").read_text(encoding="utf-8")
            summary = (job_dir / "summary.md").read_text(encoding="utf-8")
            self.assertIn('"status": "timeout"', status)
            self.assertIn("Status: `timeout`", summary)
            self.assertIn("Timeout seconds: `1`", summary)

    def test_global_failure_returns_nonzero_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            report = Path(tmp) / "report.md"
            proc = subprocess.run(
                [
                    sys.executable,
                    "scripts/run/run_stage1.py",
                    "--run-root",
                    str(run_root),
                    "--report",
                    str(report),
                    "--python",
                    sys.executable,
                    "--models",
                    "acoustic",
                    "--gtsinger-root",
                    str(Path(tmp) / "missing_gtsinger"),
                ],
                cwd=ROOT,
                check=False,
            )

            self.assertNotEqual(proc.returncode, 0)
            self.assertTrue(report.exists())
            self.assertIn("Global failures", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
