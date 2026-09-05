from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from legacy.run_targeted_repair_then_stage1 import final_stage1_status, write_json


class RepairWrapperTest(unittest.TestCase):
    def test_final_status_preserves_stage1_optional_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "stage1"
            write_json({"status": "complete_with_failures"}, run_root / "status.json")

            self.assertEqual(final_stage1_status(0, run_root), "complete_with_failures")

    def test_final_status_marks_nonzero_stage1_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(final_stage1_status(2, Path(tmp) / "stage1"), "stage1_failed")


if __name__ == "__main__":
    unittest.main()
