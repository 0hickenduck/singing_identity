from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.data_prep.repair_gtsinger_missing_wavs import load_existing_queue, write_queue


class RepairQueueTest(unittest.TestCase):
    def test_existing_queue_loads_unique_sorted_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "gtsinger"
            queue = Path(tmp) / "missing_queue.json"
            write_queue(queue, root, "test", ["b.wav", "a.wav", "b.wav", ""])

            self.assertEqual(load_existing_queue(queue), ["a.wav", "b.wav"])


if __name__ == "__main__":
    unittest.main()
