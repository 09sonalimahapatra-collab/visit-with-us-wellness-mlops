"""Smoke test for the data-registration component."""

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_registration import validate_dataset  # noqa: E402


class DataRegistrationTest(unittest.TestCase):
    def test_tourism_dataset_has_expected_schema(self):
        summary = validate_dataset(PROJECT_ROOT / "data" / "tourism.csv")
        self.assertEqual(summary["validation_status"], "passed")
        self.assertGreater(summary["rows"], 0)
        self.assertIn("1", summary["target_distribution"])


if __name__ == "__main__":
    unittest.main()
