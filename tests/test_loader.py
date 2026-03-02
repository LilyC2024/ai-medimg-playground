from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dicom_loader import load_dicom_series  # noqa: E402


def _series_path() -> Path:
    configured = os.getenv("DICOM_SERIES_DIR_TEST") or os.getenv("DICOM_SERIES_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (REPO_ROOT / "data" / "dicom_series_01").resolve()


@unittest.skipUnless(_series_path().exists(), "DICOM series not found for loader tests.")
class TestDicomLoader(unittest.TestCase):
    def test_loader_returns_expected_shape(self) -> None:
        volume = load_dicom_series(_series_path())
        self.assertEqual(volume.volume_hu.shape, (34, 512, 512))
        self.assertEqual(volume.volume_hu.dtype, np.float32)

    def test_z_positions_are_monotonic(self) -> None:
        volume = load_dicom_series(_series_path())
        z_positions = np.asarray(volume.metadata.z_positions, dtype=np.float32)
        self.assertEqual(z_positions.size, 34)

        diffs = np.diff(z_positions)
        self.assertTrue(np.all(diffs >= 0), msg=f"Found non-monotonic z ordering: {diffs.tolist()}")
        self.assertTrue(np.any(diffs > 0), msg="All z positions are identical; expected ordered slices.")


if __name__ == "__main__":
    unittest.main()
