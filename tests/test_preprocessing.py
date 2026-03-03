from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import PreprocessConfig  # noqa: E402
from preprocessing import (  # noqa: E402
    clip_and_normalize_hu,
    create_head_mask,
    expand_bbox_with_margin,
    bbox_from_mask,
    choose_target_spacing,
    load_npz_volume,
    save_npz_volume,
)


class TestPreprocessing(unittest.TestCase):
    def test_choose_target_spacing_keeps_coarse_z(self) -> None:
        config = PreprocessConfig(
            target_spacing_xy_mm=1.0,
            target_spacing_z_mm=1.0,
            keep_z_if_coarse=True,
            coarse_z_threshold_mm=3.0,
        )
        target = choose_target_spacing((5.0, 0.8, 0.8), config)
        self.assertEqual(target, (5.0, 1.0, 1.0))

    def test_head_mask_bbox_and_margin(self) -> None:
        volume = np.full((8, 12, 12), -1000.0, dtype=np.float32)
        volume[:, 3:9, 2:10] = 50.0

        mask = create_head_mask(volume, threshold_hu=-350.0, opening_iterations=0, closing_iterations=0)
        bbox = bbox_from_mask(mask)
        expanded = expand_bbox_with_margin(
            bbox=bbox,
            margin_mm_zyx=(1.0, 1.0, 1.0),
            spacing_zyx=(1.0, 1.0, 1.0),
            volume_shape_zyx=tuple(int(v) for v in volume.shape),
        )

        self.assertEqual((bbox.z_min, bbox.y_min, bbox.x_min), (0, 3, 2))
        self.assertEqual((bbox.z_max, bbox.y_max, bbox.x_max), (8, 9, 10))
        self.assertEqual((expanded.z_min, expanded.y_min, expanded.x_min), (0, 2, 1))
        self.assertEqual((expanded.z_max, expanded.y_max, expanded.x_max), (8, 10, 11))

    def test_clip_and_normalize_range(self) -> None:
        values = np.array([-1200.0, -1000.0, 0.0, 1000.0, 1800.0], dtype=np.float32)
        normalized = clip_and_normalize_hu(
            values,
            clip_min_hu=-1000.0,
            clip_max_hu=1000.0,
            normalize_min=0.0,
            normalize_max=1.0,
        )
        self.assertTrue(np.all(normalized >= 0.0))
        self.assertTrue(np.all(normalized <= 1.0))
        self.assertAlmostEqual(float(normalized[0]), 0.0, places=6)
        self.assertAlmostEqual(float(normalized[-1]), 1.0, places=6)
        self.assertAlmostEqual(float(normalized[2]), 0.5, places=6)

    def test_npz_roundtrip(self) -> None:
        volume = np.random.RandomState(7).randn(4, 5, 6).astype(np.float32)
        spacing = (2.5, 1.0, 1.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "volume.npz"
            save_npz_volume(volume, spacing, path)
            loaded_volume, loaded_spacing = load_npz_volume(path)

        self.assertEqual(loaded_volume.shape, volume.shape)
        self.assertTrue(np.allclose(loaded_volume, volume))
        self.assertEqual(loaded_spacing, spacing)


if __name__ == "__main__":
    unittest.main()
