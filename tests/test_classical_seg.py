from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines.classical_seg import (  # noqa: E402
    bone_mask,
    brain_mask,
    generate_classical_masks,
    summarize_mask_quality,
)
from config import SegmentationConfig  # noqa: E402


class TestClassicalSegmentation(unittest.TestCase):
    def _synthetic_volume(self) -> np.ndarray:
        volume = np.full((12, 40, 40), -1000.0, dtype=np.float32)

        # Brain-ish soft tissue block.
        volume[2:10, 11:29, 11:29] = 35.0

        # Surrounding high-HU skull-like ring.
        volume[2:10, 9:31, 9:31] = 850.0
        volume[2:10, 12:28, 12:28] = 35.0

        # Tiny noise islands that should be filtered.
        volume[0, 0, 0] = 1200.0
        volume[11, 39, 39] = 10.0
        return volume

    def test_bone_mask_filters_islands(self) -> None:
        volume = self._synthetic_volume()
        mask = bone_mask(
            volume_hu=volume,
            threshold_hu=300.0,
            opening_iterations=0,
            closing_iterations=0,
            min_component_voxels=32,
            keep_largest_component=True,
        )
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(int(mask[0, 0, 0]), 0)
        self.assertEqual(int(mask[4, 9, 9]), 1)

    def test_brain_mask_recovers_soft_tissue(self) -> None:
        volume = self._synthetic_volume()
        mask = brain_mask(
            volume_hu=volume,
            window_center=40.0,
            window_width=120.0,
            norm_min=0.05,
            norm_max=0.95,
            head_threshold_hu=-300.0,
            opening_iterations=0,
            closing_iterations=0,
            fill_holes=False,
            min_component_voxels=32,
            keep_largest_component=True,
        )
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(int(mask[5, 20, 20]), 1)
        self.assertEqual(int(mask[0, 0, 0]), 0)

    def test_generate_masks_and_stats(self) -> None:
        volume = self._synthetic_volume()
        result = generate_classical_masks(volume, SegmentationConfig())
        self.assertEqual(result.bone_mask_3d.shape, volume.shape)
        self.assertEqual(result.brain_mask_3d.shape, volume.shape)
        self.assertEqual(result.pseudo_labels_3d.shape, volume.shape)

        stats = summarize_mask_quality(result.brain_mask_3d, spacing_zyx=(1.0, 1.0, 1.0))
        self.assertGreater(int(stats["voxel_count"]), 0)
        self.assertGreaterEqual(int(stats["max_contiguous_run_slices"]), 1)
        self.assertGreaterEqual(float(stats["largest_component_ratio"]), 0.5)


if __name__ == "__main__":
    unittest.main()
