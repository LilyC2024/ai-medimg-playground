from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.unet_small import compute_segmentation_metrics  # noqa: E402


class TestUNetMetrics(unittest.TestCase):
    def test_metrics_are_one_for_perfect_prediction(self) -> None:
        target = torch.tensor(
            [
                [
                    [0, 1],
                    [2, 1],
                ],
            ],
            dtype=torch.int64,
        )
        pred = target.clone()
        metrics = compute_segmentation_metrics(pred, target, num_classes=3)
        self.assertAlmostEqual(float(metrics["dice"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["iou"]), 1.0, places=6)

    def test_metrics_match_expected_toy_values(self) -> None:
        target = torch.tensor(
            [
                [
                    [0, 1],
                    [1, 0],
                ],
            ],
            dtype=torch.int64,
        )
        pred = torch.tensor(
            [
                [
                    [0, 1],
                    [0, 0],
                ],
            ],
            dtype=torch.int64,
        )
        metrics = compute_segmentation_metrics(pred, target, num_classes=2)
        self.assertAlmostEqual(float(metrics["dice"]), 0.666667, places=5)
        self.assertAlmostEqual(float(metrics["iou"]), 0.500001, places=5)


if __name__ == "__main__":
    unittest.main()
