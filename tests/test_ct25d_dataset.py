from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.ct25d_dataset import (  # noqa: E402
    CT25DDataset,
    Compose25D,
    RandomFlip25D,
    RandomIntensityJitter25D,
    RandomRotate25D,
    assign_group_splits,
    build_25d_stack,
)
from preprocessing import save_npz_volume  # noqa: E402


class TestCT25DDataset(unittest.TestCase):
    def test_build_25d_stack_handles_edges(self) -> None:
        volume = np.stack(
            [np.full((3, 3), fill_value=index, dtype=np.float32) for index in range(4)],
            axis=0,
        )

        start_stack = build_25d_stack(volume, center_index=0)
        end_stack = build_25d_stack(volume, center_index=3)

        self.assertEqual(start_stack.shape, (3, 3, 3))
        self.assertTrue(np.all(start_stack[0] == 0.0))
        self.assertTrue(np.all(start_stack[1] == 0.0))
        self.assertTrue(np.all(start_stack[2] == 1.0))
        self.assertTrue(np.all(end_stack[0] == 2.0))
        self.assertTrue(np.all(end_stack[1] == 3.0))
        self.assertTrue(np.all(end_stack[2] == 3.0))

    def test_assign_group_splits_keeps_each_group_in_one_split(self) -> None:
        assignments = assign_group_splits(
            group_ids=["a", "b", "c", "d", "e", "f"],
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            seed=21,
        )

        self.assertEqual(set(assignments.keys()), {"a", "b", "c", "d", "e", "f"})
        self.assertTrue(set(assignments.values()).issubset({"train", "val", "test"}))
        self.assertGreaterEqual(sum(split == "train" for split in assignments.values()), 1)
        self.assertGreaterEqual(sum(split == "val" for split in assignments.values()), 1)
        self.assertGreaterEqual(sum(split == "test" for split in assignments.values()), 1)

    def test_dataset_transforms_are_deterministic_when_seeded(self) -> None:
        volume = np.linspace(0.0, 1.0, num=4 * 6 * 6, dtype=np.float32).reshape(4, 6, 6)
        labels = np.zeros((4, 6, 6), dtype=np.uint8)
        labels[1, 2:4, 2:4] = 3

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            volume_path = tmp_path / "volume.npz"
            label_path = tmp_path / "pseudo_labels_3d.npz"
            index_path = tmp_path / "index.csv"

            save_npz_volume(volume, spacing_zyx=(1.0, 1.0, 1.0), output_path=volume_path)
            np.savez_compressed(label_path, pseudo_labels=labels)

            pd.DataFrame(
                [
                    {
                        "split": "train",
                        "patient_id": "p1",
                        "study_instance_uid": "study1",
                        "series_instance_uid": "series1",
                        "split_group_id": "p1|series1",
                        "series_dir": str(tmp_path),
                        "volume_path": str(volume_path),
                        "label_volume_path": str(label_path),
                        "slice_mask_path": None,
                        "slice_index": 1,
                        "stack_prev_index": 0,
                        "stack_center_index": 1,
                        "stack_next_index": 2,
                        "depth": 4,
                        "height": 6,
                        "width": 6,
                        "spacing_z_mm": 1.0,
                        "spacing_y_mm": 1.0,
                        "spacing_x_mm": 1.0,
                    },
                ],
            ).to_csv(index_path, index=False)

            transforms = Compose25D(
                [
                    RandomFlip25D(horizontal_prob=1.0, vertical_prob=1.0),
                    RandomRotate25D(max_degrees=5.0, probability=1.0),
                    RandomIntensityJitter25D(probability=1.0),
                ],
            )
            dataset_a = CT25DDataset(index_csv_path=index_path, split="train", transforms=transforms, seed=123)
            dataset_b = CT25DDataset(index_csv_path=index_path, split="train", transforms=transforms, seed=123)
            dataset_c = CT25DDataset(index_csv_path=index_path, split="train", transforms=transforms, seed=999)

            sample_a = dataset_a[0]
            sample_b = dataset_b[0]
            sample_c = dataset_c[0]

        self.assertTrue(np.allclose(sample_a["image"].numpy(), sample_b["image"].numpy()))
        self.assertTrue(np.array_equal(sample_a["mask"].numpy(), sample_b["mask"].numpy()))
        self.assertFalse(np.allclose(sample_a["image"].numpy(), sample_c["image"].numpy()))
        self.assertEqual(int(sample_a["mask"].max().item()), 3)
        self.assertEqual(int(sample_b["mask"].max().item()), 3)


if __name__ == "__main__":
    unittest.main()
