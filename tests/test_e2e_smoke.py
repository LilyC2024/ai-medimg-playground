from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def _repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts).resolve()


def _has_smoke_artifacts() -> bool:
    required = [
        _repo_path("data", "dicom_series_01"),
        _repo_path("saved_models", "best.pt"),
        _repo_path("onnx", "model.onnx"),
    ]
    return all(path.exists() for path in required)


@unittest.skipUnless(_has_smoke_artifacts(), "Smoke-test artifacts are not available.")
class TestDeploymentSmoke(unittest.TestCase):
    def test_cli_inference_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory(prefix="day8_smoke_") as tmp_dir:
            output_dir = Path(tmp_dir) / "infer_out"
            command = [
                sys.executable,
                str(_repo_path("deploy", "cli_infer.py")),
                "--series-dir",
                str(_repo_path("data", "dicom_series_01")),
                "--checkpoint",
                str(_repo_path("saved_models", "best.pt")),
                "--onnx-path",
                str(_repo_path("onnx", "model.onnx")),
                "--processed-dir",
                str(_repo_path("data_processed")),
                "--output-dir",
                str(output_dir),
                "--threads",
                "1",
                "--batch-size",
                "1",
                "--mask-format",
                "npz",
                "--output-formats",
                "mask",
                "--skip-validation",
            ]

            env = os.environ.copy()
            env.setdefault("PYTHONHASHSEED", "0")
            result = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=f"CLI smoke test failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
            )

            mask_path = output_dir / "prediction_mask.npz"
            report_path = output_dir / "day7_infer_report.json"
            preprocess_report_path = output_dir / "day7_preprocess_report.json"

            self.assertTrue(mask_path.exists(), msg="Expected mask output was not created.")
            self.assertTrue(report_path.exists(), msg="Expected inference report was not created.")
            self.assertTrue(preprocess_report_path.exists(), msg="Expected preprocess report was not created.")

            with np.load(mask_path) as mask_data:
                predicted_labels = mask_data["predicted_labels"]
                spacing_zyx = mask_data["spacing_zyx"]

            self.assertEqual(predicted_labels.ndim, 3)
            self.assertEqual(tuple(spacing_zyx.shape), (3,))
            self.assertGreater(int(predicted_labels.shape[0]), 0)

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["batch_size"], 1)
            self.assertEqual(report["num_threads"], 1)
            self.assertEqual(report["mask_output_format"], "npz")
            self.assertEqual(report["overlay_count"], 0)
            self.assertIsNone(report["onnx_validation"])
            self.assertIn("reference_metrics_against_day3_pseudo_labels", report)


if __name__ == "__main__":
    unittest.main()
