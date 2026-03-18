from __future__ import annotations

import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deploy.inference_runtime import run_deployment_inference, unzip_series_bytes  # noqa: E402
from config import PreprocessConfig  # noqa: E402

app = FastAPI(title="AI MedImg Playground Day 7 API", version="0.7.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    dicom_zip: UploadFile = File(...),
    threads: int = Form(1),
    batch_size: int = Form(1),
) -> dict[str, object]:
    payload = await dicom_zip.read()
    extracted_dir = unzip_series_bytes(payload)
    output_dir = Path(tempfile.mkdtemp(prefix="day7_predict_"))
    try:
        result = run_deployment_inference(
            series_dir=extracted_dir,
            checkpoint_path=REPO_ROOT / "saved_models" / "best.pt",
            onnx_path=REPO_ROOT / "onnx" / "model.onnx",
            output_dir=output_dir,
            preprocess_config=replace(PreprocessConfig(), mask_opening_iterations=1, mask_closing_iterations=2),
            batch_size=batch_size,
            num_threads=threads,
            export_onnx=not (REPO_ROOT / "onnx" / "model.onnx").exists(),
            validate_onnx=False,
            mask_output_format="npz",
            save_overlays=False,
            enable_postprocess=True,
        )
        return {
            "mask_volume_path": str(result.output_paths.mask_volume_path),
            "report_path": str(result.output_paths.report_path),
            "slice_count": result.report["slice_count"],
            "uncertainty_summary": result.report["uncertainty_summary"],
        }
    finally:
        shutil.rmtree(extracted_dir.parent, ignore_errors=True)
