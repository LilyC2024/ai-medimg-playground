# Day 7 Deployment Guide

## Primary CLI

```powershell
./.venv/Scripts/python.exe deploy/cli_infer.py `
  --series-dir data/dicom_series_01 `
  --checkpoint saved_models/best.pt `
  --onnx-path onnx/model.onnx `
  --output-dir outputs/day7_infer_demo `
  --threads 1 `
  --batch-size 1 `
  --mask-format npz `
  --force-export
```

## Optional Variants

NIfTI mask only:

```powershell
./.venv/Scripts/python.exe deploy/cli_infer.py --mask-format nii.gz --output-formats mask
```

FastAPI service:

```powershell
python -m uvicorn deploy.app:app --host 0.0.0.0 --port 8000
```

Docker:

```powershell
docker build -t ai-medimg-day7 .
docker run --rm -p 8000:8000 ai-medimg-day7
```

## Input Contract

- One folder containing a single axial CT DICOM series for the CLI.
- One ZIP file containing one axial CT DICOM series for `/predict`.
- Checkpoint and ONNX must correspond to the same model configuration.

## Output Contract

- `prediction_mask.npz` contains `predicted_labels` with shape `(Z, Y, X)` and `spacing_zyx`.
- `prediction_mask.nii.gz` is available when `--mask-format nii.gz` is selected.
- `overlays/slice_*.png` contains per-slice deployment review figures.
- `day7_infer_report.json` contains runtime settings, uncertainty summary, class voxel counts, and ONNX validation values.
- `day7_preprocess_report.json` contains raw-to-processed shape/spacing and crop-box metadata.
