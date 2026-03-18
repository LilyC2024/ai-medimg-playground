# AI Medical Imaging Playground

Reference pipeline for CT head volumes across seven stages:
- Day 1: robust DICOM ingestion and HU validation
- Day 2: preprocessing (resampling, ROI crop, normalization, cache)
- Day 3: classical segmentation baseline (bone + brain-ish pseudo labels)
- Day 4: 2.5D dataset builder, index CSV, and training-ready dataloaders
- Day 5: lightweight 2.5D U-Net training, inference, and overlay review
- Day 6: robustness checks, postprocessing, uncertainty proxy, and standardized reporting
- Day 7: deployable ONNX export, CPU inference CLI, optional FastAPI, and Docker packaging

## Quick Start

```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

Optional path overrides:

```powershell
$env:DICOM_SERIES_DIR="C:/path/to/dicom_series"
$env:DICOM_OUTPUT_DIR="C:/path/to/outputs"
$env:DICOM_PROCESSED_DIR="C:/path/to/data_processed"
```

## Main Commands

Day 1 (inspection and diagnostics):

```powershell
./.venv/Scripts/python.exe scripts/inspect_series.py
```

Day 2 (preprocessing and cache generation):

```powershell
./.venv/Scripts/python.exe scripts/preprocess_series.py
```

Day 3 (classical masks, pseudo labels, overlay review):

```powershell
./.venv/Scripts/python.exe scripts/classical_baseline.py
```

Day 4 (2.5D index, dataset module, and batch sanity check):

```powershell
./.venv/Scripts/python.exe scripts/make_index.py
```

Day 5 (lightweight U-Net training on CPU + full-series inference):

```powershell
./.venv/Scripts/python.exe scripts/train.py
./.venv/Scripts/python.exe scripts/infer.py
```

Day 6 (reporting and automated checks):

```powershell
./.venv/Scripts/python.exe scripts/report.py
./.venv/Scripts/python.exe scripts/check.py
```

Day 7 (deployable CPU inference from raw DICOM):

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

Optional Day 7 variations:

```powershell
./.venv/Scripts/python.exe deploy/cli_infer.py --mask-format nii.gz --output-formats mask
python -m uvicorn deploy.app:app --host 0.0.0.0 --port 8000
docker build -t ai-medimg-day7 .
docker run --rm -p 8000:8000 ai-medimg-day7
```

Run all tests:

```powershell
./.venv/Scripts/python.exe -m unittest discover -s tests -v
```

## Day 7 Deployment Contract

Input contract:
- `deploy/cli_infer.py` expects a folder containing one axial CT DICOM series.
- The checkpoint must match the exported ONNX graph (`saved_models/best.pt` -> `onnx/model.onnx`).
- The FastAPI `/predict` endpoint expects a ZIP file whose contents are the DICOM slices for one series.

Output contract:
- `prediction_mask.npz` stores `predicted_labels` as a `(Z, Y, X)` uint8 volume plus `spacing_zyx`.
- `prediction_mask.nii.gz` is available with `--mask-format nii.gz`.
- `overlays/` contains one PNG per slice with grayscale background, predicted mask overlay, and uncertainty heatmap.
- `day7_infer_report.json` records runtime settings, ONNX validation results, class voxel counts, and uncertainty summary.
- `day7_preprocess_report.json` records raw-vs-processed shape, spacing, crop box, and preprocessing parameters.

## Outputs

Day 1:
- `outputs/day1_metadata.json`
- `outputs/day1_hu_hist.png`
- `outputs/day1_montage.png`

Day 2:
- `data_processed/volume.nii.gz` (or `volume.npz`)
- `outputs/day2_before_after.png`
- `outputs/day2_preprocess_report.json`

Day 3:
- `data_processed/pseudo_labels/pseudo_labels_3d.npz`
- `data_processed/pseudo_labels/slices/slice_*.npz`
- `outputs/overlays/slice_*.png`
- `outputs/day3_classical_report.json`

Day 4:
- `src/data/ct25d_dataset.py`
- `scripts/make_index.py`
- `data_processed/index.csv`
- `outputs/day4_batch_viz.png`
- `outputs/day4_data_report.json`

Day 5:
- `src/models/unet_small.py`
- `scripts/train.py`
- `scripts/infer.py`
- `saved_models/best.pt`
- `outputs/day5_curves.png`
- `outputs/day5_overlays/`
- `outputs/day5_train_report.json`
- `outputs/day5_infer_report.json`

Day 6:
- `src/robustness.py`
- `scripts/report.py`
- `scripts/check.py`
- `docs/model_card.md`
- `outputs/report.md`
- `outputs/report_montage.png`

Day 7:
- `deploy/cli_infer.py`
- `deploy/inference_runtime.py`
- `deploy/app.py`
- `onnx/model.onnx`
- `outputs/day7_infer_demo/`
- `Dockerfile`

## Verified Day 7 Demo Results

Repository demo run on March 18, 2026:
- ONNX export written to `onnx/model.onnx`.
- ONNX vs PyTorch validation on a 4-slice sample batch passed with `max_abs_diff=1.907e-06` and `max_prob_diff=2.980e-07`.
- The CLI produced `29` overlays plus `prediction_mask.npz` under `outputs/day7_infer_demo/`.
- Two back-to-back CPU CLI runs produced the same SHA-256 hash for `prediction_mask.npz`: `4A49D9D24F96D1230DEEA5AC0F86CC3DC9A9E72C0C79BFC3DC4C42FC365D1BE3`.
- Against the Day 3 pseudo labels, the deployment path measured Dice `0.2745` and IoU `0.2255` on the sample series.

## Pipeline Summary

```text
Raw DICOM series
  -> slice ordering + decode + HU conversion
  -> preprocessing (spacing, head ROI crop, HU clip/normalize)
  -> processed cache volume
  -> classical segmentation (bone, brain-ish masks)
  -> postprocessing (morphology, connected components, hole filling)
  -> pseudo labels + visual overlays + non-GT quality metrics
  -> 2.5D stack builder ((z-1, z, z+1) predicts z)
  -> slice-level index CSV + group-safe split labels
  -> PyTorch Dataset/DataLoader for training-time ingestion
  -> lightweight 2.5D U-Net training on pseudo labels
  -> full-series inference + overlay comparison against classical baseline
  -> uncertainty proxy + postprocessing + standardized evaluation report
  -> ONNX export + CPU runtime CLI + optional API/container deployment
```

## Key Tuning Parameters

Preprocessing (`scripts/preprocess_series.py`):
- spacing: `--xy-spacing-mm`, `--target-z-mm`, `--keep-z-if-coarse`
- crop: `--head-threshold-hu`, `--mask-opening-iters`, `--mask-closing-iters`, `--crop-margin-mm`
- intensity: `--hu-clip-min`, `--hu-clip-max`, `--normalize-min`, `--normalize-max`
- cache: `--save-format {nii.gz,npz}`

Classical segmentation (`scripts/classical_baseline.py`):
- bone mask: `--bone-threshold-hu`, `--bone-open-iters`, `--bone-close-iters`, `--bone-min-voxels`
- brain-ish mask: `--brain-window-center`, `--brain-window-width`, `--brain-norm-min`, `--brain-norm-max`, `--brain-head-threshold-hu`
- cleanup: `--brain-fill-holes`, `--brain-min-voxels`, `--brain-keep-largest`, `--bone-keep-largest`
- overlay export: `--overlay-max-slices`, `--overlay-min-representative`

2.5D data module (`scripts/make_index.py`):
- splits: `--train-ratio`, `--val-ratio`, `--test-ratio`, `--seed`
- loader preview: `--batch-size`, `--num-workers`
- CPU-friendly augmentations: `--rotation-deg`, `--disable-intensity-jitter`

Lightweight DL training (`scripts/train.py`):
- optimization: `--epochs`, `--batch-size`, `--learning-rate`
- reproducibility: `--seed`, `--num-workers`, `--num-threads`
- model/input size: `--image-size`, `--base-channels`

Inference (`scripts/infer.py`):
- runtime: `--checkpoint`, `--batch-size`, `--device`
- paths: `--index-path`, `--processed-dir`, `--output-dir`
- robustness: `--uncertainty-method`, `--disable-postprocess`, `--brain-min-voxels`, `--bone-min-voxels`

Deployment (`deploy/cli_infer.py`):
- artifact control: `--onnx-path`, `--force-export`, `--skip-validation`
- runtime: `--threads`, `--batch-size`, `--mask-format`, `--output-formats`
- preprocessing: `--xy-spacing-mm`, `--target-z-mm`, `--head-threshold-hu`, `--crop-margin-mm`
- postprocessing: `--disable-postprocess`

## Known Limitations

- No manual ground-truth labels are included; Day 3 outputs are pseudo labels.
- Classical thresholds may fail on severe artifacts (beam hardening, motion, metal).
- Thick-slice scans can reduce 3D continuity and segmentation stability.
- The current repository contains one CT series, so leakage-safe Day 4 splitting falls back to `train` only until more patient/series groups are added.
- Day 5 supervision still uses Day 3 pseudo labels, so the reported Dice/IoU values validate the learning pipeline, not real clinical accuracy.
- Day 6 uncertainty scores are heuristic review aids, not calibrated probabilities of failure.
- Day 7 deployment is CPU-only and validated on one sample series; broader operational testing still needs more data and real annotations.

## Repository Layout

```text
scripts/
  inspect_series.py
  preprocess_series.py
  classical_baseline.py
  make_index.py
  train.py
  infer.py
  report.py
  check.py
deploy/
  inference_runtime.py
  cli_infer.py
  app.py
src/
  dicom_loader.py
  preprocessing.py
  baselines/classical_seg.py
  data/ct25d_dataset.py
  models/unet_small.py
  robustness.py
  visualization.py
  config.py
tests/
  test_loader.py
  test_preprocessing.py
  test_classical_seg.py
  test_ct25d_dataset.py
  test_unet_metrics.py
Dockerfile
```
