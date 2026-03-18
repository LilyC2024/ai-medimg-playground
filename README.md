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

Repository demo run on March 18, 2026 after the limitation-improvement pass:
- ONNX export written to `onnx/model.onnx`.
- ONNX vs PyTorch validation on a 4-slice sample batch passed with `max_abs_diff=3.624e-05` and `max_prob_diff=6.378e-06`.
- The CLI produced `29` overlays plus `prediction_mask.npz` under `outputs/day7_infer_demo/`.
- Two back-to-back CPU CLI runs produced the same SHA-256 hash for `prediction_mask.npz`: `249012A93CCF75E841E57CF6975067443A1A907B939FBDF5B9CBD34B908124A2`.
- Against the refreshed Day 3 pseudo labels, the deployment path measured Dice `0.3331` and IoU `0.2314` on the sample series.

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

## Limitation Improvements

A dedicated limitation-reduction pass was run after the Day 7 deployment milestone. The goal was to improve the weakest parts of the pipeline without pretending that unavailable data or clinical labels could be invented.

Applied methods and observed improvements:
- Threshold brittleness in the Day 3 brain-ish mask was reduced with an adaptive threshold sweep in [classical_seg.py](C:/AI/ai-medimg-playground/src/baselines/classical_seg.py). The selector now evaluates 60 brain-window/head-threshold candidates and records the top-ranked settings. On the sample CT, brain-mask voxel count increased from `1,476` to `10,731`, and slice coverage increased from `12/29` to `25/29`.
- Single-series evaluation leakage was partially reduced with buffered intra-series holdout bands in [ct25d_dataset.py](C:/AI/ai-medimg-playground/src/data/ct25d_dataset.py) and [make_index.py](C:/AI/ai-medimg-playground/scripts/make_index.py). The index moved from `29` train slices only to `17 train / 4 val / 4 test / 4 buffer`, and Day 5 no longer needs `train-fallback` evaluation.
- Pseudo-label imbalance was addressed with class-balanced loss weighting in [unet_small.py](C:/AI/ai-medimg-playground/src/models/unet_small.py) and [train.py](C:/AI/ai-medimg-playground/scripts/train.py). On the refreshed holdout split, best eval Dice increased from `0.2456` to `0.3302`.
- Uncertainty calibration was improved with temperature scaling in [calibration.py](C:/AI/ai-medimg-playground/src/calibration.py), then propagated through [train.py](C:/AI/ai-medimg-playground/scripts/train.py), [infer.py](C:/AI/ai-medimg-playground/scripts/infer.py), and [inference_runtime.py](C:/AI/ai-medimg-playground/deploy/inference_runtime.py). Expected calibration error on the holdout split dropped from `0.4526` to `0.1588`.
- Deployment consistency was refreshed by re-exporting ONNX from the improved checkpoint and revalidating the CPU CLI. Day 7 deployment Dice against the refreshed pseudo labels increased from `0.2745` to `0.3331`, and repeated CLI runs remained deterministic with matching SHA-256 hashes.

Comparison caveat:
- The Day 3 pseudo labels are now broader and more anatomically plausible than before, so some before/after segmentation metrics are not perfectly apples-to-apples. Where possible, the comparison above emphasizes methodology improvements, holdout evaluation quality, calibration, and the refreshed deployment run.

The fuller narrative for this pass lives in [docs/limitation_improvements.md](C:/AI/ai-medimg-playground/docs/limitation_improvements.md) and [limitation_improvements.md](C:/AI/ai-medimg-playground/text%20summary/limitation_improvements.md).

## Known Limitations

- No manual ground-truth labels are included; Day 3 outputs are pseudo labels.
- Classical thresholds are now partially mitigated by adaptive brain-mask candidate selection, but severe artifacts (beam hardening, motion, metal) can still break the heuristics.
- Thick-slice scans can reduce 3D continuity and segmentation stability.
- The repository still contains one CT series; Day 4 now uses buffered intra-series val/test holdouts, which is more honest than pure train fallback but still not patient-level generalization.
- Day 5 supervision still uses Day 3 pseudo labels, so the reported Dice/IoU values validate the learning pipeline, not real clinical accuracy. The pseudo labels are stronger after the limitation-improvement pass, but they are still not manual ground truth.
- Day 6/7 uncertainty is now temperature-scaled on buffered holdout pseudo labels, which improved ECE, but it is still not calibrated clinical failure probability.
- Day 7 deployment is still CPU-only and validated on one sample series; broader operational testing still needs more data and real annotations.

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
