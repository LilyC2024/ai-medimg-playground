# AI Medical Imaging Playground

Reference pipeline for CT head volumes across three stages:
- Day 1: robust DICOM ingestion and HU validation
- Day 2: preprocessing (resampling, ROI crop, normalization, cache)
- Day 3: classical segmentation baseline (bone + brain-ish pseudo labels)

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Optional path overrides:

```powershell
$env:DICOM_SERIES_DIR="C:\path\to\dicom_series"
$env:DICOM_OUTPUT_DIR="C:\path\to\outputs"
$env:DICOM_PROCESSED_DIR="C:\path\to\data_processed"
```

## Main Commands

Day 1 (inspection and diagnostics):

```powershell
.\.venv\Scripts\python.exe scripts\inspect_series.py
```

Day 2 (preprocessing and cache generation):

```powershell
.\.venv\Scripts\python.exe scripts\preprocess_series.py
```

Day 3 (classical masks, pseudo labels, overlay review):

```powershell
.\.venv\Scripts\python.exe scripts\classical_baseline.py
```

Run all tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

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

## Pipeline Summary

```text
Raw DICOM series
  -> slice ordering + decode + HU conversion
  -> preprocessing (spacing, head ROI crop, HU clip/normalize)
  -> processed cache volume
  -> classical segmentation (bone, brain-ish masks)
  -> postprocessing (morphology, connected components, hole filling)
  -> pseudo labels + visual overlays + non-GT quality metrics
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

## Known Limitations

- No manual ground-truth labels are included; Day 3 outputs are pseudo labels.
- Classical thresholds may fail on severe artifacts (beam hardening, motion, metal).
- Thick-slice scans can reduce 3D continuity and segmentation stability.

## Repository Layout

```text
scripts/
  inspect_series.py
  preprocess_series.py
  classical_baseline.py
src/
  dicom_loader.py
  preprocessing.py
  baselines/classical_seg.py
  visualization.py
  config.py
tests/
  test_loader.py
  test_preprocessing.py
  test_classical_seg.py
```
