# Day 1-3: DICOM Ingestion, Preprocessing, and Classical Segmentation

This repository now contains:
- Day 1 workflow: load CT DICOM, convert to HU, validate ordering, export diagnostics.
- Day 2 workflow: spacing resample, head ROI crop, HU clip/normalize, cached save/reload timing.
- Day 3 workflow: classical mask baseline (bone + brain-ish), pseudo-label generation, and overlay-based review.

## Project Structure

```text
ai-medimg-playground/
|- data/                    # Local DICOM data (git-ignored except .gitkeep)
|- data_processed/          # Cached preprocessed volume output (.nii.gz/.npz)
|- outputs/                 # Generated images/metadata (git-ignored except .gitkeep)
|  `- overlays/             # Day 3 per-slice overlay review PNGs
|- scripts/inspect_series.py
|- scripts/preprocess_series.py
|- scripts/classical_baseline.py
|- src/baselines/classical_seg.py
|- src/config.py
|- src/dicom_loader.py
|- src/preprocessing.py
|- src/visualization.py
|- tests/test_classical_seg.py
|- tests/test_loader.py
|- tests/test_preprocessing.py
|- requirements.txt
`- pyproject.toml
```

## Environment Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Configure Data Path (No Hard-Coded Paths)

By default, the loader expects `data/dicom_series_01/`.
You can override with environment variables or CLI flags:

```powershell
$env:DICOM_SERIES_DIR="C:\path\to\your\series"
$env:DICOM_OUTPUT_DIR="C:\path\to\your\outputs"
```

## Day 1: Inspection + Diagnostics

```powershell
.\.venv\Scripts\python.exe scripts\inspect_series.py
```

This writes:
- `outputs/day1_metadata.json`
- `outputs/day1_hu_hist.png`
- `outputs/day1_montage.png`

To open an interactive axial scroll viewer with brain/bone windowing:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_series.py --show
```

## Run Unit Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Day 2: Preprocessing Pipeline

Run full preprocessing:

```powershell
.\.venv\Scripts\python.exe scripts\preprocess_series.py
```

Default artifacts:
- `data_processed/volume.nii.gz` (or `volume.npz` when `--save-format npz`)
- `outputs/day2_before_after.png`
- `outputs/day2_preprocess_report.json`

### Text Pipeline Diagram

```text
DICOM series (unordered files)
  -> sort by z/instance
  -> decode pixels + HU conversion (slope/intercept)
  -> volume_hu[z,y,x], spacing_zyx
  -> resample y/x to target spacing (default 1.0 mm)
  -> keep z as-is when coarse (default threshold: >= 3.0 mm)
  -> threshold-based head mask (air/head separation)
  -> largest connected component
  -> bounding box + physical margin
  -> crop ROI
  -> HU clipping (default [-1000, 1000])
  -> normalization (default [0, 1])
  -> save cache (.nii.gz or .npz)
  -> timed reload for speed comparison
  -> save QA figure with crop-box overlay
```

### Tunable Parameters (CLI)

The following are common knobs for future tuning:
- `--xy-spacing-mm` (default `1.0`)
- `--target-z-mm` + `--keep-z-if-coarse` + `--coarse-z-threshold-mm`
- `--head-threshold-hu`
- `--mask-opening-iters`, `--mask-closing-iters`
- `--crop-margin-mm z,y,x` (default `2,10,10`)
- `--hu-clip-min`, `--hu-clip-max`
- `--normalize-min`, `--normalize-max`
- `--save-format {nii.gz,npz}`

Example using NumPy cache:

```powershell
.\.venv\Scripts\python.exe scripts\preprocess_series.py --save-format npz
```

## Day 3: Classical Segmentation Baseline

Run full Day 3 pipeline (raw DICOM -> preprocess -> pseudo labels -> overlays):

```powershell
.\.venv\Scripts\python.exe scripts\classical_baseline.py
```

Default Day 3 artifacts:
- `data_processed/pseudo_labels/pseudo_labels_3d.npz`
- `data_processed/pseudo_labels/slices/slice_*.npz` (per-slice masks)
- `outputs/overlays/slice_*.png` (overlay review UI)
- `outputs/day3_classical_report.json`

### Day 3 Mask Strategy

```text
preprocessed HU volume
  -> bone mask candidate by high HU threshold
  -> brain-ish candidate by brain-window normalization threshold
  -> morphology cleanup + small-component removal
  -> largest connected component plausibility filter
  -> (brain) optional hole filling
  -> per-slice + 3D pseudo-label export
  -> non-GT quality metrics + overlay review PNGs
```

### Day 3 Common Tunable Parameters

- Bone:
  - `--bone-threshold-hu`
  - `--bone-open-iters`, `--bone-close-iters`
  - `--bone-min-voxels`, `--bone-keep-largest`
- Brain-ish:
  - `--brain-window-center`, `--brain-window-width`
  - `--brain-norm-min`, `--brain-norm-max`
  - `--brain-head-threshold-hu`
  - `--brain-open-iters`, `--brain-close-iters`
  - `--brain-fill-holes`, `--brain-min-voxels`, `--brain-keep-largest`
- Overlay export:
  - `--overlay-max-slices`
  - `--overlay-min-representative`

### Failure Modes and Next Steps

- Partial-volume effects near skull boundaries can break thin structures.
- Beam-hardening streak artifacts may trigger false positive high-HU bone regions.
- Motion artifacts can fragment brain continuity across adjacent slices.
- Metal/post-op implants can dominate simple thresholding and topology assumptions.
- Thick-slice scans reduce 3D continuity reliability for morphology rules.

Planned follow-up:
- Adaptive per-scan thresholding and stronger intensity standardization.
- Atlas/shape priors to constrain plausible intracranial regions.
- Uncertainty heuristics and manual QC for flagged slices.
- Transition from pseudo labels to supervised refinement (e.g., lightweight U-Net).

## Slope/Intercept and Windowing Notes

- Raw CT pixels are stored as scanner values, not physical HU.
- HU conversion is slice-wise: `HU = pixel_value * RescaleSlope + RescaleIntercept`.
- Typical air appears near `-1000 HU`; dense bone is high positive HU.
- Brain soft tissue is best viewed with brain window (`L=40, W=80`).
- Bone detail is emphasized with bone window (`L=600, W=2000`).
- Histogram + montage outputs are quick checks that conversion and contrast settings are sensible.
