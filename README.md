# Day 1: DICOM Ingestion and Validation

This repository contains the Day 1 workflow for loading a CT DICOM series, converting to HU, validating ordering, and exporting diagnostics.

## Project Structure

```text
ai-medimg-playground/
|- data/                    # Local DICOM data (git-ignored except .gitkeep)
|- outputs/                 # Generated images/metadata (git-ignored except .gitkeep)
|- scripts/inspect_series.py
|- src/config.py
|- src/dicom_loader.py
|- src/visualization.py
|- tests/test_loader.py
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

## Run Inspection + Export Diagnostics

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

## Slope/Intercept and Windowing (5-8 line explainer)

- Raw CT pixels are stored as scanner values, not physical HU.
- HU conversion is slice-wise: `HU = pixel_value * RescaleSlope + RescaleIntercept`.
- Typical air appears near `-1000 HU`; dense bone is high positive HU.
- Brain soft tissue is best viewed with brain window (`L=40, W=80`).
- Bone detail is emphasized with bone window (`L=600, W=2000`).
- Histogram + montage outputs are quick checks that conversion and contrast settings are sensible.
