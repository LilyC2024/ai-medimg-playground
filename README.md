# AI Medical Imaging Playground 2026

A minimal, practical workspace for experimenting with medical imaging workflows in Python.

## What This Repository Is For

Use this project to:
- prototype image processing and analysis pipelines,
- test model ideas on local datasets,
- keep notes and exploratory notebooks organized.

## Project Layout

```text
ai-medimg-playground/
|- src/          # Python source code
|- notebooks/    # Jupyter notebooks and experiments
|- docs/         # Notes and documentation
|- data/         # Local datasets (git-ignored, keep only .gitkeep)
|- models/       # Trained weights/artifacts (git-ignored, keep only .gitkeep)
|- outputs/      # Generated outputs (git-ignored, keep only .gitkeep)
|- requirements.txt
`- README.md
```

## Quick Start

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Start working

- Add reusable code to `src/`
- Explore ideas in `notebooks/`
- Store temporary outputs in `outputs/`

## Dependencies

Core libraries are listed in [`requirements.txt`](./requirements.txt), including:
- image I/O and processing (`pydicom`, `SimpleITK`, `scikit-image`, `opencv-python`)
- numerical tooling (`numpy`, `scipy`, `pandas`)
- deep learning and inference (`torch`, `monai`, `onnxruntime`)
- lightweight serving (`fastapi`, `uvicorn`)

## Git and Data Policy

This repo intentionally does **not** track:
- local datasets,
- trained model files,
- generated outputs,
- virtual environments.

The `.gitignore` keeps only the placeholder `.gitkeep` files in `data/`, `models/`, and `outputs/`.

## Suggested Next Steps

- Add a first pipeline module in `src/` (for example `src/preprocess.py`)
- Create a starter notebook in `notebooks/`
- Document your experiment workflow in `docs/`
