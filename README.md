# ai-medimg-playground

Minimal playground for medical imaging experiments.

## Current structure

- `src/` for Python source code
- `notebooks/` for exploratory notebooks
- `docs/` for notes and documentation
- `data/` for local datasets (ignored by Git)
- `models/` for trained weights/artifacts (ignored by Git)
- `outputs/` for generated results (ignored by Git)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Git policy

This repository intentionally does not track local datasets, model files, runtime outputs, or virtual environments.
