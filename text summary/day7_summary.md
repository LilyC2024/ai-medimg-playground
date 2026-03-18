# Day 7 Essay: Deployment with ONNX, CPU Inference CLI, and Optional API Packaging

Day 7 turned the project from a research-style prototype into something that can be run like a real deployable tool. The focus was not on changing model architecture or squeezing out more training accuracy. Instead, the goal was to package the existing Day 5 and Day 6 work into a deterministic CPU inference path that starts from raw DICOM, exports the network to ONNX, runs with ONNX Runtime, writes predictable artifacts, and can optionally be exposed through FastAPI or a Docker container.

The most important shift is architectural: inference is no longer tied only to the training dataset/index flow. The new deployment path reads a DICOM folder directly, reuses the Day 2 preprocessing logic, builds 2.5D stacks, executes ONNX Runtime on CPU, applies the Day 6 postprocessing rules, and writes output masks plus qualitative overlays. That makes the repository easier to demonstrate, easier to hand off, and much closer to what an interviewer or collaborator would expect from a portfolio-ready medical imaging project.

## What each file contributes

- [cli_infer.py](C:/AI/ai-medimg-playground/deploy/cli_infer.py): Main Day 7 command-line entry point.
  - Accepts a raw DICOM folder.
  - Exports ONNX when needed.
  - Runs deterministic CPU inference with thread and output-format controls.
  - Writes deployment reports and optional pseudo-label comparison metrics.
- [inference_runtime.py](C:/AI/ai-medimg-playground/deploy/inference_runtime.py): Shared Day 7 deployment runtime.
  - Loads the PyTorch checkpoint.
  - Exports and validates the ONNX graph.
  - Reuses Day 2 preprocessing and Day 6 postprocessing.
  - Saves prediction volumes, overlays, and JSON reports.
- [app.py](C:/AI/ai-medimg-playground/deploy/app.py): Optional FastAPI wrapper.
  - Exposes `/health` and `/predict`.
  - Accepts zipped DICOM as input.
  - Reuses the same deployment runtime as the CLI.
- [visualization.py](C:/AI/ai-medimg-playground/src/visualization.py): Adds Day 7 overlay export with prediction and uncertainty views.
- [README.md](C:/AI/ai-medimg-playground/README.md): Updated with Day 7 usage docs, example commands, deployment outputs, and the input/output contract.
- [Dockerfile](C:/AI/ai-medimg-playground/Dockerfile): Optional container packaging for the FastAPI service.
- [requirements.txt](C:/AI/ai-medimg-playground/requirements.txt) and [pyproject.toml](C:/AI/ai-medimg-playground/pyproject.toml): Updated for ONNX export/runtime and FastAPI upload support.

## Verified results in this repository

The Day 7 demo was run on March 18, 2026 against the repository sample series in `data/dicom_series_01`.

Produced artifacts:
- `onnx/model.onnx`
- `outputs/day7_infer_demo/prediction_mask.npz`
- `outputs/day7_infer_demo/overlays/` with `29` PNGs
- `outputs/day7_infer_demo/day7_infer_report.json`
- `outputs/day7_infer_demo/day7_preprocess_report.json`
- `outputs/day7_infer_demo/day7_onnx_validation.json`

Measured deployment checks:
- ONNX validation sample batch shape: `(4, 3, 256, 256)`
- max absolute logit difference vs PyTorch: `1.9073486328125e-06`
- mean absolute logit difference vs PyTorch: `4.459558056169044e-08`
- max probability difference vs PyTorch: `2.980232238769531e-07`
- ONNX tolerance check: `pass`

Measured inference outputs:
- processed deployment volume shape: `(29, 207, 171)`
- processed spacing: `(5.0, 1.0, 1.0)` mm
- uncertainty mean over predicted foreground: `0.8799`
- pseudo-label Dice from the Day 7 deployment path: `0.2745`
- pseudo-label IoU from the Day 7 deployment path: `0.2255`

Determinism check:
- Two consecutive Day 7 CLI runs produced the same SHA-256 for `prediction_mask.npz`.
- Shared hash: `4A49D9D24F96D1230DEEA5AC0F86CC3DC9A9E72C0C79BFC3DC4C42FC365D1BE3`

Optional API validation:
- The FastAPI app imported successfully after installing `python-multipart`.
- A `TestClient` request to `/predict` with a zipped copy of the sample DICOM series returned HTTP `200` and `slice_count=29`.

## Why this matters

- ONNX export proves the model can leave the training framework and still behave numerically the same within a tight tolerance.
- A CLI that starts from raw DICOM is much easier to demonstrate than a notebook or a dataset-index-only script.
- Deterministic CPU inference is useful for QA, demos, and environments without GPU access.
- A small FastAPI wrapper and Dockerfile make the project look like a deployment-ready prototype instead of a one-off experiment.
- Clear input/output contracts reduce ambiguity when someone else tries to run the pipeline.

## Final summary

Day 7 completed the deployment phase by adding ONNX export, validated CPU inference through ONNX Runtime, a raw-DICOM CLI, optional FastAPI serving, and Docker packaging. Just as importantly, the repository now documents how to run this path, what files it expects, what it outputs, and how closely the ONNX result matches the original PyTorch model. The project is still limited by one-series data and pseudo-label supervision, but it now demonstrates the full arc from ingesting medical images to packaging a deployable inference tool.
