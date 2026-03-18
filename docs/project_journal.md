# Project Journal

This journal consolidates the older day-by-day summary files into one place so the repository has a single documentation tree under `docs/`.

## Day 1

Day 1 established the ingestion baseline: configuration-driven paths, robust DICOM loading, HU conversion, ordering checks, and quick visual QA. The practical outcome was confidence that the sample CT series could be loaded reproducibly and interpreted in clinically meaningful intensity units before any preprocessing or modeling work began.

Key additions:
- `src/dicom_loader.py`
- `src/config.py`
- `src/visualization.py`
- `scripts/inspect_series.py`
- `tests/test_loader.py`

## Day 2

Day 2 turned the raw HU volume into a reusable modeling input through spacing control, head-focused cropping, clipping, normalization, and cached serialization. The pipeline also added before/after QA and timing comparisons so preprocessing became both inspectable and efficient to rerun.

Key additions:
- `src/preprocessing.py`
- `scripts/preprocess_series.py`
- `tests/test_preprocessing.py`

## Day 3

Day 3 introduced the classical segmentation bootstrap path. Bone and brain-ish masks were generated from CT-specific heuristics, cleaned with morphology and connected-component rules, and exported as pseudo labels for later supervised learning. This stage created the first working segmentation targets in the absence of manual ground truth.

Key additions:
- `src/baselines/classical_seg.py`
- `scripts/classical_baseline.py`
- `tests/test_classical_seg.py`

## Day 4

Day 4 converted the processed volume and pseudo labels into a real 2.5D learning interface. The repository gained slice indexing, stack construction for `(z-1, z, z+1)`, deterministic augmentations, and a PyTorch dataset/dataloader path. In the improved state of the repo, the split builder now also creates buffered intra-series holdout bands when only one study is available.

Current Day 4 snapshot:
- total indexed slices: `29`
- split layout: `17 train / 4 val / 4 test / 4 buffer`
- spacing: `5.0 x 1.0 x 1.0 mm`

Key additions:
- `src/data/ct25d_dataset.py`
- `scripts/make_index.py`
- `tests/test_ct25d_dataset.py`

## Day 5

Day 5 added the lightweight 2.5D U-Net, CPU-friendly training loop, checkpointing, inference, and metric reporting. After the later limitation-improvement pass, this stage also includes class-balanced loss weighting and temperature-scaling calibration fitted on the buffered holdout split.

Current Day 5 snapshot:
- best eval Dice: `0.3302`
- full-series inference Dice vs refreshed pseudo labels: `0.1900`
- full-series inference IoU vs refreshed pseudo labels: `0.1101`
- holdout ECE after temperature scaling: `0.1588`

Key additions:
- `src/models/unet_small.py`
- `src/calibration.py`
- `scripts/train.py`
- `scripts/infer.py`
- `tests/test_unet_metrics.py`

## Day 6

Day 6 shifted the project toward a more reviewable and robust workflow. It added postprocessing, uncertainty estimation, stricter metadata checks, a report generator, and a model card. The system became better at surfacing where outputs may need manual attention instead of only reporting aggregate segmentation scores.

Key additions:
- `src/robustness.py`
- `scripts/report.py`
- `scripts/check.py`
- `docs/model_card.md`

## Day 7

Day 7 packaged the project for real use beyond notebooks. The repo gained ONNX export, deterministic CPU inference with ONNX Runtime, a raw-DICOM CLI, an optional FastAPI wrapper, and a Docker entrypoint. The deployment path now stays aligned with the latest calibrated checkpoint.

Current Day 7 snapshot:
- ONNX validation passed within tolerance
- deployment Dice vs refreshed pseudo labels: `0.3331`
- repeated CPU CLI runs produced the same output hash
- FastAPI `/predict` test returned `200` on the sample ZIP input

Key additions:
- `deploy/cli_infer.py`
- `deploy/inference_runtime.py`
- `deploy/app.py`
- `Dockerfile`

## Limitation Improvement Pass

After the Day 7 milestone, the project went through a focused improvement pass aimed at reducing the most tractable limitations without overstating what the data could support.

Main improvements:
- adaptive Day 3 brain-mask selection instead of one fixed threshold configuration
- buffered intra-series holdout creation for more honest validation on the single available scan
- class-balanced loss weighting for Day 5 training
- temperature scaling for improved uncertainty calibration
- refreshed ONNX export and deployment validation against the updated checkpoint

Measured changes:
- brain-mask voxels: `1,476 -> 10,731`
- non-empty brain slices: `12 -> 25`
- best eval Dice: `0.2456 -> 0.3302`
- deployment Dice vs pseudo labels: `0.2745 -> 0.3331`
- calibration ECE: `0.4526 -> 0.1588`

## Current Takeaway

The repository now reads more cleanly as one coherent medical-imaging prototype: ingestion, preprocessing, pseudo-label generation, 2.5D training, robustness/reporting, deployment, and a measured improvement pass. It is still a small-scope project anchored to one CT series and pseudo-label supervision, but it now documents that scope more clearly and organizes the supporting material in one place.
