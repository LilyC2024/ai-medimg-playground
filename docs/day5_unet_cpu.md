# Day 5 Lightweight U-Net on CPU

Day 5 adds the compact 2.5D learning stage to the repository. The model stays CPU-friendly, consumes `(z-1, z, z+1)` stacks, and predicts a 4-class center-slice mask. In the current repo state, this stage also includes class-balanced loss weighting and post-training temperature scaling.

## What This Stage Owns

- `src/models/unet_small.py`
  - compact 2D U-Net
  - Dice + Cross-Entropy training loss
  - metric helpers for Dice and IoU
- `src/calibration.py`
  - temperature scaling
  - negative log-likelihood and ECE helpers
- `scripts/train.py`
  - seeded CPU training loop
  - class-weight estimation
  - calibration fitting on the buffered holdout split
- `scripts/infer.py`
  - full-series inference
  - temperature-aware probability generation
  - postprocessing and report export

## Current Repository Snapshot

The current training run used the improved Day 4 split and refreshed pseudo labels.

- device: `cpu`
- epochs: `8`
- train / val samples: `17 / 4`
- best eval Dice: `0.3302`
- best eval IoU: `0.2626`
- training time: about `29.37 s`

Current full-series inference against the refreshed pseudo labels:

- mean Dice: `0.1900`
- mean IoU: `0.1101`
- class 1 Dice: `0.0820`
- class 2 Dice: `0.3579`
- class 3 Dice: `0.1301`

Current calibration summary:

- temperature: `0.0743`
- ECE before scaling: `0.4526`
- ECE after scaling: `0.1588`

## Interpretation

This stage is still best understood as a pipeline-validation milestone rather than a clinical segmentation result. The model is now trained and evaluated in a more careful way than the original Day 5 version, but the supervision signal still comes from pseudo labels and the dataset still contains one study.

The practical success is that training, calibration, checkpointing, inference, and overlay export all run end-to-end on CPU in a reproducible way.

## Produced Artifacts

- `saved_models/best.pt`
- `outputs/day5_curves.png`
- `outputs/day5_calibration_report.json`
- `outputs/day5_train_report.json`
- `outputs/day5_infer_report.json`
- `outputs/day5_overlays/`

## Validation Status

- training loop: verified
- inference path: verified
- metric helpers: verified by `tests/test_unet_metrics.py`
- current unit-test suite: `15/15` passing
