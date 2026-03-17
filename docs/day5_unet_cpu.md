# Day 5 Lightweight U-Net on CPU

Day 5 adds a compact 2.5D deep-learning stage on top of the Day 4 data module. The model is intentionally small enough to run on CPU: it consumes the three-slice stack `(z-1, z, z+1)`, predicts a 4-class center-slice mask, and trains on the Day 3 pseudo labels so the repository can validate a full learning loop even without manual annotations.

## What was added

- `src/models/unet_small.py`
  - Compact 2D U-Net with reduced channels for CPU-friendly training.
  - Combined Dice + Cross-Entropy loss.
  - Multi-class Dice and IoU metric helpers.
- `scripts/train.py`
  - Reproducible training loop with seeded `random`, `numpy`, and `torch`.
  - 256 x 256 resized ROI training on the Day 4 2.5D dataset.
  - Exports `saved_models/best.pt`, `outputs/day5_curves.png`, and `outputs/day5_train_report.json`.
- `scripts/infer.py`
  - Loads the saved checkpoint and runs inference across the full processed series.
  - Exports `outputs/day5_overlays/` and `outputs/day5_infer_report.json`.
- `tests/test_unet_metrics.py`
  - Verifies Dice and IoU on toy arrays.

## Run results from this repository

Training was run on CPU for 6 epochs with seed `13`, batch size `4`, and input size `256 x 256`.

- Training time: about `40.37 s`
- Best eval Dice: `0.2456`
- Final full-series inference Dice vs pseudo labels: `0.2710`
- Final full-series inference IoU vs pseudo labels: `0.2239`
- Processed slice coverage: `29` slices

Per-class inference Dice against pseudo labels:

- Class `1` (brain-ish): `0.0146`
- Class `2` (bone): `0.7984`
- Class `3` (overlap): near `0`

## Interpretation

The model learns the high-contrast bone class much more easily than the softer tissue and overlap classes. That is expected for this setup because:

- supervision is pseudo-label based, not manual GT
- the dataset contains one patient/series only
- evaluation falls back to the train split to avoid fake leakage-safe validation
- the overlap class is extremely sparse

So Day 5 should be read as a pipeline validation milestone, not a claim of segmentation quality. The important success is that training, checkpointing, metric computation, inference, and overlay export all work end-to-end on CPU.

## Produced artifacts

- `saved_models/best.pt`
- `outputs/day5_curves.png`
- `outputs/day5_overlays/`
- `outputs/day5_train_report.json`
- `outputs/day5_infer_report.json`

## Quality checklist status

- Training reproducibility: checked with two repeated seeded 2-epoch runs producing identical histories
- Inference end-to-end: passed on the full processed series (`29` cropped slices derived from the original `34` raw DICOM slices)
- Metrics correctness: verified by `tests/test_unet_metrics.py`
