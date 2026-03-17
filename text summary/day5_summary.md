# Day 5 Essay: Lightweight Deep Learning Segmentation on CPU

Day 5 moved the project from classical pseudo-label generation into a true learning pipeline. The goal was not to chase final accuracy on one small study, but to prove that the repository can now train, evaluate, checkpoint, and run inference with a lightweight 2.5D model that remains practical on CPU. The model uses the Day 4 stack format, so each training example contains `(z-1, z, z+1)` as input and the center pseudo-label slice as the target.

A compact U-Net was added with reduced channel counts so the forward and backward passes remain tractable without GPU dependence. The loss combines Cross-Entropy with Dice to balance class discrimination and overlap quality, while the metric helpers compute Dice and IoU in a way that can be unit-tested independently. Training resizes the Day 2 ROI crop to `256 x 256`, logs loss and segmentation metrics per epoch, and saves the best checkpoint to `saved_models/best.pt`.

## What each file contributes

- [unet_small.py](C:/AI/ai-medimg-playground/src/models/unet_small.py): Day 5 model and metric core.
  - Compact 2D U-Net for 2.5D inputs.
  - Combined Dice + Cross-Entropy loss.
  - Multi-class Dice and IoU helpers.
- [train.py](C:/AI/ai-medimg-playground/scripts/train.py): CPU training entry point.
  - Loads Day 4 dataset/index.
  - Applies reproducible seeding.
  - Resizes ROI slices to `256 x 256`.
  - Logs curves and saves `saved_models/best.pt`.
- [infer.py](C:/AI/ai-medimg-playground/scripts/infer.py): Full-series inference entry point.
  - Loads checkpoint.
  - Predicts every processed slice.
  - Writes Day 5 overlays and inference metrics.
- [visualization.py](C:/AI/ai-medimg-playground/src/visualization.py): Adds training-curve export and Day 5 overlay comparison figures.
- [test_unet_metrics.py](C:/AI/ai-medimg-playground/tests/test_unet_metrics.py): Confirms Dice and IoU calculations on toy masks.
- [day5_unet_cpu.md](C:/AI/ai-medimg-playground/docs/day5_unet_cpu.md): Focused Day 5 runbook and results.
- [README.md](C:/AI/ai-medimg-playground/README.md): Updated to include Day 5 commands, outputs, and limitations.

## Current repo results

The Day 5 training run used seed `13`, CPU execution, batch size `4`, and `6` epochs. It produced:

- `saved_models/best.pt`
- `outputs/day5_curves.png`
- `outputs/day5_overlays/` with `29` overlay PNGs
- `outputs/day5_train_report.json`
- `outputs/day5_infer_report.json`

Measured outcomes from this run:

- best eval Dice during training: `0.2456`
- full-series inference Dice vs pseudo labels: `0.2710`
- full-series inference IoU vs pseudo labels: `0.2239`
- runtime for training: about `40.37` seconds on CPU

A reproducibility spot-check was also run with two separate seeded 2-epoch trainings, and both produced identical history values. This confirms that the seeded CPU training path is stable for this repository state.

## Qualitative comparison vs classical baseline

The model reproduces the high-contrast bone regions far more easily than the softer brain-ish mask. In practical terms, the Day 5 overlays show that the learned model can follow the skull contour reasonably well, but it under-recovers class `1` and essentially fails to model the sparse overlap class. This is not surprising because the training targets are still the Day 3 pseudo labels, the dataset contains one patient/series only, and the overlap class is tiny compared with background and bone.

## Final summary

Day 5 delivered the first deep-learning stage of the playground: a compact CPU-trainable U-Net, tested metric helpers, reproducible seeded training, checkpoint export, full-series inference, and overlay-based review. The repository now covers a complete path from raw DICOM through preprocessing, pseudo labels, data-module construction, and lightweight model training. The next meaningful improvement would be more studies and real annotations so the same pipeline can be evaluated on a true held-out split instead of train-fallback pseudo-label supervision.
