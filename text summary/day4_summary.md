# Day 4 Essay: 2.5D Dataset Builder and Training-Ready Data Module

Day 4 focused on turning the Day 1 to Day 3 artifacts into something a learning system can consume directly. The main design choice was 2.5D: for each target slice `z`, the model sees the neighboring context `(z-1, z, z+1)` and predicts the center mask. This keeps the memory and CPU cost close to 2D training while still recovering some through-plane information that would be lost with single-slice inputs.

The work starts from the Day 2 processed volume and Day 3 pseudo labels, so the new code inherits the existing spacing, crop, normalization, and mask conventions instead of inventing a separate format. A Day 4 index builder was added to write one CSV row per slice with patient ID, study/series IDs, slice index, edge-safe stack indices, and source paths. On top of that, a PyTorch dataset and dataloader interface now reads those rows back into 3-channel image stacks and center-slice masks, with deterministic CPU-friendly augmentation support for flips, small rotations, and optional intensity jitter.

## What each file contributes

- [ct25d_dataset.py](C:/AI/ai-medimg-playground/src/data/ct25d_dataset.py): Day 4 core data logic.
  - `clamp_stack_indices()` and `build_25d_stack()` guarantee correct `(z-1, z, z+1)` assembly with repeated edge slices.
  - `assign_group_splits()` assigns whole patient/series groups to one split only.
  - `build_case_index()` produces the training index table.
  - `CT25DDataset` loads stacks and labels from the index CSV.
  - `create_dataloaders()` creates train/val/test loaders for later training loops.
- [make_index.py](C:/AI/ai-medimg-playground/scripts/make_index.py): Day 4 execution entry point.
  - Discovers the current processed volume and pseudo-label bundle.
  - Writes `data_processed/index.csv`.
  - Builds a sample dataloader and exports `outputs/day4_batch_viz.png`.
  - Writes `outputs/day4_data_report.json` with counts and split summary.
- [visualization.py](C:/AI/ai-medimg-playground/src/visualization.py): Adds Day 4 batch visualization so the three input channels and center-slice mask can be checked together.
- [test_ct25d_dataset.py](C:/AI/ai-medimg-playground/tests/test_ct25d_dataset.py): Regression tests for stacking, split safety, and deterministic seeded transforms.
- [README.md](C:/AI/ai-medimg-playground/README.md): Updated to describe the Day 1 to Day 4 pipeline, new command, outputs, and the current single-series split limitation.
- [day4_data_module.md](C:/AI/ai-medimg-playground/docs/day4_data_module.md): Focused Day 4 runbook and result summary.

## Key points to remember

- 2.5D is a practical middle ground when full 3D training is too heavy for CPU-first experimentation.
- The center slice must remain the target label, even when context slices are repeated at the edges.
- Split assignment must happen at the patient/series level, not the slice level, to avoid leakage.
- Deterministic transforms matter for debugging and reproducibility, especially before full training starts.
- Batch visualization is the fastest sanity check that stack ordering and labels still line up.

## Current repo results

Running `scripts/make_index.py` on the current repository artifacts produced:

- `data_processed/index.csv` with `29` slice rows
- `outputs/day4_batch_viz.png`
- `outputs/day4_data_report.json`
- One detected patient and one detected series
- Split counts of `train=29`, `val=0`, `test=0`

This train-only split is intentional for the current state because the repository contains one patient/series group. A leakage-safe implementation should not invent validation or test rows by slicing the same study into multiple splits.

## Final summary

Day 4 delivered a real bridge from preprocessing and pseudo labels into model training: a 2.5D stack builder, slice-level dataset index, deterministic augmentation path, PyTorch dataset/dataloader interface, batch visualization, and documentation of the achieved outputs. The repository is now organized around a complete Day 1 to Day 4 progression, with the next practical step being the addition of more studies so the same data module can populate true train/val/test splits without leakage.
