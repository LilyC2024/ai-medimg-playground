# Day 4 Data Module: 2.5D Dataset Builder

Day 4 turns the preprocessing and pseudo-label outputs into a training-ready 2.5D data path. Each sample uses the local stack `(z-1, z, z+1)` to predict the center slice mask, which keeps the CPU cost close to 2D training while still exposing limited through-plane context.

## What Day 4 Owns

- `src/data/ct25d_dataset.py`
  - edge-safe 2.5D stack construction
  - deterministic augmentations
  - group-safe split logic
  - buffered intra-series holdout logic for the single-case repo state
- `scripts/make_index.py`
  - index generation from the processed volume and pseudo labels
  - split summary export
  - batch-visualization export
- `tests/test_ct25d_dataset.py`
  - stack correctness
  - deterministic transforms
  - split-behavior regression checks

## 2.5D Layout

```text
channel 0 -> slice max(z-1, 0)
channel 1 -> slice z
channel 2 -> slice min(z+1, depth-1)
label     -> pseudo label for slice z
```

The first and last slices are handled by repetition so the target always remains aligned with the center channel.

## Current Repository Snapshot

The repository still contains one CT series, but the improved Day 4 path no longer reports a pure train-only split. Instead, it creates contiguous intra-series holdout bands with a buffer region to reduce adjacent-slice leakage.

- indexed slices: `29`
- shape: `29 x 207 x 171`
- spacing: `5.0 x 1.0 x 1.0 mm`
- split counts: `17 train / 4 val / 4 test / 4 buffer`

This is still not patient-level generalization, but it is a more careful internal evaluation layout than the older train-fallback setup.

## Produced Outputs

- `data_processed/index.csv`
- `outputs/day4_batch_viz.png`
- `outputs/day4_data_report.json`

## Validation Status

- stack construction: verified
- center-slice/label alignment: verified
- intra-series holdout creation: verified
- transform determinism: verified
- current unit-test suite: `15/15` passing
