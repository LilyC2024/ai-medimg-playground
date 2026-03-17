# Day 4 Data Module: 2.5D Dataset Builder

Day 4 extends the Day 1 to Day 3 pipeline into a training-ready data path. Instead of feeding a single axial slice to a model, the new setup stacks three neighboring slices `(z-1, z, z+1)` and predicts the center slice label. This 2.5D approach is a practical compromise for CPU-bound experiments because it keeps the data interface simple like 2D training while still exposing a small amount of through-plane context from short CT volumes.

## What Day 4 adds

- `src/data/ct25d_dataset.py`
  - Edge-safe 2.5D stacking with clamped neighbors at the first and last slices.
  - Patient/series-group split assignment to prevent leakage when multiple studies are available.
  - Deterministic CPU-friendly augmentations: flips, small rotations, and optional intensity jitter.
  - `CT25DDataset` and `create_dataloaders()` for PyTorch-based training loops.
- `scripts/make_index.py`
  - Builds `data_processed/index.csv` from the Day 2 processed volume and Day 3 pseudo labels.
  - Writes `outputs/day4_batch_viz.png` for batch-level sanity checking.
  - Writes `outputs/day4_data_report.json` with shape, spacing, split counts, and transform settings.
- `tests/test_ct25d_dataset.py`
  - Verifies edge handling for 2.5D stacks.
  - Verifies split assignment stays group-safe.
  - Verifies transform output is deterministic when seeded.

## How 2.5D works here

For each slice index `z`, the input tensor is built as:

```text
channel 0 -> slice max(z-1, 0)
channel 1 -> slice z
channel 2 -> slice min(z+1, depth-1)
label     -> pseudo label for slice z
```

This keeps the center slice aligned with the target mask while giving the model one slice of context above and below. Edge slices are handled by repetition, so the first sample uses `(0, 0, 1)` and the last sample uses `(depth-2, depth-1, depth-1)`.

## Current repository results

The current repo state contains one processed CT series, so the Day 4 split logic correctly falls back to a train-only assignment to avoid leakage. The generated artifacts show:

- Input case shape: `29 x 207 x 171`
- Resampled spacing: `5.0 x 1.0 x 1.0 mm`
- Index rows written: `29`
- Unique patients: `1`
- Unique series: `1`
- Split counts: `train=29`, `val=0`, `test=0`

This is the right behavior for the current data inventory. Once more patient/series groups are added, the same index builder will assign whole groups to `train`, `val`, and `test` without mixing slices from one study across splits.

## Produced outputs

- `data_processed/index.csv`
- `outputs/day4_batch_viz.png`
- `outputs/day4_data_report.json`

## Validation status

- Stacking is correct: verified by `tests/test_ct25d_dataset.py`
- Center slice matches label: verified in dataset construction and batch visualization
- Splits are patient/series-safe: group-level split logic is enforced
- Transforms are deterministic when seeded: verified by tests
- Full repository test suite: `12/12` passing
