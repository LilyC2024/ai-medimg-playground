# Day 2 Essay: Preprocessing Pipeline and Cached Reload

Day 2 focused on turning a valid CT volume into a modeling-ready input through consistent preprocessing. The goal was to reduce irrelevant background, standardize spatial resolution and intensity scale, and save a reusable cached artifact so repeated experimentation is faster and more reproducible.

The workflow starts from the Day 1 HU volume and applies a fixed sequence: spacing control, head ROI extraction, HU clipping/windowing-style normalization, visual QA, and cached serialization. A timing comparison was added to quantify practical impact: loading raw DICOM each time versus loading a preprocessed cache file. This creates a cleaner, faster, and better-documented input path for subsequent segmentation or classification tasks.

## What each file contributes

- [config.py](C:/AI/ai-medimg-playground/src/config.py): Adds Day 2 preprocessing configuration in one place (target spacing, coarse-z policy, crop threshold/margins, HU clip, normalization range, and save format).
- [preprocessing.py](C:/AI/ai-medimg-playground/src/preprocessing.py): Core preprocessing logic.
  - Chooses target spacing with optional coarse-z preservation.
  - Resamples volume (z/y/x-aware).
  - Builds head mask from HU threshold and morphology cleanup.
  - Extracts largest-component bounding box and expands by physical margin.
  - Crops ROI, clips HU, and normalizes output range.
  - Saves/loads processed volumes as `.nii.gz` or `.npz`.
- [visualization.py](C:/AI/ai-medimg-playground/src/visualization.py): Adds Day 2 before/after QA figure generation with crop-box overlay to confirm anatomy is not cut.
- [preprocess_series.py](C:/AI/ai-medimg-playground/scripts/preprocess_series.py): Day 2 execution entry point.
  - Runs full preprocessing pipeline from raw DICOM.
  - Writes `data_processed/volume.nii.gz` (or `.npz`).
  - Writes `outputs/day2_before_after.png` and `outputs/day2_preprocess_report.json`.
  - Measures raw-load versus cached-load timing and reports speedup.
- [test_preprocessing.py](C:/AI/ai-medimg-playground/tests/test_preprocessing.py): Regression checks for spacing rules, ROI mask/bbox behavior, normalization bounds, and cache round-trip.
- [README.md](C:/AI/ai-medimg-playground/README.md): Documents Day 2 pipeline, tunable CLI parameters, and expected artifacts.

## Key points to remember

- In-plane resampling improves consistency across scans; coarse z can be intentionally preserved to avoid unstable through-plane interpolation.
- Threshold + connected-component ROI crop removes large non-anatomical background and reduces memory/IO cost.
- HU clipping and normalization produce stable intensity ranges for downstream models.
- Before/after QA visualization with crop-box overlay is essential to verify anatomy preservation.
- Cache speedup comes from skipping repeated DICOM decode + preprocessing and reading one compact processed file.

## Structured use case

1. Ensure Day 1 ingestion works and DICOM series path is configured.
2. Run `scripts/preprocess_series.py` with defaults or tuned flags.
3. Confirm `outputs/day2_before_after.png` shows intact anatomy within crop bounds.
4. Review `outputs/day2_preprocess_report.json` for shape/spacing, crop box, parameters, and timing.
5. Use `data_processed/volume.nii.gz` (or `.npz`) as fast reload input for training/inference experiments.
6. Re-run tests to ensure preprocessing behavior remains stable.

## Final summary

Day 2 delivered a practical preprocessing baseline: spacing harmonization, head-focused cropping, HU normalization, visual QA, serialized cache output, and measured reload speed gains. The result is a cleaner and faster data path that is easier to tune, validate, and reuse in later modeling stages.
