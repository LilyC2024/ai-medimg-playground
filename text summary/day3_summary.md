# Day 3 Essay: Classical Segmentation Baseline and Pseudo Labels

Day 3 focused on building a practical segmentation baseline without deep learning so the project can progress even before manual annotations are available. The strategy was to combine domain-intuitive CT rules with robust cleanup: threshold-based candidates for bone and brain-ish tissue, then morphology and connected-component plausibility constraints to suppress noise and artifacts.

The pipeline starts from raw DICOM and reuses preprocessing to produce a cropped HU volume. From there, classical masks are generated in 3D, cleaned, and exported both per-slice and as full volumes to support later supervised training. Because no ground truth labels are assumed, an evaluation harness was added to check continuity, component quality, and overlap behavior, along with per-slice overlay visuals for fast human review.

## What each file contributes

- [classical_seg.py](C:/AI/ai-medimg-playground/src/baselines/classical_seg.py): Day 3 core logic.
  - `bone_mask()`: high-HU threshold candidate with morphology and component filtering.
  - `brain_mask()`: brain-window-based candidate with cleanup and plausibility constraints.
  - `generate_classical_masks()`: produces bone, brain-ish, and combined pseudo-label volume.
  - `summarize_mask_quality()`: non-GT quality metrics (volume, continuity, component dominance).
- [classical_baseline.py](C:/AI/ai-medimg-playground/scripts/classical_baseline.py): End-to-end Day 3 runner.
  - Raw DICOM -> preprocessing -> classical segmentation.
  - Writes pseudo labels to `data_processed/pseudo_labels/`.
  - Writes per-slice overlays to `outputs/overlays/`.
  - Writes evaluation report to `outputs/day3_classical_report.json`.
- [visualization.py](C:/AI/ai-medimg-playground/src/visualization.py): Adds per-slice overlay export utility for human QA.
- [config.py](C:/AI/ai-medimg-playground/src/config.py): Adds `SegmentationConfig` with tunable Day 3 parameters.
- [test_classical_seg.py](C:/AI/ai-medimg-playground/tests/test_classical_seg.py): Regression tests for mask generation and quality summaries.
- [README.md](C:/AI/ai-medimg-playground/README.md): Documents Day 3 run command, outputs, tunable parameters, failure modes, and next-step plan.

## Key points to remember

- Classical rules can produce useful pseudo labels quickly when manual labels are not yet available.
- Morphology + connected-components are essential to remove speckles and small non-anatomical islands.
- Per-slice overlays are not optional; they are the primary qualitative validation when GT is absent.
- Non-GT metrics (component ratio, contiguous slice runs, overlap ratio) provide automated sanity checks.
- Pseudo labels are a bridge to supervised models, not a final segmentation endpoint.

## Structured use case

1. Run `scripts/classical_baseline.py` on the target DICOM series.
2. Confirm pseudo-label outputs exist in `data_processed/pseudo_labels/`.
3. Review `outputs/overlays/slice_*.png` across multiple z positions.
4. Check `outputs/day3_classical_report.json` for continuity/component warnings.
5. Tune thresholds/morphology parameters if overlays or metrics show failure patterns.
6. Use pseudo labels as bootstrap targets for later supervised training.

## Final summary

Day 3 delivered a full non-deep-learning segmentation baseline with tunable mask generation, postprocessing cleanup, automated quality summaries, and large-scale visual review outputs. The result is a workable pseudo-label pipeline that runs end-to-end from raw DICOM and provides a concrete starting point for future supervised refinement.
