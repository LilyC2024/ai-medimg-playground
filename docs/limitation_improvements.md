# Limitation Improvements

## Scope

This pass focused on reducing the most tractable limitations listed in the repository documentation without fabricating new data or pretending pseudo labels are clinical truth. The review covered the segmentation baseline, data split logic, training loop, inference path, deployment runtime, reports, and supporting tests.

## Limitation-by-Limitation Review

### 1. Classical thresholds were brittle on the Day 3 brain-ish mask

Original limitation:
- Fixed thresholds could collapse the soft-tissue mask on unusual intensity distributions or artifact-heavy scans.

Applied method:
- Added an adaptive threshold sweep in `src/baselines/classical_seg.py`.
- The code now evaluates multiple `head_threshold_hu`, `window_width`, `norm_min`, and `norm_max` combinations.
- Each candidate is scored using continuity, largest-component ratio, head-volume ratio, and bone-overlap penalty.
- The selected parameters and top-ranked alternatives are written into `outputs/day3_classical_report.json`.

Observed improvement on the sample series:
- brain-mask voxel count: `1,476 -> 10,731`
- brain-mask volume: `7.38 mL -> 53.66 mL`
- non-empty slices: `12 -> 25`
- contiguous-run length: `12 -> 25`

Interpretation:
- The pseudo-label target is much less collapsed than before, which gives the later model stages a more plausible brain-ish supervision signal.

### 2. Single-series evaluation used train fallback

Original limitation:
- With one series, Day 4 assigned everything to train, and Day 5 evaluation fell back to the training split.

Applied method:
- Added buffered intra-series split assignment in `src/data/ct25d_dataset.py`.
- Updated `scripts/make_index.py` to create contiguous holdout bands plus a context buffer to reduce adjacent-slice leakage.
- Training now uses those val slices instead of train fallback when available.

Observed improvement on the sample series:
- split layout changed from `29 train / 0 val / 0 test` to `17 train / 4 val / 4 test / 4 buffer`
- Day 5 `eval_split` changed from `train-fallback` to `val`

Interpretation:
- This is still not a patient-level generalization test, but it is a more honest check than evaluating directly on the same slices used for optimization.

### 3. Pseudo-label supervision and class imbalance hurt learning

Original limitation:
- The model was trained on weak pseudo labels and had severe class imbalance, especially for the rare overlap class.

Applied method:
- Strengthened the Day 3 pseudo labels with the adaptive selector.
- Added class-balanced cross-entropy weighting in `src/models/unet_small.py` and `scripts/train.py`.
- Retrained the lightweight U-Net on the refreshed labels and the buffered holdout split.

Observed improvement:
- best eval Dice: `0.2456 -> 0.3302`
- full-series inference Dice against the refreshed pseudo labels: `0.1629 -> 0.1900`
- class 1 Dice in full-series inference: `0.0225 -> 0.0820`
- class 3 Dice in full-series inference: near `0.0000 -> 0.1301`

Interpretation:
- The training target is now harder and broader, so raw before/after Dice is not perfectly apples-to-apples, but the holdout metric and class-wise behavior both improved.

### 4. Uncertainty was heuristic and not calibrated

Original limitation:
- Entropy-based uncertainty acted as a rough review aid, not a calibrated probability signal.

Applied method:
- Added temperature scaling in `src/calibration.py`.
- Fitted the temperature on the buffered holdout split after training.
- Stored calibration metadata in the checkpoint and propagated it into inference/deployment reports.

Observed improvement:
- temperature: `0.0743`
- holdout ECE: `0.4526 -> 0.1588`

Interpretation:
- The uncertainty output is still not clinical risk calibration, but it is measurably less miscalibrated than the raw logits.

### 5. Deployment needed to stay aligned with the improved model

Original limitation:
- Once the model changes, the ONNX artifact and deployment reports can become stale.

Applied method:
- Re-exported ONNX from the refreshed checkpoint.
- Revalidated ONNX-vs-PyTorch numerical equivalence in `deploy/inference_runtime.py`.
- Re-ran the Day 7 CLI and repeated the run to confirm deterministic output.

Observed improvement:
- deployment Dice against refreshed pseudo labels: `0.2745 -> 0.3331`
- repeated CLI hash match: `249012A93CCF75E841E57CF6975067443A1A907B939FBDF5B9CBD34B908124A2`
- ONNX validation remained within tolerance.

Interpretation:
- The deployment path remains trustworthy after the training/label updates and still behaves deterministically on CPU.

### 6. Limitations that remain only partially solved

Still unresolved or only partially mitigated:
- No manual ground-truth labels were added, so the project still cannot claim clinical accuracy.
- Only one real series is available, so even the improved val/test split is intra-series rather than patient-level.
- Thick-slice geometry is still inherent to the sample data; the buffered split and improved reporting make this easier to detect, but they do not remove the acquisition limitation.
- Temperature scaling improves calibration relative to pseudo labels, not true clinician-verified outcomes.

## Files Touched in This Pass

Core code:
- `src/baselines/classical_seg.py`
- `src/data/ct25d_dataset.py`
- `src/models/unet_small.py`
- `src/calibration.py`
- `scripts/classical_baseline.py`
- `scripts/make_index.py`
- `scripts/train.py`
- `scripts/infer.py`
- `scripts/report.py`
- `deploy/inference_runtime.py`

Validation and tests:
- `tests/test_ct25d_dataset.py`

Artifacts refreshed during verification:
- `data_processed/index.csv`
- `saved_models/best.pt`
- `outputs/day3_classical_report.json`
- `outputs/day4_data_report.json`
- `outputs/day5_train_report.json`
- `outputs/day5_infer_report.json`
- `outputs/day5_calibration_report.json`
- `outputs/day7_infer_demo/day7_infer_report.json`

## Final Takeaway

This improvement pass did not ?solve? the hardest scientific limitations, because some of them are fundamentally data limitations. What it did do is make the existing project more honest, more robust, and more useful: the pseudo labels are less collapsed, the evaluation split is less misleading, the training loop handles imbalance better, the uncertainty is less miscalibrated, and the deployment artifact stays aligned with the improved checkpoint.
