# Interview Notes

## 30-Second Framing

This project is a compact medical-imaging pipeline built to show engineering judgment across the full lifecycle: raw DICOM ingestion, preprocessing, pseudo-label generation, lightweight modeling, robustness checks, and deployable CPU inference.

## Key Design Choices

### Why use pseudo labels?

- No manual annotations were bundled with the sample data.
- A classical segmentation baseline made it possible to build and test the rest of the pipeline honestly.
- The repo documents that downstream metrics reflect agreement with bootstrap labels, not clinical truth.

### Why 2.5D instead of 3D?

- The sample scan uses thick slices, so full 3D continuity is limited anyway.
- 2.5D captures local through-plane context while staying CPU-friendly.
- It keeps training, export, and inference lightweight enough for a portfolio repo.

### Why CPU-first deployment?

- It is easier to reproduce on a fresh machine.
- ONNX Runtime on CPU is a realistic demo target for small operational environments.
- The project goal was reliability and readability more than absolute throughput.

## Important Tradeoffs

- **Classical bootstrap:** fast and explainable, but still sensitive to artifacts.
- **Single-series evaluation:** enough to validate the plumbing, not enough to claim generalization.
- **Temperature scaling:** improves calibration behavior on holdout pseudo labels, but does not convert uncertainty into clinical risk.
- **Docker/FastAPI support:** helpful for deployment storytelling, but the CLI is the primary reference path.

## Failure Cases to Acknowledge

- Beam hardening, motion, or metal artifacts can confuse the classical pseudo-label stage.
- Thick slices make fine 3D structure continuity harder to capture.
- Rare structures and class overlap remain hard because the supervision signal is derived and class-imbalanced.
- Confidence maps are useful for review prioritization, but not for safety-critical decision making.

## Improvements Already Applied

- Adaptive threshold candidate selection for more robust Day 3 brain-mask generation
- Buffered intra-series holdouts to reduce overly optimistic evaluation
- Class-balanced loss weighting for stronger Day 5 training
- Temperature scaling for better-calibrated confidence behavior
- ONNX export validation and deterministic CPU CLI packaging

## What I Would Improve Next

1. Add expert-reviewed labels for at least a small validation set.
2. Expand to multiple studies and patient-level splits.
3. Compare the current 2.5D model against a compact 3D baseline.
4. Add artifact-focused stress tests and more explicit failure detection.
5. Package reports and overlays into a simpler reviewer-facing dashboard.

## Five-Minute Walkthrough Structure

1. Problem: show the repo as an end-to-end imaging pipeline, not just a model.
2. Data constraints: explain why pseudo labels and 2.5D were reasonable choices here.
3. Build path: Day 1-7 moved from ingestion to deployment in discrete, testable stages.
4. Improvement pass: describe what was weak, what changed, and what improved.
5. Deployment and credibility: close with CLI, ONNX validation, tests, and honest limitations.
