# One-Page Summary

## Project

**AI Medical Imaging Playground** is a compact CT head segmentation portfolio project built to demonstrate an end-to-end medical imaging workflow on a small, transparent example.

## Problem

Medical-imaging repos often show isolated notebooks or model code without a credible path from raw input to deployable inference. This project was structured to show the whole chain:
- raw DICOM loading
- preprocessing and caching
- bootstrap labeling when manual labels are unavailable
- lightweight model training
- evaluation and calibration
- deployable CPU inference

## Method

- **Input:** one bundled axial CT head series
- **Preprocessing:** HU conversion, spacing-aware resampling, head ROI crop, intensity normalization
- **Bootstrap supervision:** Day 3 classical segmentation creates pseudo labels for brain/bone/overlap regions
- **Learning stage:** a small 2.5D U-Net predicts the center slice from three adjacent slices
- **Quality layer:** buffered intra-series holdouts, segmentation metrics, temperature scaling, uncertainty maps, and standardized reports
- **Deployment:** PyTorch checkpoint exported to ONNX and served through a deterministic CPU CLI, with optional FastAPI and Docker support

## Why 2.5D

The sample scan has thick slices and the repo is intentionally CPU-friendly. A 2.5D model gives some through-plane context without the cost and complexity of full 3D training.

## Results

- Holdout split after the improvement pass: `17 train / 4 val / 4 test / 4 buffer`
- Day 5 best holdout Dice: `0.3302`
- Calibration ECE improved from `0.4526` to `0.1588`
- ONNX and PyTorch matched within tolerance on the validation batch
- Day 7 CLI produced deterministic mask outputs across repeated runs

## Deliverables

- Reproducible scripts for each stage
- Saved checkpoint in `saved_models/best.pt`
- Exported model in `onnx/model.onnx`
- Deployable CLI in `deploy/cli_infer.py`
- Tests including an end-to-end smoke run
- Documentation designed to support both review and interview discussion

## Scope and Honesty

This repo is meant to be credible, not overstated:
- labels are pseudo labels, not manual clinical annotations
- the bundled dataset is one CT series
- results show pipeline maturity and engineering care, not clinical readiness

## Best Use in a Hiring Conversation

Use this project to tell a full story:
1. Start with the problem and constraints.
2. Explain why the pipeline was staged this way.
3. Show how you improved weak points instead of hiding them.
4. End with deployment, testing, and what you would do next with more data.
