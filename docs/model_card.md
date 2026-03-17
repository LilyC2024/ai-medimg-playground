# Model Card

## Model Details

- Name: `UNetSmall` 2.5D CT head segmentation baseline
- Version: `v0.6-robustness`
- Framework: PyTorch
- Inputs: 3-slice 2.5D stacks built from preprocessed axial CT slices
- Outputs: 4-class pseudo-label segmentation (`background`, `brain-ish`, `bone`, `overlap`) plus an uncertainty proxy

## Intended Use

- Education and prototyping for medical imaging pipelines
- Stress-testing DICOM ingestion, preprocessing, pseudo-label learning, and reporting workflows
- Prioritized manual review using slice-level metrics and uncertainty summaries

## Not Intended For

- Clinical diagnosis, treatment planning, or autonomous triage
- Any deployment where calibrated risk estimates or regulatory-grade validation are required
- Performance claims on unseen patient populations without manual ground-truth labels

## Data

- Current repository state uses one CT head DICOM series
- Supervision comes from Day 3 classical pseudo labels, not expert annotations
- Preprocessing includes spacing-aware resampling, ROI crop, HU clipping, and normalization

## Evaluation

- Slice and volume metrics are computed against pseudo labels
- Day 6 adds connected-component cleanup, small-object removal, hole filling, smoothing, and entropy/TTA uncertainty proxies
- The standardized report highlights best/worst slices for manual review

## Constraints

- Single-series data means weak generalization evidence and limited leakage-safe validation
- Thick slices, missing spacing tags, motion, beam hardening, and metal can degrade mask plausibility
- Uncertainty is heuristic and should be used as a review signal, not a decision threshold

## Failure Modes

- Missing or malformed DICOM spacing tags prevent safe physical resampling
- Motion and metal artifacts can fragment connected components or create false positive high-HU regions
- Thick-slice scans weaken 3D morphology assumptions and increase unstable boundaries
- Pseudo-label supervision can reinforce Day 3 rule-based errors during Day 5/6 learning and inference

## Human Oversight

- Review the Day 6 report before trusting segmentation outputs
- Inspect worst slices and high-uncertainty slices manually
- Treat any validation warnings in preprocessing/classical reports as blockers for downstream analysis
