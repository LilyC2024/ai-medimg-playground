# Release Notes: v1.0

## Summary

`v1.0` marks the Day 8 portfolio release of the project. The repository now tells a complete story from raw DICOM intake to deployable CPU inference, with documentation and tests organized for fast review.

## What Changed in Day 8

- Polished the main `README.md` with:
  - a clearer project pitch
  - architecture diagram
  - quickstart and repro checklist
  - dataset/privacy notes
  - results snapshot
  - cleaner documentation map
- Added `docs/one_page_summary.md` as an executive overview
- Added `docs/interview_notes.md` for design choices, tradeoffs, failure cases, and next steps
- Added `tests/test_e2e_smoke.py` to exercise the deployment CLI on the bundled sample case
- Tightened the docs structure so the repository has one primary documentation tree

## Release State

- Bundled sample DICOM case included
- Saved checkpoint included
- ONNX export included
- CPU CLI deployment path included
- Optional FastAPI and Docker path included
- Unit tests plus smoke test included

## Known Scope Limits

- Metrics are still measured against pseudo labels rather than expert annotations
- The repo still contains one bundled CT series
- The project is intended as an engineering portfolio artifact, not a clinical product
