from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = REPO_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deploy.inference_runtime import (  # noqa: E402
    attach_reference_metrics,
    run_deployment_inference,
)
from config import PreprocessConfig  # noqa: E402


def _parse_margin_triplet(raw_value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected margin as 'z,y,x', got {raw_value!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Day 7 deployment CLI: DICOM folder -> ONNX Runtime CPU inference -> mask volume + overlays.",
    )
    parser.add_argument("--series-dir", type=str, default=str(REPO_ROOT / "data" / "dicom_series_01"))
    parser.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "saved_models" / "best.pt"))
    parser.add_argument("--onnx-path", type=str, default=str(REPO_ROOT / "onnx" / "model.onnx"))
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "outputs" / "day7_infer_demo"))
    parser.add_argument("--processed-dir", type=str, default=str(REPO_ROOT / "data_processed"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--mask-format", choices=["npz", "nii.gz"], default="npz")
    parser.add_argument("--output-formats", nargs="+", choices=["mask", "overlays"], default=["mask", "overlays"])
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--disable-postprocess", action="store_true")

    parser.add_argument("--xy-spacing-mm", type=float, default=1.0)
    parser.add_argument("--target-z-mm", type=float, default=None)
    parser.add_argument("--keep-z-if-coarse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--coarse-z-threshold-mm", type=float, default=3.0)
    parser.add_argument("--resample-order", type=int, default=1)
    parser.add_argument("--head-threshold-hu", type=float, default=-350.0)
    parser.add_argument("--mask-opening-iters", type=int, default=1)
    parser.add_argument("--mask-closing-iters", type=int, default=2)
    parser.add_argument("--crop-margin-mm", type=str, default="2,10,10")
    parser.add_argument("--hu-clip-min", type=float, default=-1000.0)
    parser.add_argument("--hu-clip-max", type=float, default=1000.0)
    parser.add_argument("--normalize-min", type=float, default=0.0)
    parser.add_argument("--normalize-max", type=float, default=1.0)
    return parser


def _preprocess_config_from_args(args: argparse.Namespace) -> PreprocessConfig:
    return replace(
        PreprocessConfig(),
        target_spacing_xy_mm=args.xy_spacing_mm,
        target_spacing_z_mm=args.target_z_mm,
        keep_z_if_coarse=args.keep_z_if_coarse,
        coarse_z_threshold_mm=args.coarse_z_threshold_mm,
        resample_order=args.resample_order,
        head_threshold_hu=args.head_threshold_hu,
        mask_opening_iterations=args.mask_opening_iters,
        mask_closing_iterations=args.mask_closing_iters,
        crop_margin_mm_zyx=_parse_margin_triplet(args.crop_margin_mm),
        hu_clip_min=args.hu_clip_min,
        hu_clip_max=args.hu_clip_max,
        normalize_min=args.normalize_min,
        normalize_max=args.normalize_max,
    )


def main() -> int:
    args = _build_parser().parse_args()
    result = run_deployment_inference(
        series_dir=args.series_dir,
        checkpoint_path=args.checkpoint,
        onnx_path=args.onnx_path,
        output_dir=args.output_dir,
        preprocess_config=_preprocess_config_from_args(args),
        batch_size=args.batch_size,
        num_threads=args.threads,
        export_onnx=args.force_export,
        validate_onnx=not args.skip_validation,
        mask_output_format=args.mask_format,
        save_overlays="overlays" in set(args.output_formats),
        enable_postprocess=not args.disable_postprocess,
    )

    report = dict(result.report)
    reference_path = Path(args.processed_dir).expanduser().resolve() / "pseudo_labels" / "pseudo_labels_3d.npz"
    if reference_path.exists():
        import numpy as np

        with np.load(reference_path) as data:
            report = attach_reference_metrics(report, result.prediction_volume, data["pseudo_labels"].astype(np.uint8))
    result.output_paths.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved mask volume: {result.output_paths.mask_volume_path}")
    if "overlays" in set(args.output_formats):
        print(f"Saved overlays: {result.output_paths.overlay_dir}")
    print(f"Saved deployment report: {result.output_paths.report_path}")
    if result.output_paths.validation_report_path is not None:
        print(f"Saved ONNX validation: {result.output_paths.validation_report_path}")
        print(f"ONNX within tolerance: {report['onnx_validation']['within_tolerance']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
