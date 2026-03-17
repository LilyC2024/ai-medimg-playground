from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

# Use non-interactive backend before importing visualization/matplotlib modules.
os.environ.setdefault("MPLBACKEND", "Agg")


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import PreprocessConfig, load_app_config  # noqa: E402
from dicom_loader import load_dicom_series  # noqa: E402
from preprocessing import load_processed_volume, run_preprocessing_pipeline, save_processed_volume  # noqa: E402
from visualization import save_day2_before_after  # noqa: E402


def _parse_margin_triplet(raw_value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != 3:
        raise ValueError(
            f"Expected --crop-margin-mm as 'z,y,x' (3 values), got: {raw_value!r}",
        )
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Day 2 preprocessing: resample, crop head ROI, window/normalize, and cache volume.",
    )
    parser.add_argument("--series-dir", type=str, default=None, help="Path to input DICOM series directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for QA images and JSON metadata.")
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Directory for saved preprocessed volume cache.",
    )

    # Spacing controls.
    parser.add_argument(
        "--xy-spacing-mm",
        type=float,
        default=1.0,
        help="Target in-plane spacing in mm for y/x (smaller => higher resolution, more memory).",
    )
    parser.add_argument(
        "--target-z-mm",
        type=float,
        default=None,
        help="Optional target z spacing in mm. If omitted, z spacing is preserved.",
    )
    parser.add_argument(
        "--keep-z-if-coarse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep original z spacing when original spacing >= --coarse-z-threshold-mm.",
    )
    parser.add_argument(
        "--coarse-z-threshold-mm",
        type=float,
        default=3.0,
        help="Threshold in mm above which z is considered coarse.",
    )
    parser.add_argument(
        "--resample-order",
        type=int,
        default=1,
        help="Interpolation order for resampling: 0 nearest, 1 linear, 3 cubic.",
    )

    # ROI crop controls.
    parser.add_argument(
        "--head-threshold-hu",
        type=float,
        default=-350.0,
        help="HU threshold used to separate air/background from head.",
    )
    parser.add_argument(
        "--mask-opening-iters",
        type=int,
        default=1,
        help="Binary opening iterations (increases remove small mask speckles).",
    )
    parser.add_argument(
        "--mask-closing-iters",
        type=int,
        default=2,
        help="Binary closing iterations (increases fill small mask holes).",
    )
    parser.add_argument(
        "--crop-margin-mm",
        type=str,
        default="2,10,10",
        help="Crop box expansion margin in mm as 'z,y,x'.",
    )

    # Intensity controls.
    parser.add_argument("--hu-clip-min", type=float, default=-1000.0, help="Lower HU clip bound.")
    parser.add_argument("--hu-clip-max", type=float, default=1000.0, help="Upper HU clip bound.")
    parser.add_argument("--normalize-min", type=float, default=0.0, help="Normalized output lower bound.")
    parser.add_argument("--normalize-max", type=float, default=1.0, help="Normalized output upper bound.")

    parser.add_argument(
        "--save-format",
        choices=["nii.gz", "npz"],
        default="nii.gz",
        help="Cache format for processed volume.",
    )
    return parser


def _preprocess_config_from_args(args: argparse.Namespace) -> PreprocessConfig:
    margin_mm_zyx = _parse_margin_triplet(args.crop_margin_mm)
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
        crop_margin_mm_zyx=margin_mm_zyx,
        hu_clip_min=args.hu_clip_min,
        hu_clip_max=args.hu_clip_max,
        normalize_min=args.normalize_min,
        normalize_max=args.normalize_max,
        save_format=args.save_format,
    )


def _cache_file_name(save_format: str) -> str:
    if save_format == "nii.gz":
        return "volume.nii.gz"
    if save_format == "npz":
        return "volume.npz"
    raise ValueError(f"Unsupported save format: {save_format}")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    preprocess_config = _preprocess_config_from_args(args)
    config = load_app_config(
        series_dir=args.series_dir,
        output_dir=args.output_dir,
        processed_dir=args.processed_dir,
        preprocess=preprocess_config,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.processed_dir.mkdir(parents=True, exist_ok=True)

    raw_start = time.perf_counter()
    dicom_volume = load_dicom_series(config.series_dir)
    raw_load_seconds = time.perf_counter() - raw_start

    preprocess_start = time.perf_counter()
    result = run_preprocessing_pipeline(
        volume_hu=dicom_volume.volume_hu,
        spacing_zyx=dicom_volume.metadata.spacing_zyx,
        preprocess_config=config.preprocess,
    )
    preprocess_seconds = time.perf_counter() - preprocess_start

    cache_path = config.processed_dir / _cache_file_name(config.preprocess.save_format)
    save_start = time.perf_counter()
    saved_path = save_processed_volume(
        volume=result.processed_volume,
        spacing_zyx=result.resampled_spacing_zyx,
        output_path=cache_path,
    )
    save_seconds = time.perf_counter() - save_start

    cached_start = time.perf_counter()
    cached_volume, cached_spacing_zyx = load_processed_volume(saved_path)
    cached_load_seconds = time.perf_counter() - cached_start

    if cached_volume.shape != result.processed_volume.shape:
        raise RuntimeError(
            f"Cached volume shape mismatch. Expected {result.processed_volume.shape}, got {cached_volume.shape}.",
        )

    comparison_path = config.output_dir / "day2_before_after.png"
    save_day2_before_after(
        resampled_volume_hu=result.resampled_volume_hu,
        cropped_volume_hu=result.cropped_volume_hu,
        processed_volume=result.processed_volume,
        crop_bbox_zyx=result.crop_bbox_zyx,
        output_path=comparison_path,
    )

    timings = {
        "raw_load_seconds": raw_load_seconds,
        "preprocess_seconds": preprocess_seconds,
        "save_seconds": save_seconds,
        "cached_load_seconds": cached_load_seconds,
        "raw_vs_cached_speedup": raw_load_seconds / max(cached_load_seconds, 1e-9),
    }

    z_kept_original = abs(result.input_spacing_zyx[0] - result.resampled_spacing_zyx[0]) < 1e-6
    report = {
        "series_dir": str(config.series_dir),
        "processed_volume_path": str(saved_path),
        "comparison_figure_path": str(comparison_path),
        "input_shape_zyx": [int(v) for v in dicom_volume.volume_hu.shape],
        "resampled_shape_zyx": [int(v) for v in result.resampled_volume_hu.shape],
        "cropped_shape_zyx": [int(v) for v in result.cropped_volume_hu.shape],
        "output_shape_zyx": [int(v) for v in result.processed_volume.shape],
        "input_spacing_zyx_mm": [float(v) for v in result.input_spacing_zyx],
        "output_spacing_zyx_mm": [float(v) for v in result.resampled_spacing_zyx],
        "crop_bbox_zyx": result.crop_bbox_zyx.to_dict(),
        "clip_hu_range": [float(config.preprocess.hu_clip_min), float(config.preprocess.hu_clip_max)],
        "normalize_range": [float(config.preprocess.normalize_min), float(config.preprocess.normalize_max)],
        "z_spacing_preserved": z_kept_original,
        "cached_spacing_zyx_mm": [float(v) for v in cached_spacing_zyx],
        "dicom_validation_messages": list(dicom_volume.metadata.validation_messages),
        "timings": timings,
        "preprocess_config": {
            "target_spacing_xy_mm": config.preprocess.target_spacing_xy_mm,
            "target_spacing_z_mm": config.preprocess.target_spacing_z_mm,
            "keep_z_if_coarse": config.preprocess.keep_z_if_coarse,
            "coarse_z_threshold_mm": config.preprocess.coarse_z_threshold_mm,
            "resample_order": config.preprocess.resample_order,
            "head_threshold_hu": config.preprocess.head_threshold_hu,
            "mask_opening_iterations": config.preprocess.mask_opening_iterations,
            "mask_closing_iterations": config.preprocess.mask_closing_iterations,
            "crop_margin_mm_zyx": list(config.preprocess.crop_margin_mm_zyx),
            "hu_clip_min": config.preprocess.hu_clip_min,
            "hu_clip_max": config.preprocess.hu_clip_max,
            "normalize_min": config.preprocess.normalize_min,
            "normalize_max": config.preprocess.normalize_max,
            "save_format": config.preprocess.save_format,
        },
    }

    report_path = config.output_dir / "day2_preprocess_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Day 2 preprocessing complete.")
    print(f"Input series: {config.series_dir}")
    print(f"Saved cache: {saved_path}")
    print(f"Saved QA image: {comparison_path}")
    print(f"Saved report: {report_path}")
    print(
        "Timing (seconds): "
        f"raw_load={raw_load_seconds:.3f}, "
        f"preprocess={preprocess_seconds:.3f}, "
        f"cached_load={cached_load_seconds:.3f}, "
        f"speedup={timings['raw_vs_cached_speedup']:.2f}x",
    )

    if z_kept_original:
        print(
            "Note: z-spacing was preserved. "
            "This usually happens when --target-z-mm is omitted or original z spacing is coarse.",
        )
    if dicom_volume.metadata.validation_messages:
        print("Validation notes:")
        for message in dicom_volume.metadata.validation_messages:
            print(f"- {message}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
