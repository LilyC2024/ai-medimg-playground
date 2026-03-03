from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

# Use headless backend for server/terminal execution.
os.environ.setdefault("MPLBACKEND", "Agg")


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines.classical_seg import (  # noqa: E402
    generate_classical_masks,
    summarize_mask_quality,
)
from config import PreprocessConfig, SegmentationConfig, load_app_config  # noqa: E402
from dicom_loader import load_dicom_series  # noqa: E402
from preprocessing import run_preprocessing_pipeline, save_processed_volume  # noqa: E402
from visualization import save_day3_slice_overlays  # noqa: E402


def _parse_margin_triplet(raw_value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != 3:
        raise ValueError(
            f"Expected --crop-margin-mm as 'z,y,x' (3 values), got: {raw_value!r}",
        )
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Day 3 classical baseline: preprocess volume, generate pseudo-label masks, and export overlays.",
    )
    parser.add_argument("--series-dir", type=str, default=None, help="Path to input DICOM series directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for report/overlay outputs.")
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Directory for preprocessed volume and pseudo-label outputs.",
    )

    # Day-2 preprocess parameters.
    parser.add_argument("--xy-spacing-mm", type=float, default=1.0, help="Target y/x spacing in mm.")
    parser.add_argument("--target-z-mm", type=float, default=None, help="Optional target z spacing in mm.")
    parser.add_argument(
        "--keep-z-if-coarse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep original z if spacing is coarse.",
    )
    parser.add_argument("--coarse-z-threshold-mm", type=float, default=3.0, help="Coarse-z threshold in mm.")
    parser.add_argument("--resample-order", type=int, default=1, help="Resampling interpolation order (0/1/3).")
    parser.add_argument("--head-threshold-hu", type=float, default=-350.0, help="Head crop threshold in HU.")
    parser.add_argument("--mask-opening-iters", type=int, default=1, help="Head-mask opening iterations.")
    parser.add_argument("--mask-closing-iters", type=int, default=2, help="Head-mask closing iterations.")
    parser.add_argument("--crop-margin-mm", type=str, default="2,10,10", help="Crop margin in mm as z,y,x.")
    parser.add_argument("--hu-clip-min", type=float, default=-1000.0, help="Lower HU clipping bound.")
    parser.add_argument("--hu-clip-max", type=float, default=1000.0, help="Upper HU clipping bound.")
    parser.add_argument("--normalize-min", type=float, default=0.0, help="Normalization lower bound.")
    parser.add_argument("--normalize-max", type=float, default=1.0, help="Normalization upper bound.")
    parser.add_argument("--save-format", choices=["nii.gz", "npz"], default="nii.gz", help="Volume cache format.")

    # Day-3 segmentation parameters.
    parser.add_argument(
        "--bone-threshold-hu",
        type=float,
        default=300.0,
        help="HU threshold for bone mask candidate.",
    )
    parser.add_argument("--bone-open-iters", type=int, default=1, help="Bone mask opening iterations.")
    parser.add_argument("--bone-close-iters", type=int, default=1, help="Bone mask closing iterations.")
    parser.add_argument("--bone-min-voxels", type=int, default=64, help="Bone component size filter (voxels).")
    parser.add_argument(
        "--bone-keep-largest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep largest 3D connected component for bone mask.",
    )

    parser.add_argument("--brain-window-center", type=float, default=40.0, help="Brain window center (HU).")
    parser.add_argument("--brain-window-width", type=float, default=120.0, help="Brain window width (HU).")
    parser.add_argument(
        "--brain-norm-min",
        type=float,
        default=0.05,
        help="Lower threshold after brain-window normalization.",
    )
    parser.add_argument(
        "--brain-norm-max",
        type=float,
        default=0.95,
        help="Upper threshold after brain-window normalization.",
    )
    parser.add_argument(
        "--brain-head-threshold-hu",
        type=float,
        default=-300.0,
        help="Additional HU gate for brain-ish candidate.",
    )
    parser.add_argument("--brain-open-iters", type=int, default=0, help="Brain mask opening iterations.")
    parser.add_argument("--brain-close-iters", type=int, default=2, help="Brain mask closing iterations.")
    parser.add_argument(
        "--brain-fill-holes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fill holes in brain mask for smoother pseudo labels.",
    )
    parser.add_argument("--brain-min-voxels", type=int, default=256, help="Brain component size filter (voxels).")
    parser.add_argument(
        "--brain-keep-largest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep largest 3D connected component for brain mask.",
    )

    parser.add_argument(
        "--overlay-max-slices",
        type=int,
        default=None,
        help="Optional cap on overlays written to outputs/overlays.",
    )
    parser.add_argument(
        "--overlay-min-representative",
        type=int,
        default=10,
        help="Minimum representative slices when overlay capping is used.",
    )
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
        save_format=args.save_format,
    )


def _segmentation_config_from_args(args: argparse.Namespace) -> SegmentationConfig:
    return replace(
        SegmentationConfig(),
        bone_threshold_hu=args.bone_threshold_hu,
        bone_opening_iterations=args.bone_open_iters,
        bone_closing_iterations=args.bone_close_iters,
        bone_min_component_voxels=args.bone_min_voxels,
        bone_keep_largest_component=args.bone_keep_largest,
        brain_window_center=args.brain_window_center,
        brain_window_width=args.brain_window_width,
        brain_window_norm_min=args.brain_norm_min,
        brain_window_norm_max=args.brain_norm_max,
        brain_head_threshold_hu=args.brain_head_threshold_hu,
        brain_opening_iterations=args.brain_open_iters,
        brain_closing_iterations=args.brain_close_iters,
        brain_fill_holes=args.brain_fill_holes,
        brain_min_component_voxels=args.brain_min_voxels,
        brain_keep_largest_component=args.brain_keep_largest,
        overlay_max_slices=args.overlay_max_slices,
        overlay_min_representative_slices=args.overlay_min_representative,
    )


def _cache_file_name(save_format: str) -> str:
    if save_format == "nii.gz":
        return "volume.nii.gz"
    if save_format == "npz":
        return "volume.npz"
    raise ValueError(f"Unsupported save format: {save_format}")


def _save_pseudo_label_volume(
    output_dir: Path,
    brain_mask_3d: np.ndarray,
    bone_mask_3d: np.ndarray,
    pseudo_labels_3d: np.ndarray,
    spacing_zyx: tuple[float, float, float],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pseudo_labels_3d.npz"
    np.savez_compressed(
        output_path,
        brain_mask=brain_mask_3d.astype(np.uint8),
        bone_mask=bone_mask_3d.astype(np.uint8),
        pseudo_labels=pseudo_labels_3d.astype(np.uint8),
        spacing_zyx=np.asarray(spacing_zyx, dtype=np.float32),
    )
    return output_path


def _save_slice_masks(
    output_dir: Path,
    brain_mask_3d: np.ndarray,
    bone_mask_3d: np.ndarray,
    pseudo_labels_3d: np.ndarray,
) -> int:
    slices_dir = output_dir / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    for z_idx in range(brain_mask_3d.shape[0]):
        file_path = slices_dir / f"slice_{z_idx:03d}.npz"
        np.savez_compressed(
            file_path,
            brain_mask=brain_mask_3d[z_idx].astype(np.uint8),
            bone_mask=bone_mask_3d[z_idx].astype(np.uint8),
            pseudo_labels=pseudo_labels_3d[z_idx].astype(np.uint8),
        )
        saved_count += 1
    return saved_count


def _build_quality_flags(
    brain_stats: dict[str, float | int],
    bone_stats: dict[str, float | int],
    overlap_ratio: float,
) -> list[str]:
    flags: list[str] = []
    if float(brain_stats["largest_component_ratio"]) < 0.85:
        flags.append("Brain mask has fragmented components; consider stronger closing or larger min-component size.")
    if float(bone_stats["largest_component_ratio"]) < 0.85:
        flags.append("Bone mask has fragmented components; consider raising bone threshold or cleanup iterations.")
    if overlap_ratio > 0.20:
        flags.append("Brain/bone overlap is high; tune brain window thresholds or bone threshold.")
    if int(brain_stats["max_contiguous_run_slices"]) < 8:
        flags.append("Brain mask continuity is short across z; inspect possible crop or threshold failure.")
    if int(bone_stats["non_empty_slice_count"]) < 8:
        flags.append("Bone mask appears in few slices; inspect threshold for this scan.")
    return flags


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    preprocess_config = _preprocess_config_from_args(args)
    segmentation_config = _segmentation_config_from_args(args)
    config = load_app_config(
        series_dir=args.series_dir,
        output_dir=args.output_dir,
        processed_dir=args.processed_dir,
        preprocess=preprocess_config,
        segmentation=segmentation_config,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.processed_dir.mkdir(parents=True, exist_ok=True)

    raw_start = time.perf_counter()
    dicom_volume = load_dicom_series(config.series_dir)
    raw_load_seconds = time.perf_counter() - raw_start

    preprocess_start = time.perf_counter()
    preprocess_result = run_preprocessing_pipeline(
        volume_hu=dicom_volume.volume_hu,
        spacing_zyx=dicom_volume.metadata.spacing_zyx,
        preprocess_config=config.preprocess,
    )
    preprocess_seconds = time.perf_counter() - preprocess_start

    processed_cache_path = config.processed_dir / _cache_file_name(config.preprocess.save_format)
    save_processed_volume(
        volume=preprocess_result.processed_volume,
        spacing_zyx=preprocess_result.resampled_spacing_zyx,
        output_path=processed_cache_path,
    )

    seg_start = time.perf_counter()
    segmentation = generate_classical_masks(
        volume_hu=preprocess_result.cropped_volume_hu,
        segmentation_config=config.segmentation,
    )
    seg_seconds = time.perf_counter() - seg_start

    pseudo_dir = config.processed_dir / "pseudo_labels"
    pseudo_volume_path = _save_pseudo_label_volume(
        output_dir=pseudo_dir,
        brain_mask_3d=segmentation.brain_mask_3d,
        bone_mask_3d=segmentation.bone_mask_3d,
        pseudo_labels_3d=segmentation.pseudo_labels_3d,
        spacing_zyx=preprocess_result.resampled_spacing_zyx,
    )
    saved_slice_count = _save_slice_masks(
        output_dir=pseudo_dir,
        brain_mask_3d=segmentation.brain_mask_3d,
        bone_mask_3d=segmentation.bone_mask_3d,
        pseudo_labels_3d=segmentation.pseudo_labels_3d,
    )

    overlays_dir = config.output_dir / "overlays"
    overlay_files = save_day3_slice_overlays(
        volume_hu=preprocess_result.cropped_volume_hu,
        brain_mask_3d=segmentation.brain_mask_3d,
        bone_mask_3d=segmentation.bone_mask_3d,
        output_dir=overlays_dir,
        max_slices=config.segmentation.overlay_max_slices,
        min_representative_slices=config.segmentation.overlay_min_representative_slices,
    )

    brain_stats = summarize_mask_quality(
        mask_3d=segmentation.brain_mask_3d,
        spacing_zyx=preprocess_result.resampled_spacing_zyx,
    )
    bone_stats = summarize_mask_quality(
        mask_3d=segmentation.bone_mask_3d,
        spacing_zyx=preprocess_result.resampled_spacing_zyx,
    )

    overlap_voxels = int(
        np.logical_and(segmentation.brain_mask_3d.astype(bool), segmentation.bone_mask_3d.astype(bool)).sum(),
    )
    union_voxels = int(
        np.logical_or(segmentation.brain_mask_3d.astype(bool), segmentation.bone_mask_3d.astype(bool)).sum(),
    )
    overlap_ratio = float(overlap_voxels / max(union_voxels, 1))

    quality_flags = _build_quality_flags(brain_stats, bone_stats, overlap_ratio)
    failure_modes = [
        "Partial-volume effect at skull edges can create mixed-intensity voxels and broken contours.",
        "Beam-hardening streaks may appear as false high-HU regions and leak into bone masks.",
        "Motion artifacts can fragment soft-tissue continuity and lower largest-component ratio.",
        "Post-surgical metal can dominate HU thresholding and distort skull candidate masks.",
        "Thick slices (coarse z) reduce through-plane continuity and weaken 3D morphology assumptions.",
    ]
    future_actions = [
        "Introduce adaptive thresholding and slice-wise intensity normalization for domain shifts.",
        "Add atlas- or template-based priors to constrain intracranial region plausibility.",
        "Use CRF/graph-cut refinement or lightweight U-Net fine-tuning on pseudo labels.",
        "Run uncertainty checks and manual spot-review on low-continuity or high-overlap slices.",
    ]

    report = {
        "series_dir": str(config.series_dir),
        "input_shape_zyx": [int(v) for v in dicom_volume.volume_hu.shape],
        "cropped_shape_zyx": [int(v) for v in preprocess_result.cropped_volume_hu.shape],
        "spacing_zyx_mm": [float(v) for v in preprocess_result.resampled_spacing_zyx],
        "pseudo_labels_volume_path": str(pseudo_volume_path),
        "per_slice_mask_count": int(saved_slice_count),
        "overlays_dir": str(overlays_dir),
        "overlay_file_count": int(len(overlay_files)),
        "metrics": {
            "brain_mask": brain_stats,
            "bone_mask": bone_stats,
            "brain_bone_overlap_voxels": overlap_voxels,
            "brain_bone_overlap_ratio": overlap_ratio,
            "timings_seconds": {
                "raw_load": raw_load_seconds,
                "preprocess": preprocess_seconds,
                "segmentation": seg_seconds,
            },
        },
        "quality_flags": quality_flags,
        "failure_modes": failure_modes,
        "future_actions": future_actions,
        "segmentation_config": {
            "bone_threshold_hu": config.segmentation.bone_threshold_hu,
            "bone_opening_iterations": config.segmentation.bone_opening_iterations,
            "bone_closing_iterations": config.segmentation.bone_closing_iterations,
            "bone_min_component_voxels": config.segmentation.bone_min_component_voxels,
            "bone_keep_largest_component": config.segmentation.bone_keep_largest_component,
            "brain_window_center": config.segmentation.brain_window_center,
            "brain_window_width": config.segmentation.brain_window_width,
            "brain_window_norm_min": config.segmentation.brain_window_norm_min,
            "brain_window_norm_max": config.segmentation.brain_window_norm_max,
            "brain_head_threshold_hu": config.segmentation.brain_head_threshold_hu,
            "brain_opening_iterations": config.segmentation.brain_opening_iterations,
            "brain_closing_iterations": config.segmentation.brain_closing_iterations,
            "brain_fill_holes": config.segmentation.brain_fill_holes,
            "brain_min_component_voxels": config.segmentation.brain_min_component_voxels,
            "brain_keep_largest_component": config.segmentation.brain_keep_largest_component,
        },
    }

    report_path = config.output_dir / "day3_classical_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Day 3 classical baseline complete.")
    print(f"Input series: {config.series_dir}")
    print(f"Pseudo-label volume: {pseudo_volume_path}")
    print(f"Per-slice pseudo labels: {saved_slice_count} files in {pseudo_dir / 'slices'}")
    print(f"Overlay PNGs: {len(overlay_files)} files in {overlays_dir}")
    print(f"Report: {report_path}")
    print(
        "Timing (seconds): "
        f"raw_load={raw_load_seconds:.3f}, preprocess={preprocess_seconds:.3f}, segmentation={seg_seconds:.3f}",
    )
    if quality_flags:
        print("Quality flags:")
        for flag in quality_flags:
            print(f"- {flag}")
    else:
        print("Quality flags: none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
