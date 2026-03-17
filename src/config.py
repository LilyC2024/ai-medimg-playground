from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERIES_DIR = REPO_ROOT / "data" / "dicom_series_01"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data_processed"


@dataclass(frozen=True)
class PreprocessConfig:
    """Tunable parameters for Day 2 preprocessing.

    Most fields here are intentionally exposed so future experiments can be done
    from one place without changing preprocessing logic.
    """

    # Target in-plane spacing for x/y resampling (millimeters).
    target_spacing_xy_mm: float = 1.0
    # Optional z target spacing. When None, z spacing is kept unchanged.
    target_spacing_z_mm: float | None = None
    # Keep original z spacing when it is very coarse (common in thick-slice CT).
    keep_z_if_coarse: bool = True
    # "Very coarse" threshold in millimeters used by keep_z_if_coarse.
    coarse_z_threshold_mm: float = 3.0
    # Interpolation order passed to scipy.ndimage.zoom (0=nearest, 1=linear, 3=cubic).
    resample_order: int = 1

    # HU threshold for separating air from patient/head region.
    head_threshold_hu: float = -350.0
    # Morphology cleanup iterations for binary opening (remove tiny noise blobs).
    mask_opening_iterations: int = 10
    # Morphology cleanup iterations for binary closing (fill small mask holes).
    mask_closing_iterations: int = 10
    # Margin (z, y, x) in millimeters added around the detected head bounding box.
    crop_margin_mm_zyx: tuple[float, float, float] = (2.0, 10.0, 10.0)

    # Intensity clipping range in HU before normalization.
    hu_clip_min: float = -1000.0
    hu_clip_max: float = 1000.0
    # Output normalization range.
    normalize_min: float = 0.0
    normalize_max: float = 1.0

    # Saved processed volume format: "nii.gz" or "npz".
    save_format: str = "nii.gz"


@dataclass(frozen=True)
class SegmentationConfig:
    """Tunable parameters for Day 3 classical segmentation baseline."""

    # Bone is typically high HU. Higher threshold gives cleaner but thinner skull masks.
    bone_threshold_hu: float = -100.0
    # Morphology cleanup for bone mask.
    bone_opening_iterations: int = 1
    bone_closing_iterations: int = 1
    # Remove tiny disconnected islands below this size (voxels).
    bone_min_component_voxels: int = 32
    # Keep only largest connected component to enforce anatomical plausibility.
    bone_keep_largest_component: bool = True

    # Brain-ish candidate via brain window normalization then thresholding.
    brain_window_center: float = 40.0
    brain_window_width: float = 120.0
    # Normalized window range to keep as brain-ish tissue candidate.
    brain_window_norm_min: float = 0.05
    brain_window_norm_max: float = 0.95
    # Additional HU gate to suppress air/non-patient regions.
    brain_head_threshold_hu: float = -10.0
    # Morphology cleanup for brain mask.
    brain_opening_iterations: int = 0
    brain_closing_iterations: int = 2
    # Fill interior holes after cleanup.
    brain_fill_holes: bool = True
    # Remove tiny disconnected islands below this size (voxels).
    brain_min_component_voxels: int = 256
    # Keep only largest connected component to enforce anatomical plausibility.
    brain_keep_largest_component: bool = True

    # Overlay export controls.
    overlay_max_slices: int | None = None
    overlay_min_representative_slices: int = 20


@dataclass(frozen=True)
class AppConfig:
    series_dir: Path
    output_dir: Path
    processed_dir: Path
    preprocess: PreprocessConfig
    segmentation: SegmentationConfig


def _resolve_path(raw_value: str | None, fallback: Path) -> Path:
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    return fallback.resolve()


def load_app_config(
    series_dir: str | None = None,
    output_dir: str | None = None,
    processed_dir: str | None = None,
    preprocess: PreprocessConfig | None = None,
    segmentation: SegmentationConfig | None = None,
) -> AppConfig:
    configured_series_dir = series_dir or os.getenv("DICOM_SERIES_DIR")
    configured_output_dir = output_dir or os.getenv("DICOM_OUTPUT_DIR")
    configured_processed_dir = processed_dir or os.getenv("DICOM_PROCESSED_DIR")
    return AppConfig(
        series_dir=_resolve_path(configured_series_dir, DEFAULT_SERIES_DIR),
        output_dir=_resolve_path(configured_output_dir, DEFAULT_OUTPUT_DIR),
        processed_dir=_resolve_path(configured_processed_dir, DEFAULT_PROCESSED_DIR),
        preprocess=preprocess or PreprocessConfig(),
        segmentation=segmentation or SegmentationConfig(),
    )
