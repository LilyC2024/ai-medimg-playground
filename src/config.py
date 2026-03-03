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
    mask_opening_iterations: int = 1
    # Morphology cleanup iterations for binary closing (fill small mask holes).
    mask_closing_iterations: int = 2
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
class AppConfig:
    series_dir: Path
    output_dir: Path
    processed_dir: Path
    preprocess: PreprocessConfig


def _resolve_path(raw_value: str | None, fallback: Path) -> Path:
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    return fallback.resolve()


def load_app_config(
    series_dir: str | None = None,
    output_dir: str | None = None,
    processed_dir: str | None = None,
    preprocess: PreprocessConfig | None = None,
) -> AppConfig:
    configured_series_dir = series_dir or os.getenv("DICOM_SERIES_DIR")
    configured_output_dir = output_dir or os.getenv("DICOM_OUTPUT_DIR")
    configured_processed_dir = processed_dir or os.getenv("DICOM_PROCESSED_DIR")
    return AppConfig(
        series_dir=_resolve_path(configured_series_dir, DEFAULT_SERIES_DIR),
        output_dir=_resolve_path(configured_output_dir, DEFAULT_OUTPUT_DIR),
        processed_dir=_resolve_path(configured_processed_dir, DEFAULT_PROCESSED_DIR),
        preprocess=preprocess or PreprocessConfig(),
    )
