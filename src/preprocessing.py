from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from config import PreprocessConfig

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover - dependency is expected in runtime envs
    sitk = None


@dataclass(frozen=True)
class BoundingBox3D:
    """Axis-aligned 3D bounding box using inclusive-exclusive indexing."""

    z_min: int
    z_max: int
    y_min: int
    y_max: int
    x_min: int
    x_max: int

    def as_slices(self) -> tuple[slice, slice, slice]:
        return (
            slice(self.z_min, self.z_max),
            slice(self.y_min, self.y_max),
            slice(self.x_min, self.x_max),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "z_min": self.z_min,
            "z_max": self.z_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "x_min": self.x_min,
            "x_max": self.x_max,
        }


@dataclass(frozen=True)
class PreprocessResult:
    """Outputs from the day-2 preprocessing pipeline."""

    resampled_volume_hu: np.ndarray
    cropped_volume_hu: np.ndarray
    processed_volume: np.ndarray
    input_spacing_zyx: tuple[float, float, float]
    resampled_spacing_zyx: tuple[float, float, float]
    crop_bbox_zyx: BoundingBox3D


def choose_target_spacing(
    spacing_zyx: tuple[float, float, float],
    preprocess_config: PreprocessConfig,
) -> tuple[float, float, float]:
    """Build output spacing from config while preserving coarse z when requested."""

    z_spacing, _, _ = spacing_zyx
    target_z = z_spacing

    if preprocess_config.target_spacing_z_mm is not None:
        should_keep_original_z = (
            preprocess_config.keep_z_if_coarse
            and z_spacing >= preprocess_config.coarse_z_threshold_mm
        )
        if not should_keep_original_z:
            target_z = float(preprocess_config.target_spacing_z_mm)

    target_xy = float(preprocess_config.target_spacing_xy_mm)
    return (float(target_z), target_xy, target_xy)


def resample_volume_to_spacing(
    volume_hu: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    target_spacing_zyx: tuple[float, float, float],
    interpolation_order: int = 1,
) -> np.ndarray:
    """Resample a z/y/x volume to target spacing using scipy zoom.

    Parameters to tune:
    - `interpolation_order`: 0 keeps sharp labels, 1 is safe default for CT intensity.
    - `target_spacing_zyx`: lower spacing values increase resolution and memory use.
    """

    zoom_factors = tuple(
        float(original) / float(target)
        for original, target in zip(spacing_zyx, target_spacing_zyx, strict=True)
    )

    if np.allclose(zoom_factors, (1.0, 1.0, 1.0), atol=1e-4):
        return volume_hu.astype(np.float32, copy=True)

    resampled = ndimage.zoom(
        volume_hu,
        zoom=zoom_factors,
        order=int(interpolation_order),
        mode="nearest",
    )
    return resampled.astype(np.float32, copy=False)


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    labels, component_count = ndimage.label(mask)
    if component_count == 0:
        raise ValueError("Head mask is empty. Adjust threshold/morphology parameters.")

    component_sizes = ndimage.sum(mask, labels=labels, index=np.arange(1, component_count + 1))
    largest_component_id = int(np.argmax(component_sizes)) + 1
    return labels == largest_component_id


def create_head_mask(
    volume_hu: np.ndarray,
    threshold_hu: float,
    opening_iterations: int = 1,
    closing_iterations: int = 2,
) -> np.ndarray:
    """Create a binary mask separating patient/head from surrounding air.

    Parameters to tune:
    - `threshold_hu`: lower values include more low-density tissue/couch.
    - `opening_iterations`: removes small noise components.
    - `closing_iterations`: fills small interior holes.
    """

    mask = volume_hu > float(threshold_hu)
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)

    if opening_iterations > 0:
        mask = ndimage.binary_opening(mask, structure=structure, iterations=int(opening_iterations))
    if closing_iterations > 0:
        mask = ndimage.binary_closing(mask, structure=structure, iterations=int(closing_iterations))

    return _largest_connected_component(mask)


def bbox_from_mask(mask: np.ndarray) -> BoundingBox3D:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("Cannot compute crop box from an empty mask.")

    zyx_min = coordinates.min(axis=0)
    zyx_max = coordinates.max(axis=0) + 1
    return BoundingBox3D(
        z_min=int(zyx_min[0]),
        z_max=int(zyx_max[0]),
        y_min=int(zyx_min[1]),
        y_max=int(zyx_max[1]),
        x_min=int(zyx_min[2]),
        x_max=int(zyx_max[2]),
    )


def expand_bbox_with_margin(
    bbox: BoundingBox3D,
    margin_mm_zyx: tuple[float, float, float],
    spacing_zyx: tuple[float, float, float],
    volume_shape_zyx: tuple[int, int, int],
) -> BoundingBox3D:
    """Expand a bounding box by physical margin and clamp to volume boundaries."""

    margin_voxels = np.rint(
        np.asarray(margin_mm_zyx, dtype=np.float32) / np.asarray(spacing_zyx, dtype=np.float32)
    ).astype(np.int32)
    margin_z, margin_y, margin_x = [int(value) for value in margin_voxels]

    z_max_bound, y_max_bound, x_max_bound = volume_shape_zyx
    return BoundingBox3D(
        z_min=max(0, bbox.z_min - margin_z),
        z_max=min(z_max_bound, bbox.z_max + margin_z),
        y_min=max(0, bbox.y_min - margin_y),
        y_max=min(y_max_bound, bbox.y_max + margin_y),
        x_min=max(0, bbox.x_min - margin_x),
        x_max=min(x_max_bound, bbox.x_max + margin_x),
    )


def crop_volume_to_bbox(volume: np.ndarray, bbox: BoundingBox3D) -> np.ndarray:
    return volume[bbox.as_slices()].astype(np.float32, copy=False)


def clip_and_normalize_hu(
    volume_hu: np.ndarray,
    clip_min_hu: float,
    clip_max_hu: float,
    normalize_min: float = 0.0,
    normalize_max: float = 1.0,
) -> np.ndarray:
    """Clip HU to a chosen range and map to output numeric range."""

    if clip_max_hu <= clip_min_hu:
        raise ValueError("clip_max_hu must be greater than clip_min_hu.")
    if normalize_max <= normalize_min:
        raise ValueError("normalize_max must be greater than normalize_min.")

    clipped = np.clip(volume_hu, float(clip_min_hu), float(clip_max_hu))
    unit = (clipped - float(clip_min_hu)) / float(clip_max_hu - clip_min_hu)
    normalized = unit * float(normalize_max - normalize_min) + float(normalize_min)
    return normalized.astype(np.float32, copy=False)


def run_preprocessing_pipeline(
    volume_hu: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    preprocess_config: PreprocessConfig,
) -> PreprocessResult:
    target_spacing_zyx = choose_target_spacing(spacing_zyx, preprocess_config)
    resampled_volume_hu = resample_volume_to_spacing(
        volume_hu=volume_hu,
        spacing_zyx=spacing_zyx,
        target_spacing_zyx=target_spacing_zyx,
        interpolation_order=preprocess_config.resample_order,
    )

    head_mask = create_head_mask(
        resampled_volume_hu,
        threshold_hu=preprocess_config.head_threshold_hu,
        opening_iterations=preprocess_config.mask_opening_iterations,
        closing_iterations=preprocess_config.mask_closing_iterations,
    )
    bbox = bbox_from_mask(head_mask)
    bbox_with_margin = expand_bbox_with_margin(
        bbox=bbox,
        margin_mm_zyx=preprocess_config.crop_margin_mm_zyx,
        spacing_zyx=target_spacing_zyx,
        volume_shape_zyx=tuple(int(dim) for dim in resampled_volume_hu.shape),
    )

    cropped_volume_hu = crop_volume_to_bbox(resampled_volume_hu, bbox_with_margin)
    processed_volume = clip_and_normalize_hu(
        cropped_volume_hu,
        clip_min_hu=preprocess_config.hu_clip_min,
        clip_max_hu=preprocess_config.hu_clip_max,
        normalize_min=preprocess_config.normalize_min,
        normalize_max=preprocess_config.normalize_max,
    )

    return PreprocessResult(
        resampled_volume_hu=resampled_volume_hu,
        cropped_volume_hu=cropped_volume_hu,
        processed_volume=processed_volume,
        input_spacing_zyx=tuple(float(value) for value in spacing_zyx),
        resampled_spacing_zyx=tuple(float(value) for value in target_spacing_zyx),
        crop_bbox_zyx=bbox_with_margin,
    )


def _looks_like_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def save_npz_volume(
    volume: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    output_path: str | Path,
) -> Path:
    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_file,
        volume=volume.astype(np.float32),
        spacing_zyx=np.asarray(spacing_zyx, dtype=np.float32),
    )
    return output_file


def load_npz_volume(path: str | Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    npz_file = Path(path).expanduser().resolve()
    with np.load(npz_file) as data:
        volume = data["volume"].astype(np.float32)
        spacing = tuple(float(value) for value in data["spacing_zyx"].tolist())
    return volume, spacing


def save_nifti_volume(
    volume: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    output_path: str | Path,
) -> Path:
    if sitk is None:
        raise RuntimeError("SimpleITK is required to save NIfTI volumes.")

    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    image = sitk.GetImageFromArray(volume.astype(np.float32))
    image.SetSpacing((float(spacing_zyx[2]), float(spacing_zyx[1]), float(spacing_zyx[0])))
    sitk.WriteImage(image, str(output_file), useCompression=True)
    return output_file


def load_nifti_volume(path: str | Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    if sitk is None:
        raise RuntimeError("SimpleITK is required to load NIfTI volumes.")

    image = sitk.ReadImage(str(Path(path).expanduser().resolve()))
    spacing_xyz = image.GetSpacing()
    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing_zyx = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
    return volume, spacing_zyx


def save_processed_volume(
    volume: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    output_path: str | Path,
) -> Path:
    output_file = Path(output_path).expanduser().resolve()
    if _looks_like_nifti(output_file):
        return save_nifti_volume(volume, spacing_zyx, output_file)
    if output_file.suffix.lower() == ".npz":
        return save_npz_volume(volume, spacing_zyx, output_file)
    raise ValueError(f"Unsupported output format for path: {output_file}")


def load_processed_volume(path: str | Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    source_path = Path(path).expanduser().resolve()
    if _looks_like_nifti(source_path):
        return load_nifti_volume(source_path)
    if source_path.suffix.lower() == ".npz":
        return load_npz_volume(source_path)
    raise ValueError(f"Unsupported input format for path: {source_path}")
