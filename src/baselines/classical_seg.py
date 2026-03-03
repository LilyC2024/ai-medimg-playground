from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from config import SegmentationConfig
from visualization import apply_window


@dataclass(frozen=True)
class SegmentationResult:
    """Container for 3D masks and a compact multi-class pseudo-label volume."""

    bone_mask_3d: np.ndarray
    brain_mask_3d: np.ndarray
    pseudo_labels_3d: np.ndarray


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    return ndimage.label(mask, structure=structure)


def _remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0:
        return mask.astype(bool, copy=False)
    labels, count = _connected_components(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)

    component_ids = np.arange(1, count + 1, dtype=np.int32)
    component_sizes = ndimage.sum(mask, labels=labels, index=component_ids)
    keep_ids = component_ids[np.asarray(component_sizes) >= int(min_size)]
    if keep_ids.size == 0:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labels, keep_ids)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = _connected_components(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)

    component_ids = np.arange(1, count + 1, dtype=np.int32)
    component_sizes = ndimage.sum(mask, labels=labels, index=component_ids)
    largest_id = int(component_ids[int(np.argmax(component_sizes))])
    return labels == largest_id


def _postprocess_binary_mask(
    mask: np.ndarray,
    opening_iterations: int,
    closing_iterations: int,
    fill_holes: bool,
    min_component_voxels: int,
    keep_largest_component: bool,
) -> np.ndarray:
    """Generic plausibility cleanup used by both bone and brain candidates."""

    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    cleaned = mask.astype(bool, copy=False)

    if opening_iterations > 0:
        cleaned = ndimage.binary_opening(cleaned, structure=structure, iterations=int(opening_iterations))
    if closing_iterations > 0:
        cleaned = ndimage.binary_closing(cleaned, structure=structure, iterations=int(closing_iterations))
    if fill_holes:
        cleaned = ndimage.binary_fill_holes(cleaned)

    cleaned = _remove_small_components(cleaned, min_component_voxels)
    if keep_largest_component:
        cleaned = _largest_component(cleaned)
    return cleaned.astype(np.uint8)


def bone_mask(
    volume_hu: np.ndarray,
    threshold_hu: float = 300.0,
    opening_iterations: int = 1,
    closing_iterations: int = 1,
    min_component_voxels: int = 64,
    keep_largest_component: bool = True,
) -> np.ndarray:
    """Create a skull/bone candidate mask from HU threshold.

    Tunable parameters:
    - `threshold_hu`: bone intensity threshold in HU; increasing this suppresses cancellous/soft bone.
    - `opening_iterations`: larger values remove isolated speckles but can erode thin structures.
    - `closing_iterations`: larger values bridge small gaps in skull contour.
    - `min_component_voxels`: filters tiny islands that are unlikely to be anatomy.
    - `keep_largest_component`: enforces one dominant connected structure.
    """

    candidate = volume_hu >= float(threshold_hu)
    return _postprocess_binary_mask(
        candidate,
        opening_iterations=opening_iterations,
        closing_iterations=closing_iterations,
        fill_holes=False,
        min_component_voxels=min_component_voxels,
        keep_largest_component=keep_largest_component,
    )


def brain_mask(
    volume_hu: np.ndarray,
    window_center: float = 40.0,
    window_width: float = 120.0,
    norm_min: float = 0.05,
    norm_max: float = 0.95,
    head_threshold_hu: float = -300.0,
    opening_iterations: int = 1,
    closing_iterations: int = 2,
    fill_holes: bool = True,
    min_component_voxels: int = 256,
    keep_largest_component: bool = True,
) -> np.ndarray:
    """Create a brain-ish soft tissue mask using brain window + morphology.

    Tunable parameters:
    - `window_center/window_width`: define the HU window transformed to [0, 1].
    - `norm_min/norm_max`: threshold range on windowed intensity for candidate extraction.
    - `head_threshold_hu`: excludes very low-HU air regions around the patient.
    - morphology/component arguments trade off smoothness vs preserving fine details.
    """

    if norm_max <= norm_min:
        raise ValueError("norm_max must be greater than norm_min.")
    if window_width <= 0:
        raise ValueError("window_width must be positive.")

    windowed = np.empty_like(volume_hu, dtype=np.float32)
    for z_idx in range(volume_hu.shape[0]):
        windowed[z_idx] = apply_window(volume_hu[z_idx], center=window_center, width=window_width)

    soft_tissue_candidate = (windowed >= float(norm_min)) & (windowed <= float(norm_max))
    inside_head_candidate = volume_hu >= float(head_threshold_hu)
    candidate = soft_tissue_candidate & inside_head_candidate

    cleaned = _postprocess_binary_mask(
        candidate,
        opening_iterations=opening_iterations,
        closing_iterations=closing_iterations,
        fill_holes=fill_holes,
        min_component_voxels=min_component_voxels,
        keep_largest_component=keep_largest_component,
    )
    if int(cleaned.sum()) > 0:
        return cleaned

    # Fallback for scans where strict window thresholds and morphology over-prune.
    relaxed_candidate = (volume_hu >= float(head_threshold_hu)) & (volume_hu <= float(window_center + window_width))
    relaxed = _postprocess_binary_mask(
        relaxed_candidate,
        opening_iterations=0,
        closing_iterations=max(1, int(closing_iterations)),
        fill_holes=True,
        min_component_voxels=max(16, int(min_component_voxels) // 4),
        keep_largest_component=True,
    )
    return relaxed


def generate_classical_masks(
    volume_hu: np.ndarray,
    segmentation_config: SegmentationConfig,
) -> SegmentationResult:
    """Run day-3 classical segmentation and return binary masks + label volume.

    Label convention in `pseudo_labels_3d`:
    - 0: background
    - 1: brain-ish tissue
    - 2: bone
    - 3: overlap (rare ambiguity where both candidates are true)
    """

    bone = bone_mask(
        volume_hu=volume_hu,
        threshold_hu=segmentation_config.bone_threshold_hu,
        opening_iterations=segmentation_config.bone_opening_iterations,
        closing_iterations=segmentation_config.bone_closing_iterations,
        min_component_voxels=segmentation_config.bone_min_component_voxels,
        keep_largest_component=segmentation_config.bone_keep_largest_component,
    ).astype(bool)

    brain = brain_mask(
        volume_hu=volume_hu,
        window_center=segmentation_config.brain_window_center,
        window_width=segmentation_config.brain_window_width,
        norm_min=segmentation_config.brain_window_norm_min,
        norm_max=segmentation_config.brain_window_norm_max,
        head_threshold_hu=segmentation_config.brain_head_threshold_hu,
        opening_iterations=segmentation_config.brain_opening_iterations,
        closing_iterations=segmentation_config.brain_closing_iterations,
        fill_holes=segmentation_config.brain_fill_holes,
        min_component_voxels=segmentation_config.brain_min_component_voxels,
        keep_largest_component=segmentation_config.brain_keep_largest_component,
    ).astype(bool)

    labels = np.zeros_like(volume_hu, dtype=np.uint8)
    labels[brain] = 1
    labels[bone] = 2
    labels[brain & bone] = 3

    return SegmentationResult(
        bone_mask_3d=bone.astype(np.uint8),
        brain_mask_3d=brain.astype(np.uint8),
        pseudo_labels_3d=labels,
    )


def _contiguous_non_empty_runs(slice_counts: np.ndarray) -> list[int]:
    runs: list[int] = []
    current = 0
    for count in slice_counts:
        if int(count) > 0:
            current += 1
        elif current > 0:
            runs.append(current)
            current = 0
    if current > 0:
        runs.append(current)
    return runs


def summarize_mask_quality(
    mask_3d: np.ndarray,
    spacing_zyx: tuple[float, float, float],
) -> dict[str, float | int]:
    """Compute non-GT evaluation metrics for a binary 3D mask."""

    mask_bool = mask_3d.astype(bool)
    voxel_count = int(mask_bool.sum())
    voxel_volume_mm3 = float(spacing_zyx[0] * spacing_zyx[1] * spacing_zyx[2])
    mask_volume_ml = (voxel_count * voxel_volume_mm3) / 1000.0

    per_slice_counts = mask_bool.sum(axis=(1, 2))
    non_empty_slice_count = int(np.count_nonzero(per_slice_counts))
    runs = _contiguous_non_empty_runs(per_slice_counts)

    labels, component_count = _connected_components(mask_bool)
    largest_component_voxels = 0
    if component_count > 0:
        component_ids = np.arange(1, component_count + 1, dtype=np.int32)
        component_sizes = ndimage.sum(mask_bool, labels=labels, index=component_ids)
        largest_component_voxels = int(np.max(component_sizes))

    largest_component_ratio = (
        float(largest_component_voxels) / float(max(voxel_count, 1))
    )

    return {
        "voxel_count": voxel_count,
        "volume_ml": float(mask_volume_ml),
        "non_empty_slice_count": non_empty_slice_count,
        "slice_coverage_ratio": float(non_empty_slice_count / max(mask_3d.shape[0], 1)),
        "connected_components_3d": int(component_count),
        "largest_component_ratio": float(largest_component_ratio),
        "max_contiguous_run_slices": int(max(runs) if runs else 0),
        "run_count": int(len(runs)),
    }
