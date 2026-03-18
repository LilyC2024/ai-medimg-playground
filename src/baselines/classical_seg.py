from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from config import SegmentationConfig
from visualization import apply_window


@dataclass(frozen=True)
class SegmentationResult:
    """Container for 3D masks, pseudo labels, and adaptive-selection diagnostics."""

    bone_mask_3d: np.ndarray
    brain_mask_3d: np.ndarray
    pseudo_labels_3d: np.ndarray
    brain_selection: dict[str, Any]


@dataclass(frozen=True)
class BrainCandidate:
    mask_3d: np.ndarray
    stats: dict[str, float | int]
    score: float
    params: dict[str, float | int]
    head_ratio: float
    overlap_ratio: float


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
    """Create a skull/bone candidate mask from HU threshold."""

    candidate = volume_hu >= float(threshold_hu)
    return _postprocess_binary_mask(
        candidate,
        opening_iterations=opening_iterations,
        closing_iterations=closing_iterations,
        fill_holes=False,
        min_component_voxels=min_component_voxels,
        keep_largest_component=keep_largest_component,
    )


def _brain_candidate_mask(
    volume_hu: np.ndarray,
    *,
    bone_mask_3d: np.ndarray | None,
    window_center: float,
    window_width: float,
    norm_min: float,
    norm_max: float,
    head_threshold_hu: float,
    opening_iterations: int,
    closing_iterations: int,
    fill_holes: bool,
    min_component_voxels: int,
    keep_largest_component: bool,
) -> np.ndarray:
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
    if bone_mask_3d is not None:
        candidate &= ~bone_mask_3d.astype(bool)

    return _postprocess_binary_mask(
        candidate,
        opening_iterations=opening_iterations,
        closing_iterations=closing_iterations,
        fill_holes=fill_holes,
        min_component_voxels=min_component_voxels,
        keep_largest_component=keep_largest_component,
    )


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

    largest_component_ratio = float(largest_component_voxels) / float(max(voxel_count, 1))

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


def _brain_candidate_score(
    candidate_mask: np.ndarray,
    *,
    bone_mask_3d: np.ndarray,
    volume_hu: np.ndarray,
) -> BrainCandidate:
    head_mask = volume_hu >= -300.0
    stats = summarize_mask_quality(candidate_mask.astype(np.uint8), spacing_zyx=(1.0, 1.0, 1.0))
    overlap = np.logical_and(candidate_mask.astype(bool), bone_mask_3d.astype(bool)).sum()
    union = np.logical_or(candidate_mask.astype(bool), bone_mask_3d.astype(bool)).sum()
    overlap_ratio = float(overlap / max(union, 1))
    head_ratio = float(candidate_mask.astype(bool).sum() / max(head_mask.sum(), 1))

    volume_penalty = 0.0
    if head_ratio < 0.08:
        volume_penalty += (0.08 - head_ratio) * 4.0
    elif head_ratio > 0.60:
        volume_penalty += (head_ratio - 0.60) * 4.0

    score = (
        2.0 * float(stats["largest_component_ratio"])
        + 1.5 * float(stats["slice_coverage_ratio"])
        + float(stats["max_contiguous_run_slices"]) / max(candidate_mask.shape[0], 1)
        - 2.0 * overlap_ratio
        - volume_penalty
    )
    return BrainCandidate(
        mask_3d=candidate_mask.astype(np.uint8),
        stats=stats,
        score=float(score),
        params={},
        head_ratio=head_ratio,
        overlap_ratio=overlap_ratio,
    )


def select_adaptive_brain_mask(
    volume_hu: np.ndarray,
    *,
    bone_mask_3d: np.ndarray,
    window_center: float,
    window_width: float,
    norm_min: float,
    norm_max: float,
    head_threshold_hu: float,
    opening_iterations: int,
    closing_iterations: int,
    fill_holes: bool,
    min_component_voxels: int,
    keep_largest_component: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    thresholds = sorted({float(head_threshold_hu), -300.0, -200.0, -100.0, -50.0})
    widths = sorted({float(window_width), 80.0, 120.0, 160.0})
    norm_mins = sorted({float(norm_min), max(0.0, float(norm_min) - 0.05), 0.0})
    norm_maxs = sorted({float(norm_max), 0.98})

    candidates: list[BrainCandidate] = []
    for candidate_head_threshold in thresholds:
        for candidate_width in widths:
            for candidate_norm_min in norm_mins:
                for candidate_norm_max in norm_maxs:
                    if candidate_norm_max <= candidate_norm_min:
                        continue
                    candidate_mask = _brain_candidate_mask(
                        volume_hu,
                        bone_mask_3d=bone_mask_3d,
                        window_center=window_center,
                        window_width=candidate_width,
                        norm_min=candidate_norm_min,
                        norm_max=candidate_norm_max,
                        head_threshold_hu=candidate_head_threshold,
                        opening_iterations=opening_iterations,
                        closing_iterations=closing_iterations,
                        fill_holes=fill_holes,
                        min_component_voxels=min_component_voxels,
                        keep_largest_component=keep_largest_component,
                    )
                    candidate = _brain_candidate_score(candidate_mask, bone_mask_3d=bone_mask_3d, volume_hu=volume_hu)
                    candidates.append(
                        BrainCandidate(
                            mask_3d=candidate.mask_3d,
                            stats=candidate.stats,
                            score=candidate.score,
                            params={
                                "window_center": float(window_center),
                                "window_width": float(candidate_width),
                                "norm_min": float(candidate_norm_min),
                                "norm_max": float(candidate_norm_max),
                                "head_threshold_hu": float(candidate_head_threshold),
                            },
                            head_ratio=candidate.head_ratio,
                            overlap_ratio=candidate.overlap_ratio,
                        ),
                    )

    best = max(candidates, key=lambda item: (item.score, item.stats["voxel_count"])) if candidates else None
    if best is None or int(best.stats["voxel_count"]) == 0:
        relaxed_candidate = _brain_candidate_mask(
            volume_hu,
            bone_mask_3d=bone_mask_3d,
            window_center=window_center,
            window_width=max(float(window_width), 160.0),
            norm_min=0.0,
            norm_max=0.99,
            head_threshold_hu=min(float(head_threshold_hu), -200.0),
            opening_iterations=0,
            closing_iterations=max(1, int(closing_iterations)),
            fill_holes=True,
            min_component_voxels=max(16, int(min_component_voxels) // 2),
            keep_largest_component=True,
        )
        best = BrainCandidate(
            mask_3d=relaxed_candidate.astype(np.uint8),
            stats=summarize_mask_quality(relaxed_candidate.astype(np.uint8), spacing_zyx=(1.0, 1.0, 1.0)),
            score=0.0,
            params={
                "window_center": float(window_center),
                "window_width": max(float(window_width), 160.0),
                "norm_min": 0.0,
                "norm_max": 0.99,
                "head_threshold_hu": min(float(head_threshold_hu), -200.0),
                "fallback": True,
            },
            head_ratio=0.0,
            overlap_ratio=0.0,
        )

    diagnostics = {
        "method": "adaptive_brain_threshold_sweep",
        "selected_params": best.params,
        "selected_score": float(best.score),
        "selected_head_ratio": float(best.head_ratio),
        "selected_bone_overlap_ratio": float(best.overlap_ratio),
        "selected_stats": best.stats,
        "candidate_count": int(len(candidates)),
        "top_candidates": [
            {
                "rank": rank,
                "score": float(candidate.score),
                "head_ratio": float(candidate.head_ratio),
                "bone_overlap_ratio": float(candidate.overlap_ratio),
                "params": candidate.params,
                "stats": candidate.stats,
            }
            for rank, candidate in enumerate(
                sorted(candidates, key=lambda item: item.score, reverse=True)[:5],
                start=1,
            )
        ],
    }
    return best.mask_3d.astype(np.uint8), diagnostics


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
    bone_mask_3d: np.ndarray | None = None,
) -> np.ndarray:
    """Create a brain-ish mask using adaptive threshold selection around the brain window."""

    if bone_mask_3d is None:
        bone_mask_3d = bone_mask(
            volume_hu=volume_hu,
            threshold_hu=max(300.0, float(window_center + window_width)),
            opening_iterations=1,
            closing_iterations=1,
            min_component_voxels=64,
            keep_largest_component=True,
        )
    mask, _ = select_adaptive_brain_mask(
        volume_hu,
        bone_mask_3d=bone_mask_3d,
        window_center=window_center,
        window_width=window_width,
        norm_min=norm_min,
        norm_max=norm_max,
        head_threshold_hu=head_threshold_hu,
        opening_iterations=opening_iterations,
        closing_iterations=closing_iterations,
        fill_holes=fill_holes,
        min_component_voxels=min_component_voxels,
        keep_largest_component=keep_largest_component,
    )
    return mask


def generate_classical_masks(
    volume_hu: np.ndarray,
    segmentation_config: SegmentationConfig,
) -> SegmentationResult:
    """Run day-3 classical segmentation and return binary masks + label volume."""

    bone = bone_mask(
        volume_hu=volume_hu,
        threshold_hu=segmentation_config.bone_threshold_hu,
        opening_iterations=segmentation_config.bone_opening_iterations,
        closing_iterations=segmentation_config.bone_closing_iterations,
        min_component_voxels=segmentation_config.bone_min_component_voxels,
        keep_largest_component=segmentation_config.bone_keep_largest_component,
    ).astype(bool)

    brain_exclusion_bone = bone_mask(
        volume_hu=volume_hu,
        threshold_hu=max(300.0, float(segmentation_config.brain_window_center + segmentation_config.brain_window_width), float(segmentation_config.bone_threshold_hu)),
        opening_iterations=max(0, int(segmentation_config.bone_opening_iterations) - 1),
        closing_iterations=max(0, int(segmentation_config.bone_closing_iterations) - 1),
        min_component_voxels=max(16, int(segmentation_config.bone_min_component_voxels) // 2),
        keep_largest_component=segmentation_config.bone_keep_largest_component,
    )

    brain_uint8, brain_selection = select_adaptive_brain_mask(
        volume_hu,
        bone_mask_3d=brain_exclusion_bone.astype(np.uint8),
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
    )
    brain = brain_uint8.astype(bool)

    labels = np.zeros_like(volume_hu, dtype=np.uint8)
    labels[brain] = 1
    labels[bone] = 2
    labels[brain & bone] = 3

    return SegmentationResult(
        bone_mask_3d=bone.astype(np.uint8),
        brain_mask_3d=brain.astype(np.uint8),
        pseudo_labels_3d=labels,
        brain_selection=brain_selection,
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
