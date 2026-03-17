from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class LabelPostprocessConfig:
    min_component_size: int = 64
    fill_holes: bool = True
    smooth_iterations: int = 1
    keep_largest_component: bool = False


def validate_spacing_zyx(
    spacing_zyx: tuple[float, float, float],
    *,
    min_spacing_mm: float = 0.05,
    max_spacing_mm: float = 10.0,
) -> list[str]:
    messages: list[str] = []
    for axis_name, spacing in zip(("z", "y", "x"), spacing_zyx, strict=True):
        if not np.isfinite(spacing):
            raise ValueError(f"Invalid {axis_name}-spacing: expected a finite value, got {spacing!r}.")
        if spacing <= 0.0:
            raise ValueError(f"Invalid {axis_name}-spacing: expected a positive value, got {spacing!r}.")
        if spacing < min_spacing_mm:
            messages.append(
                f"{axis_name}-spacing={spacing:.4f} mm is unusually small; verify DICOM spacing tags before training.",
            )
        elif spacing > max_spacing_mm:
            messages.append(
                f"{axis_name}-spacing={spacing:.4f} mm is unusually large; thick slices may hurt 3D continuity.",
            )
    return messages


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    return ndimage.label(mask.astype(bool), structure=structure)


def _remove_small_components(mask: np.ndarray, min_component_size: int) -> np.ndarray:
    if min_component_size <= 0:
        return mask.astype(bool, copy=False)

    labels, count = _connected_components(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)

    component_ids = np.arange(1, count + 1, dtype=np.int32)
    component_sizes = ndimage.sum(mask, labels=labels, index=component_ids)
    keep_ids = component_ids[np.asarray(component_sizes) >= int(min_component_size)]
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


def postprocess_binary_mask(
    mask: np.ndarray,
    *,
    min_component_size: int = 64,
    fill_holes: bool = True,
    smooth_iterations: int = 1,
    keep_largest_component: bool = False,
) -> np.ndarray:
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    cleaned = mask.astype(bool, copy=False)

    cleaned = _remove_small_components(cleaned, min_component_size=min_component_size)
    if keep_largest_component:
        cleaned = _largest_component(cleaned)
    if fill_holes:
        cleaned = ndimage.binary_fill_holes(cleaned)
    if smooth_iterations > 0:
        cleaned = ndimage.binary_opening(cleaned, structure=structure, iterations=int(smooth_iterations))
        cleaned = ndimage.binary_closing(cleaned, structure=structure, iterations=int(smooth_iterations))
    return cleaned.astype(bool, copy=False)


def postprocess_multiclass_prediction(
    probabilities: np.ndarray,
    predicted_labels: np.ndarray,
    *,
    class_configs: dict[int, LabelPostprocessConfig] | None = None,
) -> np.ndarray:
    if probabilities.ndim != 4:
        raise ValueError(f"Expected probabilities with shape (C, Z, Y, X), got {probabilities.shape}.")
    if predicted_labels.shape != probabilities.shape[1:]:
        raise ValueError(
            "Predicted labels and probabilities must align: "
            f"got labels {predicted_labels.shape} vs probs {probabilities.shape[1:]}.",
        )

    class_configs = class_configs or {}
    num_classes = int(probabilities.shape[0])
    candidate_masks = np.zeros_like(probabilities, dtype=bool)
    for class_index in range(1, num_classes):
        config = class_configs.get(class_index, LabelPostprocessConfig())
        candidate_masks[class_index] = postprocess_binary_mask(
            predicted_labels == class_index,
            min_component_size=config.min_component_size,
            fill_holes=config.fill_holes,
            smooth_iterations=config.smooth_iterations,
            keep_largest_component=config.keep_largest_component,
        )

    masked_scores = probabilities.copy()
    masked_scores[0] = 0.0
    for class_index in range(1, num_classes):
        masked_scores[class_index, ~candidate_masks[class_index]] = -1.0

    refined = masked_scores.argmax(axis=0).astype(np.uint8)
    refined[~candidate_masks[1:].any(axis=0)] = 0
    return refined


def compute_entropy_uncertainty(probabilities: np.ndarray, *, epsilon: float = 1e-8) -> np.ndarray:
    if probabilities.ndim < 2:
        raise ValueError("Expected probabilities with at least 2 dimensions and class axis first.")
    clipped = np.clip(probabilities.astype(np.float32, copy=False), epsilon, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=0)
    max_entropy = float(np.log(max(int(probabilities.shape[0]), 2)))
    return (entropy / max_entropy).astype(np.float32, copy=False)


def summarize_uncertainty(
    uncertainty_map: np.ndarray,
    *,
    foreground_mask: np.ndarray | None = None,
) -> dict[str, object]:
    if foreground_mask is not None and foreground_mask.shape != uncertainty_map.shape:
        raise ValueError("Foreground mask must match uncertainty map shape.")

    values = uncertainty_map.astype(np.float32, copy=False)
    if foreground_mask is not None and np.any(foreground_mask):
        selected = values[foreground_mask.astype(bool)]
    else:
        selected = values.reshape(-1)

    per_slice_mean = values.mean(axis=(1, 2))
    return {
        "mean": float(selected.mean()) if selected.size else 0.0,
        "max": float(selected.max()) if selected.size else 0.0,
        "p95": float(np.percentile(selected, 95)) if selected.size else 0.0,
        "per_slice_mean": [float(value) for value in per_slice_mean.tolist()],
    }
