from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from preprocessing import BoundingBox3D

BRAIN_WINDOW = (40.0, 80.0)
BONE_WINDOW = (600.0, 2000.0)


def apply_window(slice_hu: np.ndarray, center: float, width: float) -> np.ndarray:
    lower = center - width / 2.0
    upper = center + width / 2.0
    clipped = np.clip(slice_hu, lower, upper)
    return ((clipped - lower) / (upper - lower)).astype(np.float32)


def save_hu_histogram(volume_hu: np.ndarray, output_path: str | Path) -> None:
    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    flat = volume_hu.astype(np.float32).ravel()
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(flat, bins=256, range=(-1200, 3000), color="#2A9D8F", alpha=0.9)
    axis.set_title("HU Histogram")
    axis.set_xlabel("Hounsfield Units (HU)")
    axis.set_ylabel("Voxel Count")
    axis.axvline(-1000, color="#264653", linestyle="--", linewidth=1, label="Air ~ -1000")
    axis.axvline(40, color="#E76F51", linestyle="--", linewidth=1, label="Brain WL center ~ 40")
    axis.axvline(600, color="#F4A261", linestyle="--", linewidth=1, label="Bone WL center ~ 600")
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_file, dpi=160)
    plt.close(fig)


def save_montage(volume_hu: np.ndarray, output_path: str | Path, slices_per_row: int = 8) -> None:
    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    slice_count = volume_hu.shape[0]
    selected_indices = np.linspace(0, slice_count - 1, num=min(slice_count, slices_per_row), dtype=int)

    fig, axes = plt.subplots(2, len(selected_indices), figsize=(2.1 * len(selected_indices), 5))
    if len(selected_indices) == 1:
        axes = np.asarray([[axes[0]], [axes[1]]])

    for col, idx in enumerate(selected_indices):
        brain = apply_window(volume_hu[idx], *BRAIN_WINDOW)
        bone = apply_window(volume_hu[idx], *BONE_WINDOW)

        axes[0, col].imshow(brain, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, col].imshow(bone, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, col].set_title(f"z={idx}", fontsize=8)
        axes[0, col].axis("off")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Brain\nW=80 L=40")
    axes[1, 0].set_ylabel("Bone\nW=2000 L=600")
    fig.suptitle("Axial Montage by Window/Level", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_file, dpi=160)
    plt.close(fig)


class AxialScrollViewer:
    def __init__(self, volume_hu: np.ndarray) -> None:
        self.volume_hu = volume_hu
        self.slice_count = int(volume_hu.shape[0])
        self.index = self.slice_count // 2

        self.fig, (self.ax_brain, self.ax_bone) = plt.subplots(1, 2, figsize=(10, 5))
        self.im_brain = self.ax_brain.imshow(
            apply_window(self.volume_hu[self.index], *BRAIN_WINDOW),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        self.im_bone = self.ax_bone.imshow(
            apply_window(self.volume_hu[self.index], *BONE_WINDOW),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        self.ax_brain.set_title("Brain (W=80, L=40)")
        self.ax_bone.set_title("Bone (W=2000, L=600)")
        self.ax_brain.axis("off")
        self.ax_bone.axis("off")

        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self._draw()

    def _step(self, delta: int) -> None:
        self.index = max(0, min(self.slice_count - 1, self.index + delta))
        self._draw()

    def _on_scroll(self, event: object) -> None:
        button = getattr(event, "button", "")
        if button == "up":
            self._step(1)
        elif button == "down":
            self._step(-1)

    def _on_key_press(self, event: object) -> None:
        key = getattr(event, "key", "")
        if key in {"right", "up", "d"}:
            self._step(1)
        elif key in {"left", "down", "a"}:
            self._step(-1)

    def _draw(self) -> None:
        current = self.volume_hu[self.index]
        self.im_brain.set_data(apply_window(current, *BRAIN_WINDOW))
        self.im_bone.set_data(apply_window(current, *BONE_WINDOW))
        self.fig.suptitle(
            f"Axial slice {self.index + 1}/{self.slice_count} (scroll mouse wheel or arrow keys)",
            fontsize=11,
        )
        self.fig.canvas.draw_idle()


def show_axial_scroll(volume_hu: np.ndarray) -> AxialScrollViewer:
    viewer = AxialScrollViewer(volume_hu)
    plt.show()
    return viewer


def save_day2_before_after(
    resampled_volume_hu: np.ndarray,
    cropped_volume_hu: np.ndarray,
    processed_volume: np.ndarray,
    crop_bbox_zyx: BoundingBox3D,
    output_path: str | Path,
) -> None:
    """Save visual QA image for Day 2 preprocessing.

    The figure shows:
    - pre-crop brain/bone windows on one slice,
    - crop box overlay to verify anatomy preservation,
    - post-crop + normalized result for comparison.
    """

    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Slice index inside the full resampled volume.
    z_idx = max(crop_bbox_zyx.z_min, min(crop_bbox_zyx.z_max - 1, resampled_volume_hu.shape[0] // 2))

    # Matching slice index in the cropped volume coordinates.
    cropped_z_idx = int(np.clip(z_idx - crop_bbox_zyx.z_min, 0, cropped_volume_hu.shape[0] - 1))

    before_brain = apply_window(resampled_volume_hu[z_idx], *BRAIN_WINDOW)
    before_bone = apply_window(resampled_volume_hu[z_idx], *BONE_WINDOW)
    after_brain = apply_window(cropped_volume_hu[cropped_z_idx], *BRAIN_WINDOW)
    after_norm = processed_volume[cropped_z_idx]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
    axes[0].imshow(before_brain, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Before crop\nBrain W=80 L=40")
    axes[0].axis("off")

    axes[1].imshow(before_bone, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("Before crop + ROI box\nBone W=2000 L=600")
    rectangle = Rectangle(
        (crop_bbox_zyx.x_min, crop_bbox_zyx.y_min),
        crop_bbox_zyx.x_max - crop_bbox_zyx.x_min,
        crop_bbox_zyx.y_max - crop_bbox_zyx.y_min,
        fill=False,
        linewidth=2.0,
        edgecolor="#FF6B35",
    )
    axes[1].add_patch(rectangle)
    axes[1].axis("off")

    axes[2].imshow(after_brain, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title("After crop\nBrain W=80 L=40")
    axes[2].axis("off")

    axes[3].imshow(after_norm, cmap="gray", vmin=0.0, vmax=1.0)
    axes[3].set_title("After clip+normalize\n[0, 1]")
    axes[3].axis("off")

    fig.suptitle(f"Day 2 Preprocess QA (z={z_idx}, cropped-z={cropped_z_idx})", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def save_day3_slice_overlays(
    volume_hu: np.ndarray,
    brain_mask_3d: np.ndarray,
    bone_mask_3d: np.ndarray,
    output_dir: str | Path,
    max_slices: int | None = None,
    min_representative_slices: int = 10,
) -> list[Path]:
    """Save per-slice overlay PNGs for qualitative review.

    Parameters to tune:
    - `max_slices`: optional cap on number of slices exported for very large studies.
    - `min_representative_slices`: ensures enough coverage when a cap is active.
    """

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    total_slices = int(volume_hu.shape[0])
    if max_slices is None or max_slices >= total_slices:
        slice_indices = np.arange(total_slices, dtype=int)
    else:
        requested = max(int(min_representative_slices), int(max_slices))
        requested = min(requested, total_slices)
        slice_indices = np.linspace(0, total_slices - 1, num=requested, dtype=int)

    saved_files: list[Path] = []
    for z_idx in slice_indices:
        background = apply_window(volume_hu[z_idx], center=BRAIN_WINDOW[0], width=BRAIN_WINDOW[1])
        brain_slice = brain_mask_3d[z_idx].astype(bool)
        bone_slice = bone_mask_3d[z_idx].astype(bool)

        fig, axis = plt.subplots(1, 1, figsize=(5.2, 5.2))
        axis.imshow(background, cmap="gray", vmin=0.0, vmax=1.0)
        if np.any(brain_slice):
            axis.contour(brain_slice, levels=[0.5], colors=["#00FF85"], linewidths=1.1)
        if np.any(bone_slice):
            axis.contour(bone_slice, levels=[0.5], colors=["#FF5C5C"], linewidths=1.0)

        axis.set_title(f"z={z_idx} | green=brain-ish, red=bone", fontsize=9)
        axis.axis("off")
        fig.tight_layout()

        file_path = output_path / f"slice_{z_idx:03d}.png"
        fig.savefig(file_path, dpi=150)
        plt.close(fig)
        saved_files.append(file_path)

    return saved_files


def _to_numpy(array_like: object) -> np.ndarray:
    if hasattr(array_like, "detach"):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def save_day4_batch_viz(
    batch: dict[str, object],
    output_path: str | Path,
    title: str = "Day 4 2.5D Batch Sanity Check",
    max_items: int = 4,
) -> None:
    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    images = _to_numpy(batch["image"]).astype(np.float32)
    masks = _to_numpy(batch["mask"]).astype(np.int64)
    slice_indices = list(_to_numpy(batch["slice_index"]).tolist())

    if images.ndim != 4:
        raise ValueError(f"Expected batch['image'] to have shape (B, 3, H, W), got {images.shape}.")
    if masks.ndim != 3:
        raise ValueError(f"Expected batch['mask'] to have shape (B, H, W), got {masks.shape}.")

    sample_count = min(int(images.shape[0]), int(max_items))
    fig, axes = plt.subplots(sample_count, 4, figsize=(12, 3.2 * sample_count))
    if sample_count == 1:
        axes = np.asarray([axes])

    channel_titles = ("z-1", "z", "z+1")
    for row_index in range(sample_count):
        for channel_index, channel_title in enumerate(channel_titles):
            axis = axes[row_index, channel_index]
            axis.imshow(images[row_index, channel_index], cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_title(channel_title, fontsize=10)
            axis.axis("off")

        mask_axis = axes[row_index, 3]
        mask_axis.imshow(images[row_index, 1], cmap="gray", vmin=0.0, vmax=1.0)
        mask_axis.imshow(
            np.ma.masked_where(masks[row_index] <= 0, masks[row_index]),
            cmap="viridis",
            alpha=0.55,
            interpolation="nearest",
        )
        mask_axis.set_title(f"mask @ z={slice_indices[row_index]}", fontsize=10)
        mask_axis.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def save_day5_curves(
    history: dict[str, list[float] | list[int]],
    output_path: str | Path,
) -> None:
    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.asarray(history["epoch"], dtype=np.int32)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(epochs, history["train_loss"], label="train", color="#264653", linewidth=2)
    axes[0].plot(epochs, history["eval_loss"], label="eval", color="#E76F51", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, history["train_dice"], label="train Dice", color="#2A9D8F", linewidth=2)
    axes[1].plot(epochs, history["eval_dice"], label="eval Dice", color="#E9C46A", linewidth=2)
    axes[1].plot(epochs, history["eval_iou"], label="eval IoU", color="#F4A261", linewidth=2)
    axes[1].set_title("Segmentation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.suptitle("Day 5 Lightweight U-Net Training", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def save_day5_prediction_overlays(
    center_slices: np.ndarray,
    predicted_labels: np.ndarray,
    reference_labels: np.ndarray,
    output_dir: str | Path,
    slice_indices: list[int] | None = None,
) -> list[Path]:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if slice_indices is None:
        slice_indices = list(range(int(center_slices.shape[0])))

    saved_files: list[Path] = []
    for row_index, slice_index in enumerate(slice_indices):
        background = np.clip(center_slices[row_index], 0.0, 1.0)
        predicted = predicted_labels[row_index]
        reference = reference_labels[row_index]

        fig, axes = plt.subplots(1, 3, figsize=(11.5, 4))
        axes[0].imshow(background, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0].set_title(f"Input z={slice_index}")
        axes[0].axis("off")

        axes[1].imshow(background, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1].imshow(np.ma.masked_where(predicted <= 0, predicted), cmap="viridis", alpha=0.55)
        axes[1].set_title("DL prediction")
        axes[1].axis("off")

        axes[2].imshow(background, cmap="gray", vmin=0.0, vmax=1.0)
        axes[2].imshow(np.ma.masked_where(reference <= 0, reference), cmap="magma", alpha=0.45)
        if np.any(predicted > 0):
            axes[2].contour(predicted > 0, levels=[0.5], colors=["#00FF85"], linewidths=1.0)
        axes[2].set_title("Classical pseudo label + DL contour")
        axes[2].axis("off")

        fig.tight_layout()
        file_path = output_path / f"slice_{slice_index:03d}.png"
        fig.savefig(file_path, dpi=150)
        plt.close(fig)
        saved_files.append(file_path)

    return saved_files


def save_report_slice_montage(
    center_slices: np.ndarray,
    predicted_labels: np.ndarray,
    reference_labels: np.ndarray,
    uncertainty: np.ndarray | None,
    slice_indices: list[int],
    output_path: str | Path,
    *,
    title: str = "Evaluation Montage",
) -> Path:
    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    row_count = max(len(slice_indices), 1)
    column_count = 4 if uncertainty is not None else 3
    fig, axes = plt.subplots(row_count, column_count, figsize=(3.8 * column_count, 3.1 * row_count))
    if row_count == 1:
        axes = np.asarray([axes])

    for row_index, slice_index in enumerate(slice_indices):
        background = np.clip(center_slices[row_index], 0.0, 1.0)
        predicted = predicted_labels[row_index]
        reference = reference_labels[row_index]

        axes[row_index, 0].imshow(background, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_index, 0].set_title(f"Input z={slice_index}")
        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(background, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_index, 1].imshow(np.ma.masked_where(predicted <= 0, predicted), cmap="viridis", alpha=0.55)
        axes[row_index, 1].set_title("Prediction")
        axes[row_index, 1].axis("off")

        axes[row_index, 2].imshow(background, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row_index, 2].imshow(np.ma.masked_where(reference <= 0, reference), cmap="magma", alpha=0.45)
        if np.any(predicted > 0):
            axes[row_index, 2].contour(predicted > 0, levels=[0.5], colors=["#00FF85"], linewidths=0.9)
        axes[row_index, 2].set_title("Reference + contour")
        axes[row_index, 2].axis("off")

        if uncertainty is not None:
            axes[row_index, 3].imshow(uncertainty[row_index], cmap="inferno", vmin=0.0, vmax=1.0)
            axes[row_index, 3].set_title("Uncertainty")
            axes[row_index, 3].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_file, dpi=170)
    plt.close(fig)
    return output_file
