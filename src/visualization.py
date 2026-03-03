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
