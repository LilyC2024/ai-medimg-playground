from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.ct25d_dataset import CT25DDataset  # noqa: E402
from models.unet_small import UNetSmall, compute_segmentation_metrics  # noqa: E402
from robustness import (  # noqa: E402
    LabelPostprocessConfig,
    compute_entropy_uncertainty,
    postprocess_multiclass_prediction,
    summarize_uncertainty,
)
from visualization import save_day5_prediction_overlays  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day 5 full-series inference for the lightweight 2.5D U-Net.")
    parser.add_argument("--index-path", type=str, default=str(REPO_ROOT / "data_processed" / "index.csv"))
    parser.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "saved_models" / "best.pt"))
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--processed-dir", type=str, default=str(REPO_ROOT / "data_processed"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--uncertainty-method", choices=["entropy", "tta", "none"], default="entropy")
    parser.add_argument("--disable-postprocess", action="store_true")
    parser.add_argument("--brain-min-voxels", type=int, default=256)
    parser.add_argument("--bone-min-voxels", type=int, default=96)
    parser.add_argument("--overlap-min-voxels", type=int, default=32)
    parser.add_argument("--smooth-iters", type=int, default=1)
    return parser


def _predict_probabilities(
    model: UNetSmall,
    images: torch.Tensor,
    device: torch.device,
    uncertainty_method: str,
) -> torch.Tensor:
    resized_images = images.to(device)
    logits = model(resized_images)
    probabilities = torch.softmax(logits, dim=1)

    if uncertainty_method != "tta":
        return probabilities

    augmentations = (
        (),
        (-1,),
        (-2,),
    )
    probability_sum = probabilities
    for dims in augmentations[1:]:
        flipped_images = torch.flip(resized_images, dims=dims)
        flipped_logits = model(flipped_images)
        probability_sum = probability_sum + torch.flip(torch.softmax(flipped_logits, dim=1), dims=dims)
    return probability_sum / float(len(augmentations))


def _slice_metrics(
    prediction_volume: np.ndarray,
    target_volume: np.ndarray,
    slice_indices: list[int],
) -> list[dict[str, float | int]]:
    metrics: list[dict[str, float | int]] = []
    for row_index, slice_index in enumerate(slice_indices):
        slice_result = compute_segmentation_metrics(
            torch.from_numpy(prediction_volume[row_index : row_index + 1]),
            torch.from_numpy(target_volume[row_index : row_index + 1]),
            num_classes=4,
        )
        metrics.append(
            {
                "slice_index": int(slice_index),
                "dice": float(slice_result["dice"]),
                "iou": float(slice_result["iou"]),
            },
        )
    return metrics


def main() -> int:
    args = _build_parser().parse_args()
    index_path = Path(args.index_path).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    processed_dir = Path(args.processed_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = UNetSmall(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    device = torch.device(args.device)
    model.to(device)

    resize_height = int(checkpoint["resize"]["height"])
    resize_width = int(checkpoint["resize"]["width"])

    dataset = CT25DDataset(index_csv_path=index_path, split=None, transforms=None, seed=int(checkpoint.get("seed", 13)))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)

    all_predictions = []
    all_targets = []
    all_backgrounds = []
    all_uncertainty = []
    all_slice_indices = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].float()
            targets = batch["mask"].long()
            resized_images = F.interpolate(images, size=(resize_height, resize_width), mode="bilinear", align_corners=False)
            probabilities = _predict_probabilities(model, resized_images, device=device, uncertainty_method=args.uncertainty_method)
            probabilities = F.interpolate(
                probabilities.cpu(),
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-6)
            predictions = probabilities.argmax(dim=1)

            if not args.disable_postprocess:
                refined_labels = postprocess_multiclass_prediction(
                    probabilities=probabilities.permute(1, 0, 2, 3).numpy(),
                    predicted_labels=predictions.numpy(),
                    class_configs={
                        1: LabelPostprocessConfig(
                            min_component_size=args.brain_min_voxels,
                            fill_holes=True,
                            smooth_iterations=args.smooth_iters,
                            keep_largest_component=True,
                        ),
                        2: LabelPostprocessConfig(
                            min_component_size=args.bone_min_voxels,
                            fill_holes=False,
                            smooth_iterations=args.smooth_iters,
                            keep_largest_component=True,
                        ),
                        3: LabelPostprocessConfig(
                            min_component_size=args.overlap_min_voxels,
                            fill_holes=True,
                            smooth_iterations=args.smooth_iters,
                            keep_largest_component=False,
                        ),
                    },
                )
                predictions = torch.from_numpy(refined_labels.astype(np.int64))

            uncertainty = compute_entropy_uncertainty(probabilities.permute(1, 0, 2, 3).numpy())

            all_predictions.append(predictions)
            all_targets.append(targets)
            all_backgrounds.append(images[:, 1])
            all_uncertainty.append(torch.from_numpy(uncertainty))
            all_slice_indices.extend(int(value) for value in batch["slice_index"])

    prediction_volume = torch.cat(all_predictions, dim=0)
    target_volume = torch.cat(all_targets, dim=0)
    background_volume = torch.cat(all_backgrounds, dim=0)
    uncertainty_volume = torch.cat(all_uncertainty, dim=0)

    metrics = compute_segmentation_metrics(prediction_volume, target_volume, num_classes=4)
    prediction_np = prediction_volume.numpy().astype(np.uint8)
    target_np = target_volume.numpy().astype(np.uint8)
    background_np = background_volume.numpy().astype(np.float32)
    uncertainty_np = uncertainty_volume.numpy().astype(np.float32)
    uncertainty_summary = summarize_uncertainty(uncertainty_np, foreground_mask=prediction_np > 0)
    per_slice_metrics = _slice_metrics(prediction_np, target_np, all_slice_indices)
    best_slices = sorted(per_slice_metrics, key=lambda item: (-float(item["dice"]), int(item["slice_index"])))[:3]
    worst_slices = sorted(per_slice_metrics, key=lambda item: (float(item["dice"]), int(item["slice_index"])))[:3]

    prediction_path = processed_dir / "day5_predictions.npz"
    np.savez_compressed(
        prediction_path,
        predicted_labels=prediction_np,
        reference_labels=target_np,
        center_slices=background_np,
        uncertainty=uncertainty_np,
        slice_indices=np.asarray(all_slice_indices, dtype=np.int32),
    )

    overlay_dir = output_dir / "day5_overlays"
    saved_overlay_files = save_day5_prediction_overlays(
        center_slices=background_np,
        predicted_labels=prediction_np,
        reference_labels=target_np,
        output_dir=overlay_dir,
        slice_indices=all_slice_indices,
    )

    report = {
        "checkpoint": str(checkpoint_path),
        "prediction_volume_path": str(prediction_path),
        "overlay_dir": str(overlay_dir),
        "overlay_count": int(len(saved_overlay_files)),
        "slice_count": int(prediction_np.shape[0]),
        "metrics": metrics,
        "per_slice_metrics": per_slice_metrics,
        "best_slices": best_slices,
        "worst_slices": worst_slices,
        "uncertainty": {
            "method": args.uncertainty_method,
            "summary": uncertainty_summary,
        },
        "postprocessing": {
            "enabled": bool(not args.disable_postprocess),
            "brain_min_voxels": int(args.brain_min_voxels),
            "bone_min_voxels": int(args.bone_min_voxels),
            "overlap_min_voxels": int(args.overlap_min_voxels),
            "smooth_iterations": int(args.smooth_iters),
        },
        "best_eval_dice_from_checkpoint": float(checkpoint.get("best_eval_dice", 0.0)),
        "reference_note": "Metrics are computed against Day 3 pseudo labels because no manual GT labels are included.",
    }
    report_path = output_dir / "day5_infer_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved predictions: {prediction_path}")
    print(f"Saved overlays: {overlay_dir}")
    print(f"Saved inference report: {report_path}")
    print(f"Dice={metrics['dice']:.4f} IoU={metrics['iou']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
