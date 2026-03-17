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
from visualization import save_day5_prediction_overlays  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day 5 full-series inference for the lightweight 2.5D U-Net.")
    parser.add_argument("--index-path", type=str, default=str(REPO_ROOT / "data_processed" / "index.csv"))
    parser.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "saved_models" / "best.pt"))
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--processed-dir", type=str, default=str(REPO_ROOT / "data_processed"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


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
    all_slice_indices = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].float()
            targets = batch["mask"].long()
            resized_images = F.interpolate(images, size=(resize_height, resize_width), mode="bilinear", align_corners=False)
            logits = model(resized_images.to(device))
            logits = F.interpolate(logits.cpu(), size=targets.shape[-2:], mode="bilinear", align_corners=False)
            predictions = logits.argmax(dim=1)

            all_predictions.append(predictions)
            all_targets.append(targets)
            all_backgrounds.append(images[:, 1])
            all_slice_indices.extend(int(value) for value in batch["slice_index"])

    prediction_volume = torch.cat(all_predictions, dim=0)
    target_volume = torch.cat(all_targets, dim=0)
    background_volume = torch.cat(all_backgrounds, dim=0)

    metrics = compute_segmentation_metrics(prediction_volume, target_volume, num_classes=4)
    prediction_np = prediction_volume.numpy().astype(np.uint8)
    target_np = target_volume.numpy().astype(np.uint8)
    background_np = background_volume.numpy().astype(np.float32)

    prediction_path = processed_dir / "day5_predictions.npz"
    np.savez_compressed(
        prediction_path,
        predicted_labels=prediction_np,
        reference_labels=target_np,
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
