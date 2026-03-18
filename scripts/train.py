from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from calibration import fit_temperature, summarize_temperature_scaling  # noqa: E402
from data.ct25d_dataset import CT25DDataset, build_default_train_transforms  # noqa: E402
from models.unet_small import UNetSmall, combined_dice_ce_loss, compute_segmentation_metrics  # noqa: E402
from visualization import save_day5_curves  # noqa: E402


@dataclass(frozen=True)
class ResizeConfig:
    height: int
    width: int


class ResizedSegmentationDataset(Dataset):
    def __init__(self, base_dataset: CT25DDataset, resize: ResizeConfig) -> None:
        self.base_dataset = base_dataset
        self.resize = resize

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.base_dataset[index]
        image = sample["image"].float().unsqueeze(0)
        mask = sample["mask"].long().unsqueeze(0).unsqueeze(0).float()

        resized_image = F.interpolate(
            image,
            size=(self.resize.height, self.resize.width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        resized_mask = F.interpolate(
            mask,
            size=(self.resize.height, self.resize.width),
            mode="nearest",
        ).squeeze(0).squeeze(0).long()

        return {
            **sample,
            "image": resized_image,
            "mask": resized_mask,
        }


def _parse_hw(raw_value: str) -> ResizeConfig:
    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected image size as 'height,width', got {raw_value!r}")
    return ResizeConfig(height=int(parts[0]), width=int(parts[1]))


def _set_seed(seed: int, num_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(num_threads))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day 5 CPU-friendly 2.5D U-Net training.")
    parser.add_argument("--index-path", type=str, default=str(REPO_ROOT / "data_processed" / "index.csv"))
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--model-dir", type=str, default=str(REPO_ROOT / "saved_models"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--image-size", type=str, default="256,256")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--rotation-deg", type=float, default=7.5)
    parser.add_argument("--disable-intensity-jitter", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def _make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        generator=generator,
    )


def _estimate_class_weights(dataset: Dataset, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for index in range(len(dataset)):
        mask = dataset[index]["mask"].reshape(-1)
        counts += torch.bincount(mask, minlength=num_classes).to(torch.float64)
    weights = counts.sum() / counts.clamp_min(1.0)
    weights = weights / weights.mean().clamp_min(1e-6)
    return weights.to(dtype=torch.float32)


def _run_epoch(
    model: UNetSmall,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    num_classes: int,
    class_weights: torch.Tensor | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    metric_batches = []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = combined_dice_ce_loss(logits, masks, num_classes=num_classes, class_weights=class_weights)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * int(images.shape[0])
        batch_metrics = compute_segmentation_metrics(logits.detach().cpu(), masks.detach().cpu(), num_classes=num_classes)
        metric_batches.append(batch_metrics)

    sample_count = max(len(loader.dataset), 1)
    mean_loss = total_loss / sample_count
    mean_dice = float(np.mean([item["dice"] for item in metric_batches])) if metric_batches else 0.0
    mean_iou = float(np.mean([item["iou"] for item in metric_batches])) if metric_batches else 0.0
    return {
        "loss": mean_loss,
        "dice": mean_dice,
        "iou": mean_iou,
    }


def _collect_logits_and_targets(model: UNetSmall, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_targets.append(masks.cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


def main() -> int:
    args = _build_parser().parse_args()
    resize = _parse_hw(args.image_size)
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    index_path = Path(args.index_path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(args.seed, args.num_threads)
    device = torch.device(args.device)

    train_transforms = build_default_train_transforms(
        rotation_degrees=args.rotation_deg,
        enable_intensity_jitter=not args.disable_intensity_jitter,
    )
    base_train = CT25DDataset(index_csv_path=index_path, split="train", transforms=train_transforms, seed=args.seed)
    base_eval = CT25DDataset(index_csv_path=index_path, split="val", transforms=None, seed=args.seed)
    eval_split = "val"
    if len(base_eval) == 0:
        base_eval = CT25DDataset(index_csv_path=index_path, split="train", transforms=None, seed=args.seed)
        eval_split = "train-fallback"

    train_dataset = ResizedSegmentationDataset(base_train, resize=resize)
    eval_dataset = ResizedSegmentationDataset(base_eval, resize=resize)
    train_loader = _make_loader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, seed=args.seed)
    eval_loader = _make_loader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, seed=args.seed)

    class_weights = _estimate_class_weights(train_dataset, num_classes=4).to(device)

    model = UNetSmall(in_channels=3, num_classes=4, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_dice": [],
        "train_iou": [],
        "eval_loss": [],
        "eval_dice": [],
        "eval_iou": [],
    }
    best_eval_dice = -1.0
    best_checkpoint_path = model_dir / "best.pt"
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, device=device, num_classes=4, class_weights=class_weights)
        eval_metrics = _run_epoch(model, eval_loader, optimizer=None, device=device, num_classes=4, class_weights=class_weights)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["train_dice"].append(train_metrics["dice"])
        history["train_iou"].append(train_metrics["iou"])
        history["eval_loss"].append(eval_metrics["loss"])
        history["eval_dice"].append(eval_metrics["dice"])
        history["eval_iou"].append(eval_metrics["iou"])

        if eval_metrics["dice"] >= best_eval_dice:
            best_eval_dice = eval_metrics["dice"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": {
                        "in_channels": 3,
                        "num_classes": 4,
                        "base_channels": args.base_channels,
                    },
                    "resize": {
                        "height": resize.height,
                        "width": resize.width,
                    },
                    "best_eval_dice": best_eval_dice,
                    "seed": args.seed,
                    "eval_split": eval_split,
                    "temperature": 1.0,
                    "class_weights": class_weights.cpu().tolist(),
                },
                best_checkpoint_path,
            )

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
            f"eval_loss={eval_metrics['loss']:.4f} eval_dice={eval_metrics['dice']:.4f}",
        )

    train_seconds = time.perf_counter() - train_start
    curves_path = output_dir / "day5_curves.png"
    save_day5_curves(history=history, output_path=curves_path)

    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["state_dict"])
    logits, targets = _collect_logits_and_targets(model, eval_loader, device=device)
    temperature = fit_temperature(logits.to(device), targets.to(device)) if logits.numel() > 0 else 1.0
    calibration_summary = summarize_temperature_scaling(logits, targets, temperature) if logits.numel() > 0 else {
        "temperature": 1.0,
        "nll_before": 0.0,
        "nll_after": 0.0,
        "ece_before": 0.0,
        "ece_after": 0.0,
    }
    best_checkpoint["temperature"] = float(temperature)
    best_checkpoint["calibration"] = calibration_summary
    torch.save(best_checkpoint, best_checkpoint_path)

    calibration_report_path = output_dir / "day5_calibration_report.json"
    calibration_report_path.write_text(json.dumps(calibration_summary, indent=2), encoding="utf-8")

    report = {
        "index_path": str(index_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "curves_path": str(curves_path),
        "calibration_report_path": str(calibration_report_path),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "resize": {"height": resize.height, "width": resize.width},
        "device": str(device),
        "eval_split": eval_split,
        "train_samples": int(len(train_dataset)),
        "eval_samples": int(len(eval_dataset)),
        "best_eval_dice": float(best_eval_dice),
        "class_weights": [float(value) for value in class_weights.cpu().tolist()],
        "history": history,
        "temperature": float(temperature),
        "calibration": calibration_summary,
        "train_seconds": float(train_seconds),
        "notes": [
            "Pseudo labels from Day 3 are used as supervision targets.",
            "When only one series is available, Day 4 now creates buffered intra-series holdout slices for evaluation.",
        ],
    }
    report_path = output_dir / "day5_train_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved best checkpoint: {best_checkpoint_path}")
    print(f"Saved curves: {curves_path}")
    print(f"Saved calibration report: {calibration_report_path}")
    print(f"Saved training report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
