from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetSmall(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        self.encoder1 = DoubleConv(in_channels, base_channels)
        self.encoder2 = DownBlock(base_channels, base_channels * 2)
        self.encoder3 = DownBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = DownBlock(base_channels * 4, base_channels * 8)
        self.decoder3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.decoder2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.decoder1 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        bottleneck = self.bottleneck(enc3)
        dec3 = self.decoder3(bottleneck, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)
        return self.head(dec1)


@dataclass(frozen=True)
class MetricResult:
    dice: float
    iou: float


def one_hot_labels(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    if targets.ndim != 3:
        raise ValueError(f"Expected targets with shape (N, H, W), got {tuple(targets.shape)}")
    encoded = F.one_hot(targets.long(), num_classes=num_classes)
    return encoded.permute(0, 3, 1, 2).float()


def multiclass_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    include_background: bool = False,
    smooth: float = 1e-5,
) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    targets_one_hot = one_hot_labels(targets, num_classes=num_classes).to(probabilities.device)

    if not include_background:
        probabilities = probabilities[:, 1:]
        targets_one_hot = targets_one_hot[:, 1:]

    dims = (0, 2, 3)
    intersection = torch.sum(probabilities * targets_one_hot, dim=dims)
    denominator = torch.sum(probabilities + targets_one_hot, dim=dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def combined_dice_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ce_weight: float = 0.5,
    dice_weight: float = 0.5,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets.long(), weight=class_weights)
    dice = multiclass_dice_loss(logits, targets, num_classes=num_classes, include_background=False)
    return ce_weight * ce + dice_weight * dice


def compute_segmentation_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    include_background: bool = False,
    smooth: float = 1e-5,
) -> dict[str, float | dict[str, float]]:
    if predictions.ndim == 4:
        pred_labels = predictions.argmax(dim=1)
    else:
        pred_labels = predictions.long()
    target_labels = targets.long()

    per_class_dice: dict[str, float] = {}
    per_class_iou: dict[str, float] = {}
    class_indices = range(num_classes) if include_background else range(1, num_classes)

    dice_values = []
    iou_values = []
    for class_index in class_indices:
        pred_mask = pred_labels == class_index
        target_mask = target_labels == class_index
        intersection = torch.logical_and(pred_mask, target_mask).sum(dtype=torch.float32)
        union = torch.logical_or(pred_mask, target_mask).sum(dtype=torch.float32)
        pred_sum = pred_mask.sum(dtype=torch.float32)
        target_sum = target_mask.sum(dtype=torch.float32)

        dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
        iou = (intersection + smooth) / (union + smooth)

        per_class_dice[str(class_index)] = float(dice.item())
        per_class_iou[str(class_index)] = float(iou.item())
        dice_values.append(dice)
        iou_values.append(iou)

    mean_dice = torch.stack(dice_values).mean() if dice_values else torch.tensor(1.0)
    mean_iou = torch.stack(iou_values).mean() if iou_values else torch.tensor(1.0)
    return {
        "dice": float(mean_dice.item()),
        "iou": float(mean_iou.item()),
        "per_class_dice": per_class_dice,
        "per_class_iou": per_class_iou,
    }
