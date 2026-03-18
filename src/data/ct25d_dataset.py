from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
from scipy import ndimage

from preprocessing import load_processed_volume

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - torch is optional at import time
    torch = None
    DataLoader = Any  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class CT25DCase:
    patient_id: str
    study_instance_uid: str
    series_instance_uid: str
    series_dir: Path
    volume_path: Path
    label_volume_path: Path
    slice_labels_dir: Path | None
    spacing_zyx: tuple[float, float, float]
    volume_shape_zyx: tuple[int, int, int]
    split_group_id: str


def clamp_stack_indices(center_index: int, depth: int) -> tuple[int, int, int]:
    if depth <= 0:
        raise ValueError("depth must be positive for 2.5D stacking.")
    if center_index < 0 or center_index >= depth:
        raise IndexError(f"center_index {center_index} is out of bounds for depth {depth}.")

    prev_index = max(0, center_index - 1)
    next_index = min(depth - 1, center_index + 1)
    return (prev_index, center_index, next_index)


def build_25d_stack(volume: np.ndarray, center_index: int) -> np.ndarray:
    prev_index, center_index, next_index = clamp_stack_indices(center_index, int(volume.shape[0]))
    stack = np.stack(
        [
            volume[prev_index],
            volume[center_index],
            volume[next_index],
        ],
        axis=0,
    )
    return stack.astype(np.float32, copy=False)


def _stable_identifier(raw_value: object, fallback: str) -> str:
    text = str(raw_value).strip()
    return text if text else fallback


def _read_series_identifiers(series_dir: str | Path) -> dict[str, str]:
    series_path = Path(series_dir).expanduser().resolve()
    files = sorted(path for path in series_path.iterdir() if path.is_file())
    if not files:
        raise FileNotFoundError(f"No DICOM files found in series directory: {series_path}")

    dataset = pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True)
    patient_id = _stable_identifier(dataset.get("PatientID", ""), fallback=series_path.name)
    study_uid = _stable_identifier(dataset.get("StudyInstanceUID", ""), fallback=f"{patient_id}-study")
    series_uid = _stable_identifier(dataset.get("SeriesInstanceUID", ""), fallback=f"{patient_id}-series")
    return {
        "patient_id": patient_id,
        "study_instance_uid": study_uid,
        "series_instance_uid": series_uid,
    }


def _resolve_volume_path(processed_dir: Path) -> Path:
    for candidate_name in ("volume.nii.gz", "volume.nii", "volume.npz"):
        candidate = processed_dir / candidate_name
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find processed volume in {processed_dir}. Expected volume.nii.gz, volume.nii, or volume.npz.",
    )


def discover_legacy_case(
    series_dir: str | Path,
    processed_dir: str | Path,
    label_volume_name: str = "pseudo_labels_3d.npz",
) -> CT25DCase:
    series_path = Path(series_dir).expanduser().resolve()
    processed_path = Path(processed_dir).expanduser().resolve()
    volume_path = _resolve_volume_path(processed_path)
    label_volume_path = (processed_path / "pseudo_labels" / label_volume_name).resolve()

    if not label_volume_path.exists():
        raise FileNotFoundError(f"Pseudo-label volume not found: {label_volume_path}")

    volume, spacing_zyx = load_processed_volume(volume_path)
    with np.load(label_volume_path) as data:
        pseudo_labels = data["pseudo_labels"]

    if tuple(int(v) for v in volume.shape) != tuple(int(v) for v in pseudo_labels.shape):
        raise ValueError(
            "Processed volume and pseudo-label volume shape mismatch: "
            f"{tuple(volume.shape)} vs {tuple(pseudo_labels.shape)}",
        )

    identifiers = _read_series_identifiers(series_path)
    split_group_id = f"{identifiers['patient_id']}|{identifiers['series_instance_uid']}"
    return CT25DCase(
        patient_id=identifiers["patient_id"],
        study_instance_uid=identifiers["study_instance_uid"],
        series_instance_uid=identifiers["series_instance_uid"],
        series_dir=series_path,
        volume_path=volume_path,
        label_volume_path=label_volume_path,
        slice_labels_dir=(processed_path / "pseudo_labels" / "slices").resolve(),
        spacing_zyx=tuple(float(value) for value in spacing_zyx),
        volume_shape_zyx=tuple(int(value) for value in volume.shape),
        split_group_id=split_group_id,
    )


def assign_group_splits(
    group_ids: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 13,
) -> dict[str, str]:
    unique_groups = list(dict.fromkeys(group_ids))
    if not unique_groups:
        return {}

    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios < 0):
        raise ValueError("Split ratios must be non-negative.")
    if float(ratios.sum()) <= 0:
        raise ValueError("At least one split ratio must be positive.")

    rng = np.random.default_rng(int(seed))
    rng.shuffle(unique_groups)

    if len(unique_groups) == 1:
        return {unique_groups[0]: "train"}
    if len(unique_groups) == 2:
        secondary_split = "val" if val_ratio >= test_ratio else "test"
        return {unique_groups[0]: "train", unique_groups[1]: secondary_split}

    ratios = ratios / ratios.sum()
    desired = ratios * len(unique_groups)
    counts = np.floor(desired).astype(int)
    counts[0] = max(counts[0], 1)

    target_minimums = np.asarray([1, 1 if val_ratio > 0 else 0, 1 if test_ratio > 0 else 0], dtype=int)
    counts = np.maximum(counts, target_minimums)

    while int(counts.sum()) < len(unique_groups):
        next_index = int(np.argmax(desired - counts))
        counts[next_index] += 1

    while int(counts.sum()) > len(unique_groups):
        removable = [
            idx for idx, count in enumerate(counts.tolist())
            if count > target_minimums[idx]
        ]
        if not removable:
            break
        remove_index = max(removable, key=lambda idx: counts[idx] - desired[idx])
        counts[remove_index] -= 1

    split_names = ("train", "val", "test")
    assignments: dict[str, str] = {}
    cursor = 0
    for split_name, split_count in zip(split_names, counts.tolist(), strict=True):
        for group_id in unique_groups[cursor:cursor + split_count]:
            assignments[group_id] = split_name
        cursor += split_count

    return assignments


def assign_single_case_slice_splits(
    depth: int,
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    context_radius: int = 1,
) -> list[str]:
    if depth <= 0:
        raise ValueError("depth must be positive.")

    assignments = ["train"] * int(depth)
    requested_val = int(round(depth * max(val_ratio, 0.0))) if val_ratio > 0 else 0
    requested_test = int(round(depth * max(test_ratio, 0.0))) if test_ratio > 0 else 0
    val_count = max(requested_val, 1 if val_ratio > 0 and depth >= 6 else 0)
    test_count = max(requested_test, 1 if test_ratio > 0 and depth >= 8 else 0)

    def _reserve(center_fraction: float, split_name: str, count: int) -> None:
        if count <= 0:
            return
        center = int(round((depth - 1) * center_fraction))
        start = max(0, center - count // 2)
        end = min(depth, start + count)
        start = max(0, end - count)

        # Slide the block until it no longer collides with a holdout block.
        while any(assignments[idx] != "train" for idx in range(start, end)) and end < depth:
            start += 1
            end += 1
        while any(assignments[idx] != "train" for idx in range(start, end)) and start > 0:
            start -= 1
            end -= 1

        for idx in range(start, end):
            assignments[idx] = split_name

        buffer_start = max(0, start - int(context_radius))
        buffer_end = min(depth, end + int(context_radius))
        for idx in range(buffer_start, buffer_end):
            if assignments[idx] == "train":
                assignments[idx] = "buffer"

    _reserve(0.35, "val", val_count)
    _reserve(0.70, "test", test_count)

    if not any(split == "train" for split in assignments):
        for idx, split in enumerate(assignments):
            if split == "buffer":
                assignments[idx] = "train"
                break
    return assignments



def build_case_index(
    cases: list[CT25DCase],
    group_split_map: dict[str, str],
    slice_split_overrides: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    slice_split_overrides = slice_split_overrides or {}
    for case in cases:
        case_slice_splits = slice_split_overrides.get(case.split_group_id)
        default_split_name = group_split_map.get(case.split_group_id, "train")
        for slice_index in range(case.volume_shape_zyx[0]):
            split_name = case_slice_splits[slice_index] if case_slice_splits is not None else default_split_name
            prev_index, center_index, next_index = clamp_stack_indices(
                center_index=slice_index,
                depth=case.volume_shape_zyx[0],
            )
            slice_mask_path = None
            if case.slice_labels_dir is not None:
                candidate = case.slice_labels_dir / f"slice_{slice_index:03d}.npz"
                if candidate.exists():
                    slice_mask_path = str(candidate.resolve())

            records.append(
                {
                    "split": split_name,
                    "patient_id": case.patient_id,
                    "study_instance_uid": case.study_instance_uid,
                    "series_instance_uid": case.series_instance_uid,
                    "split_group_id": case.split_group_id,
                    "series_dir": str(case.series_dir),
                    "volume_path": str(case.volume_path),
                    "label_volume_path": str(case.label_volume_path),
                    "slice_mask_path": slice_mask_path,
                    "slice_index": int(slice_index),
                    "stack_prev_index": int(prev_index),
                    "stack_center_index": int(center_index),
                    "stack_next_index": int(next_index),
                    "depth": int(case.volume_shape_zyx[0]),
                    "height": int(case.volume_shape_zyx[1]),
                    "width": int(case.volume_shape_zyx[2]),
                    "spacing_z_mm": float(case.spacing_zyx[0]),
                    "spacing_y_mm": float(case.spacing_zyx[1]),
                    "spacing_x_mm": float(case.spacing_zyx[2]),
                },
            )

    dataframe = pd.DataFrame.from_records(records)
    if dataframe.empty:
        raise ValueError("No Day 4 index rows were created.")
    return dataframe.sort_values(
        by=["split", "patient_id", "series_instance_uid", "slice_index"],
        kind="stable",
    ).reset_index(drop=True)


def summarize_index(index_df: pd.DataFrame) -> dict[str, Any]:
    split_summary = {}
    for split_name, split_frame in index_df.groupby("split", sort=False):
        split_summary[str(split_name)] = {
            "slice_count": int(len(split_frame)),
            "patient_count": int(split_frame["patient_id"].nunique()),
            "series_count": int(split_frame["series_instance_uid"].nunique()),
        }

    return {
        "total_rows": int(len(index_df)),
        "patient_count": int(index_df["patient_id"].nunique()),
        "series_count": int(index_df["series_instance_uid"].nunique()),
        "split_summary": split_summary,
    }


class Compose25D:
    def __init__(self, transforms: list[Any]) -> None:
        self.transforms = list(transforms)

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        for transform in self.transforms:
            image, mask = transform(image, mask, rng)
        return image, mask


class RandomFlip25D:
    def __init__(
        self,
        horizontal_prob: float = 0.5,
        vertical_prob: float = 0.5,
    ) -> None:
        self.horizontal_prob = float(horizontal_prob)
        self.vertical_prob = float(vertical_prob)

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if rng.random() < self.horizontal_prob:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()
        if rng.random() < self.vertical_prob:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()
        return image, mask


class RandomRotate25D:
    def __init__(
        self,
        max_degrees: float = 7.5,
        probability: float = 0.5,
    ) -> None:
        self.max_degrees = float(max_degrees)
        self.probability = float(probability)

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.max_degrees <= 0 or rng.random() >= self.probability:
            return image, mask

        angle = float(rng.uniform(-self.max_degrees, self.max_degrees))
        rotated_image = ndimage.rotate(
            image,
            angle=angle,
            axes=(1, 2),
            reshape=False,
            order=1,
            mode="nearest",
        )
        rotated_mask = ndimage.rotate(
            mask,
            angle=angle,
            axes=(0, 1),
            reshape=False,
            order=0,
            mode="constant",
            cval=0,
        )
        return rotated_image.astype(np.float32, copy=False), rotated_mask.astype(np.int64, copy=False)


class RandomIntensityJitter25D:
    def __init__(
        self,
        probability: float = 0.5,
        scale_range: tuple[float, float] = (0.95, 1.05),
        shift_range: tuple[float, float] = (-0.03, 0.03),
        clip_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.probability = float(probability)
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.clip_range = clip_range

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if rng.random() >= self.probability:
            return image, mask

        scale = float(rng.uniform(self.scale_range[0], self.scale_range[1]))
        shift = float(rng.uniform(self.shift_range[0], self.shift_range[1]))
        jittered = image * scale + shift
        jittered = np.clip(jittered, self.clip_range[0], self.clip_range[1])
        return jittered.astype(np.float32, copy=False), mask


def build_default_train_transforms(
    rotation_degrees: float = 7.5,
    enable_intensity_jitter: bool = True,
) -> Compose25D:
    transforms: list[Any] = [
        RandomFlip25D(horizontal_prob=0.5, vertical_prob=0.5),
        RandomRotate25D(max_degrees=rotation_degrees, probability=0.5),
    ]
    if enable_intensity_jitter:
        transforms.append(RandomIntensityJitter25D(probability=0.5))
    return Compose25D(transforms)


def _torch_required() -> None:
    if torch is None:
        raise RuntimeError("torch is required for the Day 4 CT25D dataset and dataloaders.")


class CT25DDataset(Dataset):
    def __init__(
        self,
        index_csv_path: str | Path,
        split: str | None = None,
        label_key: str = "pseudo_labels",
        transforms: Any | None = None,
        seed: int = 13,
    ) -> None:
        _torch_required()
        self.index_csv_path = Path(index_csv_path).expanduser().resolve()
        self.index_df = pd.read_csv(self.index_csv_path)
        if split is not None:
            self.index_df = self.index_df[self.index_df["split"] == split].reset_index(drop=True)

        self.label_key = label_key
        self.transforms = transforms
        self.seed = int(seed)
        self._volume_cache: dict[str, np.ndarray] = {}
        self._label_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return int(len(self.index_df))

    def _sample_seed(self, row: pd.Series) -> int:
        raw = "|".join(
            [
                str(row["split"]),
                str(row["series_instance_uid"]),
                str(int(row["slice_index"])),
                str(self.seed),
            ],
        )
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return int(digest[:16], 16) % (2**32)

    def _load_volume(self, path_value: str) -> np.ndarray:
        if path_value not in self._volume_cache:
            volume, _ = load_processed_volume(path_value)
            self._volume_cache[path_value] = volume.astype(np.float32, copy=False)
        return self._volume_cache[path_value]

    def _load_label_volume(self, path_value: str) -> np.ndarray:
        if path_value not in self._label_cache:
            with np.load(Path(path_value).expanduser().resolve()) as data:
                self._label_cache[path_value] = data[self.label_key].astype(np.int64)
        return self._label_cache[path_value]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.index_df.iloc[int(index)]
        volume = self._load_volume(str(row["volume_path"]))
        labels = self._load_label_volume(str(row["label_volume_path"]))

        slice_index = int(row["slice_index"])
        image = build_25d_stack(volume=volume, center_index=slice_index)
        mask = labels[slice_index].astype(np.int64, copy=False)

        if self.transforms is not None:
            rng = np.random.default_rng(self._sample_seed(row))
            image, mask = self.transforms(image, mask, rng)

        image_tensor = torch.from_numpy(np.ascontiguousarray(image.astype(np.float32, copy=False)))
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask.astype(np.int64, copy=False)))
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "slice_index": slice_index,
            "patient_id": str(row["patient_id"]),
            "series_instance_uid": str(row["series_instance_uid"]),
            "split": str(row["split"]),
        }


def _seed_worker(worker_id: int) -> None:
    del worker_id
    np.random.seed(torch.initial_seed() % (2**32))


def create_dataloaders(
    index_csv_path: str | Path,
    batch_size: int = 4,
    num_workers: int = 0,
    seed: int = 13,
    train_transforms: Any | None = None,
    eval_transforms: Any | None = None,
    shuffle_train: bool = True,
) -> dict[str, DataLoader]:
    _torch_required()

    dataloaders: dict[str, DataLoader] = {}
    for split_name, transforms in (
        ("train", train_transforms),
        ("val", eval_transforms),
        ("test", eval_transforms),
    ):
        dataset = CT25DDataset(
            index_csv_path=index_csv_path,
            split=split_name,
            transforms=transforms,
            seed=seed,
        )
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        use_shuffle = bool(split_name == "train" and shuffle_train and len(dataset) > 0)
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=use_shuffle,
            num_workers=int(num_workers),
            pin_memory=False,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
            generator=generator,
        )
    return dataloaders
