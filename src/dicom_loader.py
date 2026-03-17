from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import FileDataset

from robustness import validate_spacing_zyx

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover - dependency is expected in runtime envs
    sitk = None


@dataclass(frozen=True)
class SeriesMetadata:
    series_instance_uid: str
    modality: str
    slice_count: int
    rows: int
    columns: int
    spacing_zyx: tuple[float, float, float]
    orientation_lps: tuple[float, float, float, float, float, float]
    rescale_slope_range: tuple[float, float]
    rescale_intercept_range: tuple[float, float]
    z_positions: list[float]
    validation_messages: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_instance_uid": self.series_instance_uid,
            "modality": self.modality,
            "slice_count": self.slice_count,
            "rows": self.rows,
            "columns": self.columns,
            "spacing_zyx": list(self.spacing_zyx),
            "orientation_lps": list(self.orientation_lps),
            "rescale_slope_range": list(self.rescale_slope_range),
            "rescale_intercept_range": list(self.rescale_intercept_range),
            "z_positions": self.z_positions,
            "validation_messages": self.validation_messages,
        }


@dataclass(frozen=True)
class DicomVolume:
    volume_hu: np.ndarray
    metadata: SeriesMetadata


@dataclass(frozen=True)
class _SliceRecord:
    path: Path
    dataset: FileDataset
    pixels: np.ndarray
    z_position: float
    instance_number: int
    slope: float
    intercept: float


def _transfer_syntax_uid(dataset: FileDataset) -> Any:
    file_meta = getattr(dataset, "file_meta", None)
    if file_meta is not None:
        transfer_syntax = file_meta.get("TransferSyntaxUID")
        if transfer_syntax is not None:
            return transfer_syntax
    return dataset.get("TransferSyntaxUID")


def _decode_with_simpleitk(path: Path) -> np.ndarray:
    if sitk is None:
        raise RuntimeError(
            "SimpleITK is required to decode compressed DICOM pixel data but is not installed."
        )
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected 2D slice, got shape {array.shape} while reading {path}.")
    return array.astype(np.float32)


def _decode_pixels(dataset: FileDataset, path: Path) -> np.ndarray:
    transfer_syntax = _transfer_syntax_uid(dataset)
    if bool(getattr(transfer_syntax, "is_compressed", False)):
        return _decode_with_simpleitk(path)
    try:
        return dataset.pixel_array.astype(np.float32)
    except RuntimeError as exc:
        # Fallback for malformed headers where transfer syntax compression flag is missing.
        if "decompress" in str(exc).lower():
            return _decode_with_simpleitk(path)
        raise


def _get_z_position(dataset: FileDataset, path: Path) -> float:
    image_position = dataset.get("ImagePositionPatient")
    if image_position and len(image_position) >= 3:
        return float(image_position[2])
    if "SliceLocation" in dataset:
        return float(dataset.SliceLocation)
    if "InstanceNumber" in dataset:
        return float(dataset.InstanceNumber)
    raise ValueError(
        "DICOM slice is missing all position tags needed for ordering "
        f"(ImagePositionPatient, SliceLocation, InstanceNumber): {path}",
    )


def _list_dicom_files(series_dir: Path) -> list[Path]:
    files = [path for path in series_dir.iterdir() if path.is_file()]
    if not files:
        raise FileNotFoundError(f"No files found in series directory: {series_dir}")
    return sorted(files)


def _read_slice(path: Path) -> _SliceRecord:
    dataset = pydicom.dcmread(str(path), force=True)
    if "PixelData" not in dataset:
        raise ValueError(f"File has no PixelData: {path}")

    pixels = _decode_pixels(dataset, path)
    slope = float(dataset.get("RescaleSlope", 1.0))
    intercept = float(dataset.get("RescaleIntercept", 0.0))
    z_position = _get_z_position(dataset, path)
    instance_number = int(dataset.get("InstanceNumber", 0))

    return _SliceRecord(
        path=path,
        dataset=dataset,
        pixels=pixels,
        z_position=z_position,
        instance_number=instance_number,
        slope=slope,
        intercept=intercept,
    )


def _slice_spacing_z(z_positions: np.ndarray, first_dataset: FileDataset) -> float:
    if z_positions.size >= 2:
        diffs = np.diff(z_positions)
        non_zero_diffs = np.abs(diffs[np.abs(diffs) > 1e-6])
        if non_zero_diffs.size:
            return float(np.median(non_zero_diffs))
    return float(first_dataset.get("SliceThickness", 1.0))


def _pixel_spacing_from_dataset(first_dataset: FileDataset, series_path: Path) -> tuple[float, float]:
    pixel_spacing = first_dataset.get("PixelSpacing")
    if pixel_spacing is None:
        pixel_spacing = first_dataset.get("ImagerPixelSpacing")

    if pixel_spacing is None:
        raise ValueError(
            "DICOM series is missing PixelSpacing/ImagerPixelSpacing. "
            f"Cannot compute physical spacing for: {series_path}",
        )
    if len(pixel_spacing) < 2:
        raise ValueError(
            "DICOM spacing tag is malformed: expected 2 values in PixelSpacing/ImagerPixelSpacing, "
            f"got {pixel_spacing!r} for {series_path}",
        )

    row_spacing = float(pixel_spacing[0])
    col_spacing = float(pixel_spacing[1])
    return row_spacing, col_spacing


def _orientation_or_default(first_dataset: FileDataset) -> tuple[float, float, float, float, float, float]:
    orientation = first_dataset.get("ImageOrientationPatient", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    orientation_values = tuple(float(value) for value in orientation[:6])
    if len(orientation_values) != 6:
        return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    return orientation_values


def load_dicom_series(series_dir: str | Path) -> DicomVolume:
    series_path = Path(series_dir).expanduser().resolve()
    if not series_path.exists():
        raise FileNotFoundError(f"Series directory does not exist: {series_path}")
    if not series_path.is_dir():
        raise NotADirectoryError(f"Expected a directory but got: {series_path}")

    records = [_read_slice(path) for path in _list_dicom_files(series_path)]
    records.sort(key=lambda item: (item.z_position, item.instance_number, item.path.name))

    first_shape = records[0].pixels.shape
    for record in records:
        if record.pixels.shape != first_shape:
            raise ValueError(
                f"Inconsistent slice shape: expected {first_shape}, got {record.pixels.shape} ({record.path})."
            )

    volume_hu = np.stack(
        [record.pixels * record.slope + record.intercept for record in records],
        axis=0,
    ).astype(np.float32)

    first_dataset = records[0].dataset
    row_spacing, col_spacing = _pixel_spacing_from_dataset(first_dataset, series_path)
    z_positions = np.asarray([record.z_position for record in records], dtype=np.float32)
    spacing_z = _slice_spacing_z(z_positions, first_dataset)
    spacing_zyx = (spacing_z, row_spacing, col_spacing)
    validation_messages = validate_spacing_zyx(spacing_zyx)

    slopes = np.asarray([record.slope for record in records], dtype=np.float32)
    intercepts = np.asarray([record.intercept for record in records], dtype=np.float32)

    metadata = SeriesMetadata(
        series_instance_uid=str(first_dataset.get("SeriesInstanceUID", "UNKNOWN")),
        modality=str(first_dataset.get("Modality", "UNKNOWN")),
        slice_count=len(records),
        rows=int(first_shape[0]),
        columns=int(first_shape[1]),
        spacing_zyx=spacing_zyx,
        orientation_lps=_orientation_or_default(first_dataset),
        rescale_slope_range=(float(slopes.min()), float(slopes.max())),
        rescale_intercept_range=(float(intercepts.min()), float(intercepts.max())),
        z_positions=[float(value) for value in z_positions.tolist()],
        validation_messages=validation_messages,
    )
    return DicomVolume(volume_hu=volume_hu, metadata=metadata)


def write_metadata_json(metadata: SeriesMetadata, output_path: str | Path) -> None:
    output_file = Path(output_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
