from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERIES_DIR = REPO_ROOT / "data" / "dicom_series_01"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"


@dataclass(frozen=True)
class AppConfig:
    series_dir: Path
    output_dir: Path


def _resolve_path(raw_value: str | None, fallback: Path) -> Path:
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    return fallback.resolve()


def load_app_config(series_dir: str | None = None, output_dir: str | None = None) -> AppConfig:
    configured_series_dir = series_dir or os.getenv("DICOM_SERIES_DIR")
    configured_output_dir = output_dir or os.getenv("DICOM_OUTPUT_DIR")
    return AppConfig(
        series_dir=_resolve_path(configured_series_dir, DEFAULT_SERIES_DIR),
        output_dir=_resolve_path(configured_output_dir, DEFAULT_OUTPUT_DIR),
    )
