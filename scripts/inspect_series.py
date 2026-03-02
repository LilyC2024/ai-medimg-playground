from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import load_app_config  # noqa: E402
from dicom_loader import load_dicom_series, write_metadata_json  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a DICOM series and export Day 1 diagnostics.")
    parser.add_argument(
        "--series-dir",
        type=str,
        default=None,
        help="Path to a DICOM series directory. Falls back to DICOM_SERIES_DIR or default config path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where plots and metadata JSON are written.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive axial viewer with brain and bone windows.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.show:
        os.environ.setdefault("MPLBACKEND", "Agg")
    from visualization import save_hu_histogram, save_montage, show_axial_scroll  # noqa: WPS433,E402

    config = load_app_config(series_dir=args.series_dir, output_dir=args.output_dir)
    volume = load_dicom_series(config.series_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = config.output_dir / "day1_metadata.json"
    hist_path = config.output_dir / "day1_hu_hist.png"
    montage_path = config.output_dir / "day1_montage.png"

    write_metadata_json(volume.metadata, metadata_path)
    save_hu_histogram(volume.volume_hu, hist_path)
    save_montage(volume.volume_hu, montage_path)

    print("Series inspection complete.")
    print(f"Series dir: {config.series_dir}")
    print(f"Slice count: {volume.metadata.slice_count}")
    print(f"Volume shape: {tuple(volume.volume_hu.shape)}")
    print(f"HU min/max: {float(volume.volume_hu.min()):.2f} / {float(volume.volume_hu.max()):.2f}")
    print("Metadata:")
    print(json.dumps(volume.metadata.to_dict(), indent=2))
    print("Artifacts written:")
    print(f"- {metadata_path}")
    print(f"- {hist_path}")
    print(f"- {montage_path}")

    if args.show:
        show_axial_scroll(volume.volume_hu)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
