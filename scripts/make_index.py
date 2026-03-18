from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import load_app_config  # noqa: E402
from data.ct25d_dataset import (  # noqa: E402
    assign_group_splits,
    assign_single_case_slice_splits,
    build_case_index,
    build_default_train_transforms,
    create_dataloaders,
    discover_legacy_case,
    summarize_index,
)
from visualization import save_day4_batch_viz  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Day 4 data module: build a 2.5D slice index, create leakage-safe split labels, "
            "and export a batch sanity-check figure."
        ),
    )
    parser.add_argument("--series-dir", type=str, default=None, help="Path to the source DICOM series.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for Day 4 reports and figures.")
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help="Directory containing volume.nii.gz/volume.npz and pseudo_labels.",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default=None,
        help="Optional output CSV path. Defaults to <processed-dir>/index.csv.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Seed used for split assignment and transforms.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the Day 4 sanity-check loader.")
    parser.add_argument("--num-workers", type=int, default=0, help="PyTorch DataLoader worker count.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio at group level.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio at group level.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio at group level.")
    parser.add_argument(
        "--rotation-deg",
        type=float,
        default=7.5,
        help="Maximum absolute in-plane rotation for train-time augmentation.",
    )
    parser.add_argument(
        "--disable-intensity-jitter",
        action="store_true",
        help="Disable train-time intensity jitter in the sample visualization loader.",
    )
    parser.add_argument(
        "--intra-series-context-radius",
        type=int,
        default=1,
        help="Slice radius excluded around intra-series val/test holdout bands when only one case is available.",
    )
    return parser


def _pick_preview_split(index_df: pd.DataFrame) -> str:
    for split_name in ("val", "test", "train"):
        if int((index_df["split"] == split_name).sum()) > 0:
            return split_name
    raise ValueError("Index has no rows to visualize.")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    config = load_app_config(
        series_dir=args.series_dir,
        output_dir=args.output_dir,
        processed_dir=args.processed_dir,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.processed_dir.mkdir(parents=True, exist_ok=True)

    case = discover_legacy_case(
        series_dir=config.series_dir,
        processed_dir=config.processed_dir,
    )
    group_split_map = assign_group_splits(
        group_ids=[case.split_group_id],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    slice_split_overrides = None
    if len(group_split_map) == 1:
        slice_split_overrides = {
            case.split_group_id: assign_single_case_slice_splits(
                depth=case.volume_shape_zyx[0],
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                context_radius=args.intra_series_context_radius,
            ),
        }
    index_df = build_case_index(
        cases=[case],
        group_split_map=group_split_map,
        slice_split_overrides=slice_split_overrides,
    )

    index_path = Path(args.index_path).expanduser().resolve() if args.index_path else (config.processed_dir / "index.csv")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(index_path, index=False)

    train_transforms = build_default_train_transforms(
        rotation_degrees=args.rotation_deg,
        enable_intensity_jitter=not args.disable_intensity_jitter,
    )
    dataloaders = create_dataloaders(
        index_csv_path=index_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_transforms=train_transforms,
        eval_transforms=None,
        shuffle_train=False,
    )

    preview_split = _pick_preview_split(index_df)
    preview_batch = next(iter(dataloaders[preview_split]))
    batch_viz_path = config.output_dir / "day4_batch_viz.png"
    save_day4_batch_viz(
        batch=preview_batch,
        output_path=batch_viz_path,
        title=f"Day 4 2.5D Batch Sanity Check ({preview_split})",
    )

    leakage_summary = (
        "Single patient/series detected; contiguous val/test slice bands plus context buffers were created to reduce train/holdout leakage."
        if index_df["split_group_id"].nunique() < 2
        else "Patient/series groups were assigned to one split each."
    )
    report = {
        "series_dir": str(case.series_dir),
        "processed_volume_path": str(case.volume_path),
        "pseudo_label_volume_path": str(case.label_volume_path),
        "index_csv_path": str(index_path),
        "batch_visualization_path": str(batch_viz_path),
        "seed": int(args.seed),
        "case_shape_zyx": list(case.volume_shape_zyx),
        "case_spacing_zyx_mm": [float(v) for v in case.spacing_zyx],
        "summary": summarize_index(index_df),
        "group_split_map": group_split_map,
        "slice_split_overrides": slice_split_overrides,
        "buffer_slice_count": int((index_df["split"] == "buffer").sum()),
        "leakage_summary": leakage_summary,
        "train_transforms": {
            "flip": True,
            "rotation_degrees": float(args.rotation_deg),
            "intensity_jitter": bool(not args.disable_intensity_jitter),
        },
    }
    report_path = config.output_dir / "day4_data_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Day 4 data module complete.")
    print(f"Index CSV: {index_path}")
    print(f"Batch sanity-check figure: {batch_viz_path}")
    print(f"Report: {report_path}")
    print(
        "Split counts: "
        + ", ".join(
            f"{split_name}={int((index_df['split'] == split_name).sum())}"
            for split_name in ("train", "val", "test")
        ),
    )
    print(leakage_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
