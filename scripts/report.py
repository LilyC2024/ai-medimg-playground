from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from visualization import save_report_slice_montage  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day 6 standardized evaluation report generator.")
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--processed-dir", type=str, default=str(REPO_ROOT / "data_processed"))
    parser.add_argument("--report-path", type=str, default=None, help="Defaults to <output-dir>/report.md.")
    return parser


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_row(label: str, value: float | int | str) -> str:
    return f"| {label} | {value} |"


def _render_findings(classical_report: dict[str, object], infer_report: dict[str, object]) -> list[str]:
    findings: list[str] = []
    for message in classical_report.get("quality_flags", []):
        findings.append(f"- {message}")

    uncertainty = infer_report.get("uncertainty", {})
    if isinstance(uncertainty, dict):
        summary = uncertainty.get("summary", {})
        if isinstance(summary, dict) and float(summary.get("p95", 0.0)) > 0.6:
            findings.append("- Prediction uncertainty is elevated in the 95th percentile; review the worst slices manually.")

    if not findings:
        findings.append("- No automated quality gates fired, but pseudo-label supervision still limits clinical validity.")
    return findings


def main() -> int:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    processed_dir = Path(args.processed_dir).expanduser().resolve()
    report_path = Path(args.report_path).expanduser().resolve() if args.report_path else (output_dir / "report.md")

    preprocess_report = _load_json(output_dir / "day2_preprocess_report.json")
    classical_report = _load_json(output_dir / "day3_classical_report.json")
    train_report = _load_json(output_dir / "day5_train_report.json")
    infer_report = _load_json(output_dir / "day5_infer_report.json")
    prediction_path = processed_dir / "day5_predictions.npz"
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Missing inference artifact: {prediction_path}. Run scripts/infer.py before generating the report.",
        )

    with np.load(prediction_path) as data:
        center_slices = data["center_slices"].astype(np.float32)
        predicted_labels = data["predicted_labels"].astype(np.uint8)
        reference_labels = data["reference_labels"].astype(np.uint8)
        exported_slice_indices = data["slice_indices"].astype(np.int32).tolist()
        uncertainty = data["uncertainty"].astype(np.float32) if "uncertainty" in data else None

    per_slice_metrics = infer_report.get("per_slice_metrics", [])
    if not isinstance(per_slice_metrics, list) or not per_slice_metrics:
        raise ValueError("Inference report is missing per-slice metrics required for the Day 6 report.")

    sorted_best = sorted(per_slice_metrics, key=lambda item: (-float(item["dice"]), int(item["slice_index"])))[:3]
    sorted_worst = sorted(per_slice_metrics, key=lambda item: (float(item["dice"]), int(item["slice_index"])))[:3]
    montage_indices = [int(item["slice_index"]) for item in sorted_best + sorted_worst]
    row_lookup = {int(slice_index): row_index for row_index, slice_index in enumerate(exported_slice_indices)}
    row_positions = [row_lookup[slice_index] for slice_index in montage_indices]
    montage_path = save_report_slice_montage(
        center_slices=center_slices[row_positions],
        predicted_labels=predicted_labels[row_positions],
        reference_labels=reference_labels[row_positions],
        uncertainty=uncertainty[row_positions] if uncertainty is not None else None,
        slice_indices=montage_indices,
        output_path=output_dir / "report_montage.png",
        title="Day 6 Evaluation Montage (best first, then worst)",
    )

    metrics = infer_report.get("metrics", {})
    per_class_dice = metrics.get("per_class_dice", {}) if isinstance(metrics, dict) else {}
    uncertainty_info = infer_report.get("uncertainty", {})
    uncertainty_summary = uncertainty_info.get("summary", {}) if isinstance(uncertainty_info, dict) else {}

    lines = [
        "# Day 6 Evaluation Report",
        "",
        "## Summary",
        "",
        f"- Generated from: `{output_dir}`",
        f"- Prediction artifact: `{prediction_path}`",
        f"- Montage: `{montage_path}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        _metric_row("Mean Dice", f"{float(metrics.get('dice', 0.0)):.4f}"),
        _metric_row("Mean IoU", f"{float(metrics.get('iou', 0.0)):.4f}"),
        _metric_row("Brain Dice (class 1)", f"{float(per_class_dice.get('1', 0.0)):.4f}"),
        _metric_row("Bone Dice (class 2)", f"{float(per_class_dice.get('2', 0.0)):.4f}"),
        _metric_row("Overlap Dice (class 3)", f"{float(per_class_dice.get('3', 0.0)):.4f}"),
        _metric_row("Best Eval Dice", f"{float(train_report.get('best_eval_dice', 0.0)):.4f}"),
        _metric_row("Uncertainty Mean", f"{float(uncertainty_summary.get('mean', 0.0)):.4f}"),
        _metric_row("Uncertainty P95", f"{float(uncertainty_summary.get('p95', 0.0)):.4f}"),
        "",
        "## Best/Worst Slices",
        "",
        f"![Day 6 montage]({montage_path.name})",
        "",
        "| Group | Slice | Dice | IoU |",
        "| --- | --- | --- | --- |",
    ]

    for item in sorted_best:
        lines.append(
            f"| Best | {int(item['slice_index'])} | {float(item['dice']):.4f} | {float(item['iou']):.4f} |",
        )
    for item in sorted_worst:
        lines.append(
            f"| Worst | {int(item['slice_index'])} | {float(item['dice']):.4f} | {float(item['iou']):.4f} |",
        )

    lines.extend(
        [
            "",
            "## Quality Gates",
            "",
            *(_render_findings(classical_report, infer_report)),
            "",
            "## Limitations",
            "",
        ],
    )

    for limitation in (
        "Metrics are against Day 3 pseudo labels, not manual clinical ground truth.",
        "Single-series data means train/validation separation remains limited.",
        "Uncertainty is a proxy from entropy/TTA and is not calibrated risk estimation.",
        "Thick slices, metal, motion, and missing DICOM metadata can still degrade performance.",
    ):
        lines.append(f"- {limitation}")

    validation_messages = preprocess_report.get("dicom_validation_messages", [])
    if isinstance(validation_messages, list) and validation_messages:
        lines.extend(["", "## Validation Notes", ""])
        for message in validation_messages:
            lines.append(f"- {message}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved report: {report_path}")
    print(f"Saved montage: {montage_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
