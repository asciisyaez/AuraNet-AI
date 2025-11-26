"""Offline benchmark harness for wall detectors.

Usage:
    python wall_benchmark.py --dataset /path/to/samples --meters-per-pixel 0.05 --output results.json

Dataset layout:
    dataset/
      sample1.png
      sample1_mask.png   # binary wall mask (255 for wall pixels)
      sample2.jpg
      sample2_mask.png

The script pairs each image file with a mask of the same stem plus the `_mask` suffix.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from ml_wall_detection import detect_walls_ml


def walls_to_mask(walls: List[Dict], shape: Tuple[int, int]) -> np.ndarray:
    """Rasterize predicted walls into a binary mask."""

    mask = np.zeros(shape, dtype=np.uint8)
    for wall in walls:
        x1, y1, x2, y2 = (
            int(round(wall.get("x1", 0))),
            int(round(wall.get("y1", 0))),
            int(round(wall.get("x2", 0))),
            int(round(wall.get("y2", 0))),
        )
        thickness = int(round(wall.get("thickness", 6)))
        thickness = max(1, min(thickness, max(shape)))
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=thickness, lineType=cv2.LINE_AA)
    return mask


def compute_pixel_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred_bin = pred > 0
    target_bin = target > 0

    tp = float(np.logical_and(pred_bin, target_bin).sum())
    fp = float(np.logical_and(pred_bin, ~target_bin).sum())
    fn = float(np.logical_and(~pred_bin, target_bin).sum())

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    iou = tp / (tp + fp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def coverage_f1(pred: np.ndarray, target: np.ndarray, tolerance_px: int = 3) -> float:
    """Structural F1 using a tolerance band to allow slight misalignments."""

    if tolerance_px < 1:
        tolerance_px = 1

    kernel = np.ones((tolerance_px, tolerance_px), np.uint8)
    pred_dilated = cv2.dilate(pred, kernel)
    target_dilated = cv2.dilate(target, kernel)

    pred_hits = float(np.logical_and(pred > 0, target_dilated > 0).sum())
    target_hits = float(np.logical_and(target > 0, pred_dilated > 0).sum())

    pred_total = float((pred > 0).sum())
    target_total = float((target > 0).sum())

    precision = pred_hits / (pred_total + 1e-9)
    recall = target_hits / (target_total + 1e-9)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def find_samples(dataset: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for image_path in sorted(dataset.glob("*.*")):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        mask_candidate = image_path.with_name(f"{image_path.stem}_mask.png")
        if mask_candidate.exists():
            pairs.append((image_path, mask_candidate))
    return pairs


def evaluate_sample(image_path: Path, mask_path: Path, meters_per_pixel: float) -> Dict:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Unable to read mask: {mask_path}")

    walls, _preview, _diagnostics = detect_walls_ml(image, meters_per_pixel)
    pred_mask = walls_to_mask(walls, mask.shape)

    pixel_metrics = compute_pixel_metrics(pred_mask, mask)
    struct_f1 = coverage_f1(pred_mask, mask, tolerance_px=3)

    return {
        "image": str(image_path.name),
        "mask": str(mask_path.name),
        "pixel_precision": pixel_metrics["precision"],
        "pixel_recall": pixel_metrics["recall"],
        "pixel_iou": pixel_metrics["iou"],
        "pixel_f1": pixel_metrics["f1"],
        "structural_f1": struct_f1,
        "tp": pixel_metrics["tp"],
        "fp": pixel_metrics["fp"],
        "fn": pixel_metrics["fn"],
        "wall_count": len(walls),
    }


def summarize(results: List[Dict]) -> Dict[str, float]:
    if not results:
        return {}

    keys = [
        "pixel_precision",
        "pixel_recall",
        "pixel_iou",
        "pixel_f1",
        "structural_f1",
    ]
    summary: Dict[str, float] = {}
    for key in keys:
        summary[key] = float(np.mean([r[key] for r in results]))
    summary["samples"] = len(results)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark wall detection on labeled floor plans.")
    parser.add_argument("--dataset", type=Path, required=True, help="Folder with images and *_mask.png pairs")
    parser.add_argument("--meters-per-pixel", type=float, default=0.05, help="Scaling used for vectorization")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write JSON results")
    args = parser.parse_args()

    pairs = find_samples(args.dataset)
    if not pairs:
        raise SystemExit("No image/mask pairs found. Ensure files are named <stem>.png and <stem>_mask.png")

    results: List[Dict] = []
    for image_path, mask_path in pairs:
        metrics = evaluate_sample(image_path, mask_path, args.meters_per_pixel)
        results.append(metrics)
        print(
            f"{image_path.name}: IoU={metrics['pixel_iou']:.3f} F1={metrics['pixel_f1']:.3f} "
            f"StructF1={metrics['structural_f1']:.3f} Walls={metrics['wall_count']}"
        )

    summary = summarize(results)
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

    payload = {"summary": summary, "samples": results}
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
