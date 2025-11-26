"""
Benchmark harness for wall detection pipelines.
Runs detectors over paired images/masks and reports pixel IoU and structural F1.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from backend.ml_wall_detection import get_detector
from backend.vectorizer import vectorize_from_wall_mask

WallSegment = Tuple[float, float, float, float]


def _rasterize_walls(walls: List[WallSegment], shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for x1, y1, x2, y2 in walls:
        cv2.line(
            mask,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            color=255,
            thickness=3,
            lineType=cv2.LINE_AA,
        )
    return mask


def _pixel_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred_bin = pred.astype(bool)
    gt_bin = gt.astype(bool)

    tp = int(np.logical_and(pred_bin, gt_bin).sum())
    fp = int(np.logical_and(pred_bin, np.logical_not(gt_bin)).sum())
    fn = int(np.logical_and(np.logical_not(pred_bin), gt_bin).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

    return {
        "pixelPrecision": precision,
        "pixelRecall": recall,
        "pixelF1": f1,
        "iou": iou,
    }


def _segment_angle(seg: WallSegment) -> float:
    x1, y1, x2, y2 = seg
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))


def _overlap_ratio(a: WallSegment, b: WallSegment) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    if abs(ax2 - ax1) >= abs(ay2 - ay1):
        # horizontal-ish, compare x spans
        a_min, a_max = sorted([ax1, ax2])
        b_min, b_max = sorted([bx1, bx2])
    else:
        # vertical-ish, compare y spans
        a_min, a_max = sorted([ay1, ay2])
        b_min, b_max = sorted([by1, by2])

    overlap = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    shortest = max(1e-6, min(a_max - a_min, b_max - b_min))
    return overlap / shortest


def _structural_f1(pred: List[WallSegment], gt: List[WallSegment]) -> float:
    if not pred or not gt:
        return 0.0

    matched_pred = set()
    matched_gt = set()
    for i, p in enumerate(pred):
        angle_p = _segment_angle(p)
        for j, g in enumerate(gt):
            if j in matched_gt:
                continue
            angle_g = _segment_angle(g)
            if abs(angle_p - angle_g) > 5.0:
                continue
            if _overlap_ratio(p, g) >= 0.5:
                matched_pred.add(i)
                matched_gt.add(j)
                break

    precision = len(matched_pred) / len(pred)
    recall = len(matched_gt) / len(gt)
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def run_benchmark(
    images_dir: Path,
    masks_dir: Path,
    detector_name: str,
    meters_per_pixel: float,
) -> Dict[str, object]:
    detector = get_detector(detector_name)  # type: ignore[arg-type]
    samples = []
    image_paths = sorted([p for p in images_dir.glob("*.png")])

    pixel_f1s: List[float] = []
    ious: List[float] = []
    structural_f1s: List[float] = []

    for image_path in image_paths:
        name = image_path.stem
        mask_path = masks_dir / f"{name}_mask.png"
        if not mask_path.exists():
            continue

        image_bgr = cv2.imread(str(image_path))
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or gt_mask is None:
            continue

        result = detector.detect(image_bgr=image_bgr, meters_per_pixel=meters_per_pixel)
        pred_mask = _rasterize_walls(result.walls, gt_mask.shape[:2])

        metrics = _pixel_metrics(pred_mask, gt_mask)
        gt_segments, _ = vectorize_from_wall_mask(gt_mask.astype(np.uint8) // 255, image_bgr, meters_per_pixel)
        structural_f1 = _structural_f1(result.walls, gt_segments)

        pixel_f1s.append(metrics["pixelF1"])
        ious.append(metrics["iou"])
        structural_f1s.append(structural_f1)

        samples.append(
            {
                "name": name,
                "iou": metrics["iou"],
                "pixelPrecision": metrics["pixelPrecision"],
                "pixelRecall": metrics["pixelRecall"],
                "pixelF1": metrics["pixelF1"],
                "structuralF1": structural_f1,
            }
        )

    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_pixel_f1 = float(np.mean(pixel_f1s)) if pixel_f1s else 0.0
    mean_structural_f1 = float(np.mean(structural_f1s)) if structural_f1s else 0.0

    return {
        "detector": detector_name,
        "metersPerPixel": meters_per_pixel,
        "global": {
            "meanIoU": mean_iou,
            "meanPixelF1": mean_pixel_f1,
            "meanStructuralF1": mean_structural_f1,
        },
        "perSample": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark wall detectors.")
    parser.add_argument("-images-dir", required=True, type=Path, help="Directory containing PNG floorplan images")
    parser.add_argument("-masks-dir", required=True, type=Path, help="Directory containing *_mask.png wall masks")
    parser.add_argument("-detector", default="ml-v2", help="Detector name (ml, ml-v2, ml-dexined, ml-segformer)")
    parser.add_argument("-meters-per-pixel", type=float, default=0.05, help="Meters per pixel for the dataset")
    parser.add_argument("-output-json", default="wall_benchmark_results.json", help="Where to write results JSON")
    args = parser.parse_args()

    results = run_benchmark(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        detector_name=args.detector,
        meters_per_pixel=args.meters_per_pixel,
    )

    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote benchmark results to {output_path}")
    print(json.dumps(results["global"], indent=2))


if __name__ == "__main__":
    main()
