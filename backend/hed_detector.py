import logging
from typing import Dict

import cv2
import numpy as np

from backend.ml_wall_detection import WallDetector, WallDetectionResult
from backend.vectorizer import vectorize_from_edge_map

logger = logging.getLogger(__name__)


class HedWallDetector(WallDetector):
    """
    Baseline detector that approximates the legacy HED + Hough path.
    Uses a Canny-derived edge map to stay dependency-light while keeping API parity.
    """

    def detect(self, image_bgr: np.ndarray, meters_per_pixel: float) -> WallDetectionResult:
        try:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.bilateralFilter(gray, 7, 50, 50)
            # Tune thresholds relative to image contrast
            median = np.median(blurred)
            lower = int(max(0, 0.66 * median))
            upper = int(min(255, 1.33 * median))
            edges = cv2.Canny(blurred, threshold1=lower, threshold2=upper)
            edge_map = edges.astype(np.float32) / 255.0

            walls, overlay = vectorize_from_edge_map(
                edge_map=edge_map,
                original_bgr=image_bgr,
                meters_per_pixel=meters_per_pixel,
            )

            diagnostics: Dict[str, object] = {
                "detector": "ml",
                "modelName": "hed-baseline",
                "segmentCount": len(walls),
                "edgeDensity": float(edge_map.mean()),
                "notes": "OK",
            }
            return WallDetectionResult(
                walls=walls,
                overlay_png_base64=overlay,
                diagnostics=diagnostics,
            )
        except Exception as exc:  # pragma: no cover - safety
            logger.exception("HED baseline detection failed: %s", exc)
            diagnostics = {
                "detector": "ml",
                "modelName": "hed-baseline",
                "segmentCount": 0,
                "error": str(exc),
                "notes": "HED baseline failed",
            }
            return WallDetectionResult(walls=[], overlay_png_base64=None, diagnostics=diagnostics)
