from dataclasses import dataclass
from typing import Dict, Protocol, Tuple, List, Literal, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

DetectorName = Literal["ml", "ml-v2", "ml-dexined", "ml-segformer"]

WallSegment = Tuple[float, float, float, float]


@dataclass
class WallDetectionResult:
    walls: List[WallSegment]
    overlay_png_base64: Optional[str]
    diagnostics: Dict[str, object]


class WallDetector(Protocol):
    def detect(self, image_bgr: np.ndarray, meters_per_pixel: float) -> WallDetectionResult:  # pragma: no cover - protocol
        ...


_DETECTORS: Dict[DetectorName, WallDetector] = {}


def register_detector(name: DetectorName, detector: WallDetector) -> None:
    _DETECTORS[name] = detector


def get_detector(name: DetectorName) -> WallDetector:
    if name in _DETECTORS:
        return _DETECTORS[name]

    if name == "ml-v2" and "ml" in _DETECTORS:
        # Graceful fallback if v2 is unavailable
        return _DETECTORS["ml"]

    raise ValueError(f"Unknown wall detector: {name}")


def _safe_register(name: DetectorName, import_path: str, cls_name: str) -> None:
    """
    Attempt to import and register a detector without failing the whole app.
    """
    try:
        module = __import__(import_path, fromlist=[cls_name])
        detector_cls = getattr(module, cls_name)
        register_detector(name, detector_cls())
        logger.info("Registered wall detector '%s' from %s.%s", name, import_path, cls_name)
    except Exception as exc:  # pragma: no cover - defensive import
        logger.warning("Could not register detector %s (%s.%s): %s", name, import_path, cls_name, exc)


# Register detectors at import time (best-effort)
_safe_register("ml", "backend.hed_detector", "HedWallDetector")
_safe_register("ml-dexined", "backend.dexined_detector", "DexinedWallDetector")
_safe_register("ml-segformer", "backend.segformer_detector", "SegformerWallDetector")

# Alias ml-v2 to the segformer detector when available
if "ml-segformer" in _DETECTORS:
    _DETECTORS["ml-v2"] = _DETECTORS["ml-segformer"]
