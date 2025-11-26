import base64
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import urllib.request


MODEL_DIR = Path(__file__).resolve().parent / "models"
HED_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/hed.prototxt"
HED_CAFFE_URL = "https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/hed_pretrained_bsds.caffemodel"


class HEDModelUnavailable(RuntimeError):
    """Raised when the Holistically-Nested Edge Detection weights cannot be loaded."""


def ensure_hed_weights() -> Tuple[Path, Path]:
    """Ensure the HED prototxt/caffemodel pair exists locally, downloading if missing."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    prototxt_path = MODEL_DIR / "hed.prototxt"
    caffemodel_path = MODEL_DIR / "hed_pretrained_bsds.caffemodel"

    if not prototxt_path.exists():
        urllib.request.urlretrieve(HED_PROTOTXT_URL, prototxt_path)

    if not caffemodel_path.exists():
        urllib.request.urlretrieve(HED_CAFFE_URL, caffemodel_path)

    return prototxt_path, caffemodel_path


def hed_edges(image: np.ndarray) -> np.ndarray:
    prototxt_path, caffemodel_path = ensure_hed_weights()

    net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(caffemodel_path))
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1.0,
        size=(w, h),
        mean=(104.00698793, 116.66876762, 122.67891434),
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    hed = net.forward()[0, 0]
    hed = cv2.resize(hed, (w, h))
    hed = (255 * hed).astype(np.uint8)
    return hed


def detect_walls_ml(image: np.ndarray, meters_per_pixel: float) -> Tuple[List[Dict], Dict, Dict]:
    """Wall detection using a pre-trained HED model followed by vectorization."""

    try:
        edge_map = hed_edges(image)
    except Exception as exc:  # pragma: no cover - download/runtime failure path
        raise HEDModelUnavailable(
            "HED weights missing or corrupted. Downloaded files might be incomplete; "
            "delete backend/models and retry."
        ) from exc

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Suppress thin strokes (text, rulers, dimensions) before vectorization
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    text_thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        5,
    )

    stroke_map = cv2.distanceTransform(text_thresh, cv2.DIST_L2, 3)
    thin_stroke_px = max(2, int(0.02 / meters_per_pixel))
    thin_mask = (stroke_map < thin_stroke_px).astype(np.uint8) * 255
    text_candidates = cv2.bitwise_and(text_thresh, thin_mask)

    max_text_area = max(32, int((0.15 / meters_per_pixel) ** 2))
    text_mask = np.zeros_like(text_candidates)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(text_candidates, connectivity=8)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        aspect = max(width, height) / max(1, min(width, height))

        if area <= max_text_area or (min(width, height) <= thin_stroke_px * 1.6 and aspect >= 6):
            text_mask[labels == label] = 255

    clean_edges = cv2.bitwise_and(edge_map, edge_map, mask=cv2.bitwise_not(text_mask))

    # Boost edges and binarize
    enhanced = cv2.GaussianBlur(clean_edges, (5, 5), 0)
    _, binary = cv2.threshold(enhanced, 30, 255, cv2.THRESH_BINARY)

    # Connect close strokes to form thicker wall candidates
    kernel_size = max(3, int(0.06 / meters_per_pixel))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Probabilistic Hough to extract line segments
    min_wall_length_m = 0.4
    min_line_length_px = max(12, int(min_wall_length_m / meters_per_pixel))
    max_gap_px = int(min_line_length_px * 0.4)
    lines = cv2.HoughLinesP(
        closed,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=min_line_length_px,
        maxLineGap=max_gap_px,
    )

    walls: List[Dict] = []
    overlay = _render_glow_overlay(image, lines)

    if lines is not None:
        for idx, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            wall = {
                "id": f"ML-W-{idx}",
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "material": "Concrete",
                "attenuation": 12.0,
                "thickness": max(10, int(kernel_size * 1.5)),
                "height": 3.0,
                "elevation": 0.0,
                "metadata": {"color": "#475569"},
            }
            walls.append(wall)
            cv2.line(overlay, (x1, y1), (x2, y2), (64, 157, 255), 2)

    preview = {
        "overlay": _encode_overlay(overlay),
        "wall_count": len(walls),
        "processing_ms": None,
    }
    diagnostics = {
        "edge_pixel_ratio": float(np.count_nonzero(edge_map)) / edge_map.size,
        "raw_segments": len(lines) if lines is not None else 0,
        "merged_segments": len(walls),
        "gap_closures": 0,
        "text_suppressed_px": int(np.count_nonzero(text_mask)),
        "notes": "HED edge map → thin-stroke suppression → morphological closing → HoughLinesP",
    }

    return walls, preview, diagnostics


def _encode_overlay(image: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', image)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"


def _render_glow_overlay(image: np.ndarray, lines) -> np.ndarray:
    base = image.copy()
    glow = np.zeros_like(base)
    highlight_color = (37, 211, 255)  # BGR neon teal

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(glow, (x1, y1), (x2, y2), highlight_color, thickness=6, lineType=cv2.LINE_AA)

    blurred = cv2.GaussianBlur(glow, (0, 0), sigmaX=6, sigmaY=6)
    composite = cv2.addWeighted(base, 0.55, blurred, 0.45, 0)
    composite = cv2.addWeighted(composite, 0.8, glow, 0.6, 0)
    return composite
