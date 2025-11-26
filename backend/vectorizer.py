import base64
import math
from typing import List, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize

from backend.config_wall_detection import (
    CLOSE_RADIUS_METERS,
    MERGE_GAP_METERS,
    MIN_SPUR_METERS,
    MIN_WALL_METERS,
    OVERLAY_ALPHA,
    HOUGH_THRESHOLD,
)
from backend.ml_wall_detection import WallSegment

WallList = List[WallSegment]


def _encode_overlay(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        return ""
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _kernel_radius_px(meters_per_pixel: float) -> int:
    radius = int(round(CLOSE_RADIUS_METERS / max(meters_per_pixel, 1e-6)))
    return max(1, min(radius, 7))


def _prune_spurs(skeleton: np.ndarray, iterations: int) -> np.ndarray:
    """
    Remove small spurs from a skeleton by iteratively deleting end points.
    """
    skel = skeleton.copy()
    for _ in range(max(1, iterations)):
        to_delete = []
        ys, xs = np.nonzero(skel)
        for y, x in zip(ys, xs):
            neighbors = skel[max(0, y - 1): y + 2, max(0, x - 1): x + 2]
            count = int(np.count_nonzero(neighbors)) - 1  # exclude center
            if count <= 1:
                to_delete.append((y, x))
        if not to_delete:
            break
        for y, x in to_delete:
            skel[y, x] = 0
    return skel


def _segment_angle(seg: WallSegment) -> float:
    x1, y1, x2, y2 = seg
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def _segment_length(seg: WallSegment) -> float:
    x1, y1, x2, y2 = seg
    return math.hypot(x2 - x1, y2 - y1)


def _segments_mergeable(a: WallSegment, b: WallSegment, gap_px: float) -> bool:
    angle_a = _segment_angle(a)
    angle_b = _segment_angle(b)
    if abs(angle_a - angle_b) > 3.0:
        return False

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    endpoints_a = [(ax1, ay1), (ax2, ay2)]
    endpoints_b = [(bx1, by1), (bx2, by2)]

    for p1 in endpoints_a:
        for p2 in endpoints_b:
            if math.hypot(p1[0] - p2[0], p1[1] - p2[1]) <= gap_px:
                return True
    return False


def _merge_segments(segments: WallList, gap_px: float) -> WallList:
    merged = segments[:]
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if _segments_mergeable(merged[i], merged[j], gap_px):
                    points = [
                        (merged[i][0], merged[i][1]),
                        (merged[i][2], merged[i][3]),
                        (merged[j][0], merged[j][1]),
                        (merged[j][2], merged[j][3]),
                    ]
                    # pick farthest pair to keep full span
                    max_pair = max(
                        (
                            (p1, p2, math.hypot(p1[0] - p2[0], p1[1] - p2[1]))
                            for idx, p1 in enumerate(points)
                            for p2 in points[idx + 1:]
                        ),
                        key=lambda item: item[2],
                    )
                    (p1, p2, _) = max_pair
                    merged.pop(j)
                    merged.pop(i)
                    merged.append((p1[0], p1[1], p2[0], p2[1]))
                    changed = True
                    break
            if changed:
                break
    return merged


def _snap_axes(seg: WallSegment) -> WallSegment:
    x1, y1, x2, y2 = seg
    angle = abs(_segment_angle(seg))
    if angle < 3 or angle > 177:
        y_mean = (y1 + y2) / 2.0
        return (x1, y_mean, x2, y_mean)
    if 87 < angle < 93:
        x_mean = (x1 + x2) / 2.0
        return (x_mean, y1, x_mean, y2)
    return seg


def _draw_overlay(original_bgr: np.ndarray, walls: WallList) -> str:
    overlay = original_bgr.copy()
    for x1, y1, x2, y2 in walls:
        cv2.line(
            overlay,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            (0, 200, 80),
            2,
            cv2.LINE_AA,
        )
    blended = cv2.addWeighted(original_bgr, 1.0 - OVERLAY_ALPHA, overlay, OVERLAY_ALPHA, 0)
    return _encode_overlay(blended)


def vectorize_from_wall_mask(
    wall_mask: np.ndarray,
    original_bgr: np.ndarray,
    meters_per_pixel: float,
) -> Tuple[WallList, str]:
    """
    Convert a binary wall mask into vector wall segments.
    """
    mask_uint8 = (wall_mask.astype(np.uint8) * 255) if wall_mask.max() <= 1 else wall_mask.astype(np.uint8)

    kernel_radius = _kernel_radius_px(meters_per_pixel)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_radius * 2 + 1, kernel_radius * 2 + 1),
    )
    # Use larger kernel for closing gaps, smaller for opening (noise removal)
    # This prevents thin walls from being erased by MORPH_OPEN
    small_kernel_radius = max(1, min(kernel_radius, 2))
    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (small_kernel_radius * 2 + 1, small_kernel_radius * 2 + 1),
    )
    cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, small_kernel, iterations=1)

    skeleton = skeletonize(cleaned > 0).astype(np.uint8) * 255
    min_spur_px = max(1, int(round(MIN_SPUR_METERS / max(meters_per_pixel, 1e-6))))
    skeleton = _prune_spurs(skeleton, min_spur_px)

    min_line_length = max(5, int(round(MIN_WALL_METERS / max(meters_per_pixel, 1e-6))))
    max_line_gap = max(2, int(round(MERGE_GAP_METERS / max(meters_per_pixel, 1e-6))))

    lines = cv2.HoughLinesP(
        skeleton,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    segments: WallList = []
    if lines is not None:
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = line
            segments.append((float(x1), float(y1), float(x2), float(y2)))

    merge_gap_px = max(2.0, MERGE_GAP_METERS / max(meters_per_pixel, 1e-6))
    merged = _merge_segments(segments, merge_gap_px)
    snapped = [_snap_axes(seg) for seg in merged]
    min_wall_px = max(3.0, MIN_WALL_METERS / max(meters_per_pixel, 1e-6))
    filtered = [seg for seg in snapped if _segment_length(seg) >= min_wall_px]

    overlay_png = _draw_overlay(original_bgr, filtered) if len(original_bgr.shape) == 3 else ""
    return filtered, overlay_png


def vectorize_from_edge_map(
    edge_map: np.ndarray,
    original_bgr: np.ndarray,
    meters_per_pixel: float,
) -> Tuple[WallList, str]:
    """
    Vectorize from an edge map by thickening edges into a mask and reusing the mask path.
    """
    edge_norm = np.clip(edge_map, 0.0, 1.0).astype(np.float32)
    binary = (edge_norm >= 0.2).astype(np.uint8) * 255

    kernel_radius = _kernel_radius_px(meters_per_pixel)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_radius * 2 + 1, kernel_radius * 2 + 1),
    )

    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    wall_mask = (dilated > 0).astype(np.uint8)

    return vectorize_from_wall_mask(wall_mask, original_bgr, meters_per_pixel)
