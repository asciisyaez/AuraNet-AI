import cv2
import numpy as np
import base64
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional, Tuple
import uvicorn

from .ml_wall_detection import HEDModelUnavailable, detect_walls_ml

app = FastAPI()

class WallMetadata(BaseModel):
    color: str = "#475569"

class Wall(BaseModel):
    id: str
    x1: float
    y1: float
    x2: float
    y2: float
    material: str = "Concrete"
    attenuation: float = 12.0  # dB loss for concrete
    thickness: int = 10  # px thickness for 2D view
    height: float = 3.0  # meters
    elevation: float = 0.0  # meters above ground
    metadata: WallMetadata = WallMetadata()

class DetectionDiagnostics(BaseModel):
    edge_pixel_ratio: Optional[float] = None
    raw_segments: Optional[int] = None
    merged_segments: Optional[int] = None
    gap_closures: Optional[int] = None
    notes: Optional[str] = None


class DetectionPreview(BaseModel):
    overlay: Optional[str] = None
    mode: str = "balanced"
    wall_count: int = 0
    processing_ms: Optional[int] = None


class DetectionResponse(BaseModel):
    walls: List[Wall]
    preview: Optional[DetectionPreview] = None
    diagnostics: Optional[DetectionDiagnostics] = None

def decode_image(data_url: str) -> np.ndarray:
    # Remove header if present (e.g., "data:image/png;base64,")
    if ',' in data_url:
        data_url = data_url.split(',')[1]
    image_bytes = base64.b64decode(data_url)
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def detect_walls_opencv(image: np.ndarray, meters_per_pixel: float, mode: str = "balanced") -> Tuple[List[Wall], DetectionPreview, DetectionDiagnostics]:
    """Detect wall segments from a floorplan image.

    The detector emphasizes clean vector outputs with the following stages:
    1) Contrast + denoise, followed by adaptive thresholding.
    2) Morphological cleanup and gap closing (aggressiveness controlled by `mode`).
    3) Edge finding and Probabilistic Hough transform.
    4) Merging, angle snapping, and gap bridging on the vector graph.
    5) Lightweight overlay rendering for visual QA.
    """
    mode = mode or "balanced"

    # Tunable parameters per mode
    mode_config = {
        "precision": {
            "close_iterations": 1,
            "dilate_iterations": 1,
            "angle_snap": 4,
            "gap_px": 0.20,
            "hough_threshold": 70,
        },
        "balanced": {
            "close_iterations": 2,
            "dilate_iterations": 1,
            "angle_snap": 3,
            "gap_px": 0.30,
            "hough_threshold": 60,
        },
        "recall": {
            "close_iterations": 3,
            "dilate_iterations": 2,
            "angle_snap": 2,
            "gap_px": 0.40,
            "hough_threshold": 45,
        },
    }

    cfg = mode_config.get(mode, mode_config["balanced"])
    start_tick = cv2.getTickCount()

    # Convert to grayscale and enhance contrast to make faint blueprint lines pop
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Light bilateral filtering to suppress speckle noise while preserving edges
    blurred = cv2.bilateralFilter(enhanced, d=5, sigmaColor=35, sigmaSpace=15)

    # Adaptive thresholding to handle varying lighting/contrast
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    # Morphological operations to remove noise (text, furniture) and connect wall segments
    kernel_size = max(3, int(0.05 / meters_per_pixel))
    if kernel_size % 2 == 0:
        kernel_size += 1  # ensure odd size for symmetric operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    connected = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=cfg["close_iterations"])

    # Dilate slightly to bridge gaps between short segments
    dilated = cv2.dilate(connected, kernel, iterations=cfg["dilate_iterations"])

    # Edge detection (Canny)
    edges = cv2.Canny(dilated, 40, 120, apertureSize=3)

    # Hough Line Transform parameters
    min_wall_length_m = 0.45  # 45cm minimum wall length
    min_line_length_px = max(10, int(min_wall_length_m / meters_per_pixel))
    max_gap_px = max(5, int(cfg["gap_px"] / meters_per_pixel))

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=cfg["hough_threshold"],
        minLineLength=min_line_length_px,
        maxLineGap=max_gap_px,
    )

    diagnostics = DetectionDiagnostics(
        edge_pixel_ratio=float(np.mean(edges > 0)),
        raw_segments=0,
        merged_segments=0,
        gap_closures=0,
        notes=None,
    )

    if lines is None:
        preview = DetectionPreview(
            overlay=None,
            mode=mode,
            wall_count=0,
            processing_ms=int((cv2.getTickCount() - start_tick) / cv2.getTickFrequency() * 1000),
        )
        diagnostics.notes = "No line hypotheses found"
        return [], preview, diagnostics

    # Convert to list of tuples for easier processing
    all_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        all_lines.append((float(x1), float(y1), float(x2), float(y2)))
    diagnostics.raw_segments = len(all_lines)
    
    def line_angle(line):
        x1, y1, x2, y2 = line
        return np.arctan2(y2 - y1, x2 - x1)

    def lines_are_similar(line1, line2, dist_threshold, angle_threshold_deg=10):
        """Check if two lines are similar enough to merge."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        angle1 = line_angle(line1) % np.pi
        angle2 = line_angle(line2) % np.pi

        angle_diff = abs(angle1 - angle2)
        if angle_diff > np.pi / 2:
            angle_diff = np.pi - angle_diff

        if np.degrees(angle_diff) > angle_threshold_deg:
            return False

        mid1 = ((x1 + x2) / 2, (y1 + y2) / 2)
        mid2 = ((x3 + x4) / 2, (y3 + y4) / 2)
        midpoint_dist = np.sqrt((mid1[0] - mid2[0]) ** 2 + (mid1[1] - mid2[1]) ** 2)

        def point_to_line_dist(px, py, lx1, ly1, lx2, ly2):
            line_len_sq = (lx2 - lx1) ** 2 + (ly2 - ly1) ** 2
            if line_len_sq == 0:
                return np.sqrt((px - lx1) ** 2 + (py - ly1) ** 2)
            t = max(0, min(1, ((px - lx1) * (lx2 - lx1) + (py - ly1) * (ly2 - ly1)) / line_len_sq))
            proj_x = lx1 + t * (lx2 - lx1)
            proj_y = ly1 + t * (ly2 - ly1)
            return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

        distances = [
            point_to_line_dist(x1, y1, x3, y3, x4, y4),
            point_to_line_dist(x2, y2, x3, y3, x4, y4),
            point_to_line_dist(x3, y3, x1, y1, x2, y2),
            point_to_line_dist(x4, y4, x1, y1, x2, y2),
        ]

        len1 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        len2 = np.sqrt((x4 - x3) ** 2 + (y4 - y3) ** 2)
        max_len = max(len1, len2)

        return min(distances) < dist_threshold and midpoint_dist < max_len + dist_threshold

    def merge_two_lines(line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

        dx1, dy1 = x2 - x1, y2 - y1
        dx2, dy2 = x4 - x3, y4 - y3
        if dx1 * dx2 + dy1 * dy2 < 0:
            dx2, dy2 = -dx2, -dy2

        avg_dx = (dx1 + dx2) / 2
        avg_dy = (dy1 + dy2) / 2
        length = np.sqrt(avg_dx ** 2 + avg_dy ** 2)
        if length == 0:
            return line1
        avg_dx /= length
        avg_dy /= length

        projections = []
        for px, py in points:
            proj = px * avg_dx + py * avg_dy
            projections.append((proj, px, py))

        projections.sort(key=lambda x: x[0])
        _, min_x, min_y = projections[0]
        _, max_x, max_y = projections[-1]
        return (min_x, min_y, max_x, max_y)

    def extend_line(line, extension_px=8):
        x1, y1, x2, y2 = line
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length == 0:
            return line
        dx = (x2 - x1) / length
        dy = (y2 - y1) / length
        return (
            x1 - dx * extension_px,
            y1 - dy * extension_px,
            x2 + dx * extension_px,
            y2 + dy * extension_px,
        )

    def snap_line_orientation(line, divisions=4):
        """Snap line angle to the nearest multiple of pi/divisions."""
        x1, y1, x2, y2 = line
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        length = np.hypot(x2 - x1, y2 - y1)
        if length == 0:
            return line
        angle = line_angle(line)
        snapped_angle = round(angle / (np.pi / divisions)) * (np.pi / divisions)
        dx = np.cos(snapped_angle) * length / 2
        dy = np.sin(snapped_angle) * length / 2
        return (cx - dx, cy - dy, cx + dx, cy + dy)

    def snap_coordinates(lines, snap_step):
        def snap(v):
            return round(v / snap_step) * snap_step

        snapped = []
        for x1, y1, x2, y2 in lines:
            snapped.append((snap(x1), snap(y1), snap(x2), snap(y2)))
        return snapped

    def bridge_gaps(lines, gap_threshold_px, angle_threshold_deg=12):
        """Connect nearby endpoints when angles are compatible."""
        bridged = list(lines)
        endpoints = []
        for idx, line in enumerate(lines):
            x1, y1, x2, y2 = line
            endpoints.append((idx, 0, x1, y1))
            endpoints.append((idx, 1, x2, y2))

        added = 0
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                idx1, pos1, x1, y1 = endpoints[i]
                idx2, pos2, x2, y2 = endpoints[j]
                if idx1 == idx2:
                    continue
                dist = np.hypot(x2 - x1, y2 - y1)
                if dist > gap_threshold_px:
                    continue

                angle1 = line_angle(lines[idx1])
                angle2 = line_angle(lines[idx2])
                diff = abs(((angle1 - angle2 + np.pi) % np.pi) - (np.pi / 2))
                # accept near-parallel or T junctions by allowing wide tolerance
                if np.degrees(min(diff, np.pi - diff)) > angle_threshold_deg:
                    continue

                bridged.append((x1, y1, x2, y2))
                added += 1

        diagnostics.gap_closures = added
        return bridged

    extension_px = max(6, int(0.15 / meters_per_pixel))
    dist_threshold = max(15, int(0.15 / meters_per_pixel))
    snap_step = max(4, int(0.08 / meters_per_pixel))

    merged = True
    while merged:
        merged = False
        new_lines = []
        used = set()

        for i, line1 in enumerate(all_lines):
            if i in used:
                continue

            current_line = line1
            for j, line2 in enumerate(all_lines):
                if j <= i or j in used:
                    continue

                if lines_are_similar(current_line, line2, dist_threshold):
                    current_line = merge_two_lines(current_line, line2)
                    used.add(j)
                    merged = True

            new_lines.append(current_line)
            used.add(i)

        all_lines = new_lines

    # Snap angles to dominant grid and optionally bridge remaining gaps
    snapped = [snap_line_orientation(l, divisions=cfg["angle_snap"]) for l in all_lines]
    snapped = snap_coordinates(snapped, snap_step)
    bridged = bridge_gaps(snapped, gap_threshold_px=max_gap_px * 1.2)

    # Re-merge after snapping/bridging
    all_lines = bridged
    merged = True
    while merged:
        merged = False
        new_lines = []
        used = set()

        for i, line1 in enumerate(all_lines):
            if i in used:
                continue

            current_line = line1
            for j, line2 in enumerate(all_lines):
                if j <= i or j in used:
                    continue
                if lines_are_similar(current_line, line2, dist_threshold):
                    current_line = merge_two_lines(current_line, line2)
                    used.add(j)
                    merged = True

            new_lines.append(current_line)
            used.add(i)

        all_lines = new_lines

    diagnostics.merged_segments = len(all_lines)

    # Filter out very short lines after merging
    min_final_length = max(20, int(0.3 / meters_per_pixel))
    filtered_lines = []
    for x1, y1, x2, y2 in all_lines:
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length >= min_final_length:
            filtered_lines.append(extend_line((x1, y1, x2, y2), extension_px))

    detected_walls = []
    for i, (x1, y1, x2, y2) in enumerate(filtered_lines):
        detected_walls.append(
            Wall(
                id=f"detected-{i}",
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                material="Concrete",
                attenuation=12.0,
                thickness=max(8, int(0.12 / meters_per_pixel)),
                height=3.0,
                elevation=0.0,
                metadata=WallMetadata(color="#475569"),
            )
        )

    overlay = image.copy()
    for x1, y1, x2, y2 in filtered_lines:
        cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 122, 255), thickness=2, lineType=cv2.LINE_AA)

    overlay_bgr = cv2.addWeighted(image, 0.4, overlay, 0.6, 0)
    _, buffer = cv2.imencode('.png', overlay_bgr)
    overlay_base64 = base64.b64encode(buffer).decode('utf-8')
    overlay_data_url = f"data:image/png;base64,{overlay_base64}"

    preview = DetectionPreview(
        overlay=overlay_data_url,
        mode=mode,
        wall_count=len(detected_walls),
        processing_ms=int((cv2.getTickCount() - start_tick) / cv2.getTickFrequency() * 1000),
    )

    return detected_walls, preview, diagnostics

@app.post("/api/detect-walls", response_model=DetectionResponse)
async def detect_walls_endpoint(file: UploadFile = File(...), metersPerPixel: float = Form(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"walls": []}
        
    walls, preview, diagnostics = detect_walls_opencv(image, metersPerPixel)
    return {"walls": walls, "preview": preview, "diagnostics": diagnostics}

@app.post("/api/detect-walls-base64", response_model=DetectionResponse)
async def detect_walls_base64_endpoint(data: dict):
    image_data = data.get("image")
    meters_per_pixel = data.get("metersPerPixel", 0.05)
    detector = data.get("detector", "opencv")

    if not image_data:
        return {"walls": []}

    image = decode_image(image_data)

    if detector == "ml":
        try:
            wall_dicts, preview_dict, diagnostics_dict = detect_walls_ml(image, meters_per_pixel)
            walls = [Wall(**w) for w in wall_dicts]
            preview = DetectionPreview(**preview_dict) if preview_dict else None
            diagnostics = DetectionDiagnostics(**diagnostics_dict) if diagnostics_dict else None
        except HEDModelUnavailable as exc:
            diagnostics = DetectionDiagnostics(notes=str(exc))
            walls, preview = [], None
    else:
        walls, preview, diagnostics = detect_walls_opencv(image, meters_per_pixel, mode=data.get("mode", "balanced"))

    return {"walls": walls, "preview": preview, "diagnostics": diagnostics}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
