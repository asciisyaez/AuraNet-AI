"""
AuraNet-AI Backend - State-of-the-Art Wall Detection v2
Focuses on detecting STRUCTURAL walls only, ignoring text, annotations, and rulers.

Key Strategy:
1. Normalize contrast and isolate the main floor-plan region (drop margins/rulers).
2. Build a wall mask that favors thick, high-fill components and rejects text/annotations.
3. Use a Line Segment Detector (LSD) over the cleaned mask for high-precision segments.
4. Score segments by wall support (mask overlap + stroke width), discard ruler/text candidates.
5. Merge collinear segments to minimize wall count while maximizing coverage.
"""
import cv2
import numpy as np
import base64
import json
import asyncio
import time
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Dict
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from backend.ml_wall_detection import get_detector, WallDetectionResult

app = FastAPI(title="AuraNet-AI Backend - Wall Detection v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Data Models
# =============================================================================

class WallMetadata(BaseModel):
    color: str = "#475569"


class Wall(BaseModel):
    id: str
    x1: float
    y1: float
    x2: float
    y2: float
    material: str = "Concrete"
    attenuation: float = 12.0
    thickness: int = 10
    height: float = 3.0
    elevation: float = 0.0
    metadata: WallMetadata = WallMetadata()


class DetectionPreview(BaseModel):
    overlay: Optional[str] = None
    wall_count: int = 0
    processing_ms: Optional[int] = None


class DetectionResponse(BaseModel):
    walls: List[Wall]
    preview: Optional[DetectionPreview] = None
    overlayPngBase64: Optional[str] = None
    diagnostics: Optional[Dict[str, Any]] = None


class DetectWallsRequest(BaseModel):
    image: str
    metersPerPixel: float = 0.05
    detector: Optional[str] = None
    sessionId: Optional[str] = "default"


@dataclass
class WallSegment:
    x1: float
    y1: float
    x2: float
    y2: float
    thickness: float = 8.0
    
    @property
    def length(self) -> float:
        return np.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)
    
    @property
    def angle_deg(self) -> float:
        angle = math.degrees(math.atan2(self.y2 - self.y1, self.x2 - self.x1))
        return (angle + 180) % 180  # normalized to [0, 180)
    
    @property
    def is_horizontal(self) -> bool:
        return abs(self.y2 - self.y1) < abs(self.x2 - self.x1) * 0.15
    
    @property
    def is_vertical(self) -> bool:
        return abs(self.x2 - self.x1) < abs(self.y2 - self.y1) * 0.15
    
    @property
    def is_orthogonal(self) -> bool:
        return self.is_horizontal or self.is_vertical


# =============================================================================
# Image Utilities
# =============================================================================

def decode_image(data_url: str) -> np.ndarray:
    if ',' in data_url:
        data_url = data_url.split(',')[1]
    image_bytes = base64.b64decode(data_url)
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def encode_image(image: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', image)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{base64_str}"


# =============================================================================
# Core Wall Detection - LSD + Mask Scoring Approach
# =============================================================================

def find_floor_plan_region(gray: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find the main floor plan region, excluding margins with text/annotations.
    Returns (x, y, w, h) of the region of interest.
    """
    h, w = gray.shape
    
    # Threshold to find all dark elements
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return (0, 0, w, h)
    
    # Find the largest rectangular contour (likely the floor plan boundary)
    largest_area = 0
    best_rect = (0, 0, w, h)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area and area > (w * h * 0.1):  # At least 10% of image
            x, y, cw, ch = cv2.boundingRect(contour)
            # Check if it's reasonably rectangular
            if cw > w * 0.3 and ch > h * 0.3:
                largest_area = area
                best_rect = (x, y, cw, ch)
    
    # Add some margin
    x, y, rw, rh = best_rect
    margin = int(min(rw, rh) * 0.02)
    x = max(0, x - margin)
    y = max(0, y - margin)
    rw = min(w - x, rw + 2 * margin)
    rh = min(h - y, rh + 2 * margin)
    
    return (x, y, rw, rh)


def normalize_grayscale(gray: np.ndarray) -> np.ndarray:
    """Equalize contrast while preserving edges for line detection."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    # Bilateral filter keeps edges sharp
    return cv2.bilateralFilter(eq, 7, 50, 50)


def build_wall_mask(gray: np.ndarray, min_thickness: int,
                    region: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Create a binary mask that emphasizes walls and suppresses text/rulers.
    - Adaptive thresholding + closing for continuity
    - Connected component filtering by area, aspect, and fill
    - Restrict to detected floor-plan region
    """
    h, w = gray.shape
    block_size = max(31, int(0.012 * min(h, w)) | 1)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C=3
    )
    
    # Close small gaps, remove speckle noise
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, close_kernel, iterations=1)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = max(180, (h * w) * 0.00005)
    max_area = (h * w) * 0.35
    
    cleaned = np.zeros_like(binary)
    rx, ry, rw, rh = region
    margin_x = int(rw * 0.02)
    margin_y = int(rh * 0.02)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        
        if area < min_area or area > max_area:
            continue
        
        bounding_area = comp_w * comp_h
        fill_ratio = area / bounding_area if bounding_area else 0
        aspect = max(comp_w, comp_h) / (min(comp_w, comp_h) + 1e-3)
        min_dim = min(comp_w, comp_h)
        
        # Reject compact, low-fill components (text/numbers), and very thin rulers
        if fill_ratio < 0.18 and area < min_area * 6:
            continue
        if aspect > 8 and min_dim < max(2, min_thickness * 0.6):
            continue
        if min_dim < max(2, min_thickness * 0.5) and aspect > 4:
            continue
        
        # Keep if inside the floor-plan bounding box (with small padding)
        center_x = left + comp_w / 2
        center_y = top + comp_h / 2
        if not (rx - margin_x <= center_x <= rx + rw + margin_x and
                ry - margin_y <= center_y <= ry + rh + margin_y):
            continue
        
        cleaned[labels == i] = 255
    
    return cleaned


def line_support_ratio(mask: np.ndarray, segment: WallSegment, thickness: int,
                       dist_map: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Compute how much of the drawn segment overlaps with the wall mask and
    estimate stroke width using a distance transform.
    """
    h, w = mask.shape
    line_mask = np.zeros_like(mask)
    cv2.line(
        line_mask,
        (int(segment.x1), int(segment.y1)),
        (int(segment.x2), int(segment.y2)),
        255,
        thickness=thickness
    )
    
    overlap = cv2.countNonZero(cv2.bitwise_and(line_mask, mask))
    total = cv2.countNonZero(line_mask)
    support = (overlap / total) if total > 0 else 0.0
    
    # Distance transform for thickness estimation
    if dist_map is None:
        dist_map = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    ys, xs = np.nonzero(line_mask)
    sampled = dist_map[ys, xs] if len(xs) else np.array([0], dtype=np.float32)
    est_thickness = float(np.median(sampled) * 2.0) if sampled.size else 0.0
    
    return support, est_thickness


def detect_line_segments(mask: np.ndarray) -> List[WallSegment]:
    """Use OpenCV's LSD to find high-quality line segments."""
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    lines, _, _, _ = lsd.detect(mask)
    segments: List[WallSegment] = []
    
    if lines is None:
        return segments
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        segments.append(WallSegment(float(x1), float(y1), float(x2), float(y2)))
    
    return segments


def filter_segments(segments: List[WallSegment],
                    mask: np.ndarray,
                    min_wall_px: int,
                    min_thickness_px: int,
                    region: Tuple[int, int, int, int]) -> List[WallSegment]:
    """
    Score LSD segments against the wall mask to suppress rulers/text and keep
    only structural walls.
    """
    rx, ry, rw, rh = region
    inner_margin_x = rw * 0.02
    inner_margin_y = rh * 0.02
    dist_map = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    
    filtered: List[WallSegment] = []
    for seg in segments:
        if seg.length < min_wall_px * 0.7:
            continue
        
        center_x = (seg.x1 + seg.x2) / 2
        center_y = (seg.y1 + seg.y2) / 2
        if not (rx - inner_margin_x <= center_x <= rx + rw + inner_margin_x and
                ry - inner_margin_y <= center_y <= ry + rh + inner_margin_y):
            continue
        
        angle = seg.angle_deg
        # Favor orthogonal walls; allow diagonals only with strong support
        orthogonal = (abs(angle - 0) < 15 or abs(angle - 90) < 15 or abs(angle - 180) < 15)
        
        thickness_draw = max(2, int(min_thickness_px * 0.6))
        support, est_thickness = line_support_ratio(mask, seg, thickness_draw, dist_map)
        
        if support < (0.55 if orthogonal else 0.70):
            continue
        if est_thickness < min_thickness_px * 0.6:
            continue
        
        # Discard likely rulers: long thin lines hugging margins
        near_left = center_x < rx + rw * 0.05
        near_right = center_x > rx + rw * 0.95
        near_top = center_y < ry + rh * 0.05
        near_bottom = center_y > ry + rh * 0.95
        if (near_left or near_right or near_top or near_bottom) and est_thickness < min_thickness_px * 1.2:
            continue
        
        seg.thickness = max(est_thickness, float(min_thickness_px))
        filtered.append(seg)
    
    return filtered


def merge_collinear_walls(walls: List[WallSegment],
                          gap_threshold: float = 30,
                          alignment_threshold: float = 10) -> List[WallSegment]:
    """
    Merge walls that are collinear and close together (horizontal or vertical).
    """
    if len(walls) <= 1:
        return walls
    
    h_walls = [w for w in walls if w.is_horizontal]
    v_walls = [w for w in walls if w.is_vertical]
    merged: List[WallSegment] = []
    
    # Merge horizontal walls
    h_walls.sort(key=lambda w: (round(w.y1 / alignment_threshold), w.x1))
    i = 0
    while i < len(h_walls):
        current = h_walls[i]
        x1, y1, x2 = current.x1, current.y1, current.x2
        thickness = current.thickness
        j = i + 1
        while j < len(h_walls):
            next_wall = h_walls[j]
            if abs(next_wall.y1 - y1) < alignment_threshold and next_wall.x1 - x2 < gap_threshold:
                x2 = max(x2, next_wall.x2)
                thickness = max(thickness, next_wall.thickness)
                j += 1
                continue
            break
        merged.append(WallSegment(x1, y1, x2, y1, thickness))
        i = j
    
    # Merge vertical walls
    v_walls.sort(key=lambda w: (round(w.x1 / alignment_threshold), w.y1))
    i = 0
    while i < len(v_walls):
        current = v_walls[i]
        x1, y1, y2 = current.x1, current.y1, current.y2
        thickness = current.thickness
        j = i + 1
        while j < len(v_walls):
            next_wall = v_walls[j]
            if abs(next_wall.x1 - x1) < alignment_threshold and next_wall.y1 - y2 < gap_threshold:
                y2 = max(y2, next_wall.y2)
                thickness = max(thickness, next_wall.thickness)
                j += 1
                continue
            break
        merged.append(WallSegment(x1, y1, x1, y2, thickness))
        i = j
    
    return merged


def snap_to_grid(walls: List[WallSegment], grid_size: float = 5) -> List[WallSegment]:
    """
    Snap wall endpoints to a grid for cleaner alignment.
    """
    snapped = []
    for wall in walls:
        x1 = round(wall.x1 / grid_size) * grid_size
        y1 = round(wall.y1 / grid_size) * grid_size
        x2 = round(wall.x2 / grid_size) * grid_size
        y2 = round(wall.y2 / grid_size) * grid_size
        snapped.append(WallSegment(x1, y1, x2, y2, wall.thickness))
    return snapped


def filter_walls_by_region(walls: List[WallSegment], 
                           region: Tuple[int, int, int, int],
                           margin_percent: float = 0.05) -> List[WallSegment]:
    """
    Filter walls to only those within the main floor plan region.
    Excludes walls in the margins where text/scale bars typically are.
    """
    x, y, w, h = region
    margin_x = w * margin_percent
    margin_y = h * margin_percent
    
    # Define the inner region
    inner_x1 = x + margin_x
    inner_y1 = y + margin_y
    inner_x2 = x + w - margin_x
    inner_y2 = y + h - margin_y
    
    filtered = []
    for wall in walls:
        # Check if wall is mostly within the inner region
        wall_center_x = (wall.x1 + wall.x2) / 2
        wall_center_y = (wall.y1 + wall.y2) / 2
        
        if (inner_x1 <= wall_center_x <= inner_x2 and 
            inner_y1 <= wall_center_y <= inner_y2):
            filtered.append(wall)
    
    return filtered


def deduplicate_walls(walls: List[WallSegment], threshold: float = 10) -> List[WallSegment]:
    """
    Remove duplicate or very similar walls.
    """
    if len(walls) <= 1:
        return walls
    
    unique = []
    for wall in walls:
        is_duplicate = False
        for existing in unique:
            # Check if endpoints are very close
            d1 = np.sqrt((wall.x1 - existing.x1)**2 + (wall.y1 - existing.y1)**2)
            d2 = np.sqrt((wall.x2 - existing.x2)**2 + (wall.y2 - existing.y2)**2)
            d3 = np.sqrt((wall.x1 - existing.x2)**2 + (wall.y1 - existing.y2)**2)
            d4 = np.sqrt((wall.x2 - existing.x1)**2 + (wall.y2 - existing.y1)**2)
            
            if (d1 < threshold and d2 < threshold) or (d3 < threshold and d4 < threshold):
                is_duplicate = True
                # Keep the longer one
                if wall.length > existing.length:
                    unique.remove(existing)
                    unique.append(wall)
                break
        
        if not is_duplicate:
            unique.append(wall)
    
    return unique


# =============================================================================
# Main Detection Pipeline
# =============================================================================

class DetectionState:
    current_progress: Dict[str, dict] = {}

detection_state = DetectionState()


def _segments_to_walls(segments: List[Tuple[float, float, float, float]]) -> List[Wall]:
    walls: List[Wall] = []
    for i, (x1, y1, x2, y2) in enumerate(segments):
        walls.append(Wall(
            id=f"wall-{i}",
            x1=float(x1),
            y1=float(y1),
            x2=float(x2),
            y2=float(y2),
            material="Concrete",
            attenuation=12.0,
            thickness=10,
            height=3.0,
            elevation=0.0,
            metadata=WallMetadata(color="#475569")
        ))
    return walls


def detect_walls_v2(image: np.ndarray, meters_per_pixel: float,
                    progress_callback=None) -> Tuple[List[Wall], np.ndarray, int, Dict]:
    """
    State-of-the-art wall detection focusing on structural walls only.
    """
    start_time = time.time()
    stats = {}
    
    def report(stage: str, percent: int, message: str):
        if progress_callback:
            progress_callback(stage, percent, message)
    
    h, w = image.shape[:2]
    stats['dimensions'] = (w, h)
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Calculate minimum wall dimensions - tuned for stateful coverage
    min_wall_m = 0.6  # Slightly longer to avoid text but still permit interior spans
    min_wall_px = max(int(min_wall_m / meters_per_pixel), 18)
    min_thickness_px = max(int(0.09 / meters_per_pixel), 3)  # 9cm minimum thickness
    
    report("preprocessing", 10, "Finding floor plan region...")
    region = find_floor_plan_region(gray)
    stats['region'] = region
    report("preprocessing", 20, f"Floor plan region: {region[2]}x{region[3]} px")
    
    # Normalize and build a mask favoring structural walls
    report("preprocessing", 30, "Normalizing contrast and cleaning noise...")
    norm_gray = normalize_grayscale(gray)
    wall_mask = build_wall_mask(norm_gray, min_thickness=min_thickness_px, region=region)
    stats['mask_coverage'] = int(cv2.countNonZero(wall_mask))
    
    if stats['mask_coverage'] == 0:
        processing_ms = int((time.time() - start_time) * 1000)
        return [], image.copy(), processing_ms, stats
    
    # Detect line segments over the cleaned mask
    report("line_detection", 55, "Detecting high-confidence wall segments...")
    segments = detect_line_segments(wall_mask)
    stats['raw_segments'] = len(segments)
    
    filtered_segments = filter_segments(
        segments, wall_mask, min_wall_px, min_thickness_px, region
    )
    stats['filtered_segments'] = len(filtered_segments)
    report("line_detection", 70, f"Filtered to {len(filtered_segments)} supported segments")
    
    if not filtered_segments:
        processing_ms = int((time.time() - start_time) * 1000)
        return [], image.copy(), processing_ms, stats
    
    # Merge and clean
    report("merging", 80, "Merging collinear spans...")
    merged = merge_collinear_walls(
        filtered_segments,
        gap_threshold=max(40, min_wall_px * 0.4),
        alignment_threshold=max(12, min_thickness_px)
    )
    stats['after_merge'] = len(merged)
    
    report("finalization", 90, "Snapping and deduplicating...")
    snapped = snap_to_grid(merged, grid_size=3)
    unique = deduplicate_walls(snapped, threshold=12)
    final = [w for w in unique if w.length >= min_wall_px * 0.6]
    stats['final_count'] = len(final)
    report("finalization", 98, f"Finalized {len(final)} walls")
    
    # Create Wall objects
    walls = []
    for i, seg in enumerate(final):
        wall_thickness = max(6, min(20, int(seg.thickness)))
        walls.append(Wall(
            id=f"wall-{i}",
            x1=float(seg.x1), y1=float(seg.y1),
            x2=float(seg.x2), y2=float(seg.y2),
            material="Concrete",
            attenuation=12.0,
            thickness=wall_thickness,
            height=3.0,
            elevation=0.0,
            metadata=WallMetadata(color="#475569")
        ))
    
    # Create overlay
    overlay = image.copy()
    
    # Draw walls
    for seg in final:
        color = (0, 200, 80)  # Green
        thickness = max(2, int(seg.thickness / 2))
        cv2.line(overlay, 
                 (int(seg.x1), int(seg.y1)), 
                 (int(seg.x2), int(seg.y2)), 
                 color, thickness, cv2.LINE_AA)
    
    # Draw region boundary
    rx, ry, rw, rh = region
    cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), (255, 150, 0), 2)
    
    overlay = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
    
    processing_ms = int((time.time() - start_time) * 1000)
    stats['processing_ms'] = processing_ms
    
    report("finalization", 100, f"Detection complete: {len(walls)} walls in {processing_ms}ms")
    
    return walls, overlay, processing_ms, stats


# =============================================================================
# API Endpoints
# =============================================================================

async def progress_generator(session_id: str):
    last_update = None
    timeout = 120
    start = time.time()
    
    while time.time() - start < timeout:
        if session_id in detection_state.current_progress:
            update = detection_state.current_progress[session_id]
            if update != last_update:
                last_update = update.copy()
                data = json.dumps(update)
                yield f"data: {data}\n\n"
                
                if update.get('percent', 0) >= 100:
                    break
        
        await asyncio.sleep(0.1)
    
    if session_id in detection_state.current_progress:
        del detection_state.current_progress[session_id]


@app.get("/api/detection-progress/{session_id}")
async def detection_progress(session_id: str):
    return StreamingResponse(
        progress_generator(session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.post("/api/detect-walls-base64", response_model=DetectionResponse)
async def detect_walls_base64_endpoint(req: DetectWallsRequest):
    start_time = time.time()
    detector_name = (req.detector or "ml")
    session_id = req.sessionId or "default"

    detection_state.current_progress[session_id] = {
        "stage": "initializing",
        "percent": 5,
        "message": f"Preparing detector {detector_name}",
    }

    if not req.image:
        detection_state.current_progress[session_id] = {
            "stage": "error",
            "percent": 100,
            "message": "No image supplied",
        }
        return DetectionResponse(walls=[], diagnostics={"detector": detector_name, "notes": "Missing image"})

    image = decode_image(req.image)
    if image is None:
        detection_state.current_progress[session_id] = {
            "stage": "error",
            "percent": 100,
            "message": "Invalid image data",
        }
        return DetectionResponse(walls=[], diagnostics={"detector": detector_name, "notes": "Invalid image"})

    try:
        detector = get_detector(detector_name)  # type: ignore[arg-type]
    except ValueError:
        detector = get_detector("ml")
        detector_name = "ml"

    try:
        detection_state.current_progress[session_id] = {
            "stage": "detecting",
            "percent": 40,
            "message": f"Running {detector_name}",
        }
        result: WallDetectionResult = detector.detect(
            image_bgr=image,
            meters_per_pixel=req.metersPerPixel,
        )
    except Exception as exc:
        detection_state.current_progress[session_id] = {
            "stage": "error",
            "percent": 100,
            "message": "Wall detector failed",
        }
        return DetectionResponse(
            walls=[],
            diagnostics={
                "detector": detector_name,
                "error": str(exc),
                "notes": "Wall detector failed; see server logs.",
            },
        )

    walls = _segments_to_walls(result.walls)
    processing_ms = int((time.time() - start_time) * 1000)
    preview = DetectionPreview(
        overlay=result.overlay_png_base64,
        wall_count=len(walls),
        processing_ms=processing_ms,
    )
    diagnostics = result.diagnostics or {}
    diagnostics.setdefault("detector", detector_name)

    detection_state.current_progress[session_id] = {
        "stage": "completed",
        "percent": 100,
        "message": "Wall detection complete",
    }

    return DetectionResponse(
        walls=walls,
        preview=preview,
        overlayPngBase64=result.overlay_png_base64,
        diagnostics=diagnostics,
    )


@app.post("/api/detect-walls", response_model=DetectionResponse)
async def detect_walls_endpoint(
    file: UploadFile = File(...), 
    metersPerPixel: float = Form(...),
    sessionId: str = Form("default"),
    detector: Optional[str] = Form(None),
):
    start_time = time.time()
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return DetectionResponse(walls=[])
    
    detector_name = detector or "ml"
    detection_state.current_progress[sessionId] = {
        "stage": "initializing",
        "percent": 5,
        "message": f"Preparing detector {detector_name}",
    }

    try:
        detector_impl = get_detector(detector_name)  # type: ignore[arg-type]
    except ValueError:
        detector_impl = get_detector("ml")
        detector_name = "ml"

    try:
        detection_state.current_progress[sessionId] = {
            "stage": "detecting",
            "percent": 40,
            "message": f"Running {detector_name}",
        }
        result: WallDetectionResult = detector_impl.detect(
            image_bgr=image,
            meters_per_pixel=metersPerPixel,
        )
    except Exception as exc:
        detection_state.current_progress[sessionId] = {
            "stage": "error",
            "percent": 100,
            "message": "Wall detector failed",
        }
        return DetectionResponse(
            walls=[],
            diagnostics={
                "detector": detector_name,
                "error": str(exc),
                "notes": "Wall detector failed; see server logs.",
            },
        )

    walls = _segments_to_walls(result.walls)
    processing_ms = int((time.time() - start_time) * 1000)
    preview = DetectionPreview(
        overlay=result.overlay_png_base64,
        wall_count=len(walls),
        processing_ms=processing_ms,
    )
    diagnostics = result.diagnostics or {}
    diagnostics.setdefault("detector", detector_name)

    detection_state.current_progress[sessionId] = {
        "stage": "completed",
        "percent": 100,
        "message": "Wall detection complete",
    }

    return DetectionResponse(
        walls=walls,
        preview=preview,
        overlayPngBase64=result.overlay_png_base64,
        diagnostics=diagnostics,
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0-structural"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
