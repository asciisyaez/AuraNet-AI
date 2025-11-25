import cv2
import numpy as np
import base64
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

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

class DetectionResponse(BaseModel):
    walls: List[Wall]

def decode_image(data_url: str) -> np.ndarray:
    # Remove header if present (e.g., "data:image/png;base64,")
    if ',' in data_url:
        data_url = data_url.split(',')[1]
    image_bytes = base64.b64decode(data_url)
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def detect_walls_opencv(image: np.ndarray, meters_per_pixel: float) -> List[Wall]:
    # Convert to grayscale and enhance contrast to make faint blueprint lines pop
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Light blur to suppress speckle noise without destroying line edges
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Adaptive thresholding to handle varying lighting/contrast
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations to remove noise (text, furniture) and connect wall segments
    kernel_size = max(3, int(0.05 / meters_per_pixel))
    if kernel_size % 2 == 0:
        kernel_size += 1  # ensure odd size for symmetric operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    connected = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Dilate slightly to bridge gaps between short segments
    dilated = cv2.dilate(connected, kernel, iterations=1)

    # Edge detection (Canny)
    edges = cv2.Canny(dilated, 40, 120, apertureSize=3)
    
    # Hough Line Transform parameters
    min_wall_length_m = 0.5  # 50cm minimum wall length
    min_line_length_px = max(10, int(min_wall_length_m / meters_per_pixel))
    max_gap_px = max(5, int(0.3 / meters_per_pixel))  # 30cm gap allowed
    
    # Use higher threshold to reduce noise and duplicates
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60,
                            minLineLength=min_line_length_px, maxLineGap=max_gap_px)
    
    if lines is None:
        return []

    # Convert to list of tuples for easier processing
    all_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        all_lines.append((float(x1), float(y1), float(x2), float(y2)))
    
    # Merge similar/overlapping lines and extend slightly so detected walls fully cover drawn lines
    def lines_are_similar(line1, line2, dist_threshold, angle_threshold_deg=10):
        """Check if two lines are similar enough to merge."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Calculate angles
        angle1 = np.arctan2(y2 - y1, x2 - x1)
        angle2 = np.arctan2(y4 - y3, x4 - x3)
        
        # Normalize angles to [0, pi) - lines are bidirectional
        angle1 = angle1 % np.pi
        angle2 = angle2 % np.pi
        
        angle_diff = abs(angle1 - angle2)
        # Handle wrap-around at pi
        if angle_diff > np.pi / 2:
            angle_diff = np.pi - angle_diff
            
        if np.degrees(angle_diff) > angle_threshold_deg:
            return False
        
        # Check if lines are close to each other (perpendicular distance)
        # Calculate midpoints
        mid1 = ((x1 + x2) / 2, (y1 + y2) / 2)
        mid2 = ((x3 + x4) / 2, (y3 + y4) / 2)
        
        # Distance between midpoints
        midpoint_dist = np.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)
        
        # Also check distance from endpoints to the other line
        def point_to_line_dist(px, py, lx1, ly1, lx2, ly2):
            """Calculate perpendicular distance from point to line segment."""
            line_len_sq = (lx2 - lx1)**2 + (ly2 - ly1)**2
            if line_len_sq == 0:
                return np.sqrt((px - lx1)**2 + (py - ly1)**2)
            t = max(0, min(1, ((px - lx1) * (lx2 - lx1) + (py - ly1) * (ly2 - ly1)) / line_len_sq))
            proj_x = lx1 + t * (lx2 - lx1)
            proj_y = ly1 + t * (ly2 - ly1)
            return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
        
        # Check if any endpoint is close to the other line
        d1 = point_to_line_dist(x1, y1, x3, y3, x4, y4)
        d2 = point_to_line_dist(x2, y2, x3, y3, x4, y4)
        d3 = point_to_line_dist(x3, y3, x1, y1, x2, y2)
        d4 = point_to_line_dist(x4, y4, x1, y1, x2, y2)
        
        min_dist = min(d1, d2, d3, d4)
        
        # Lines are similar if they're close and parallel
        # Also check if they overlap in their extent
        len1 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        len2 = np.sqrt((x4-x3)**2 + (y4-y3)**2)
        max_len = max(len1, len2)
        
        return min_dist < dist_threshold and midpoint_dist < max_len + dist_threshold

    def merge_two_lines(line1, line2):
        """Merge two similar lines into one."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Collect all points
        points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        
        # Calculate the average direction
        dx1, dy1 = x2 - x1, y2 - y1
        dx2, dy2 = x4 - x3, y4 - y3
        
        # Ensure directions are aligned
        if dx1 * dx2 + dy1 * dy2 < 0:
            dx2, dy2 = -dx2, -dy2
        
        avg_dx = (dx1 + dx2) / 2
        avg_dy = (dy1 + dy2) / 2
        
        # Normalize direction
        length = np.sqrt(avg_dx**2 + avg_dy**2)
        if length == 0:
            return line1
        avg_dx /= length
        avg_dy /= length
        
        # Project all points onto the direction vector
        projections = []
        for px, py in points:
            proj = px * avg_dx + py * avg_dy
            projections.append((proj, px, py))
        
        # Find the extremes
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

    extension_px = max(6, int(0.15 / meters_per_pixel))

    # Merge overlapping lines iteratively
    dist_threshold = max(15, int(0.15 / meters_per_pixel))  # 15cm tolerance
    
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
    
    # Filter out very short lines after merging
    min_final_length = max(20, int(0.3 / meters_per_pixel))
    filtered_lines = []
    for x1, y1, x2, y2 in all_lines:
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if length >= min_final_length:
            filtered_lines.append(extend_line((x1, y1, x2, y2), extension_px))
    
    detected_walls = []
    for i, (x1, y1, x2, y2) in enumerate(filtered_lines):
        detected_walls.append(Wall(
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
            metadata=WallMetadata(color="#475569")
        ))
            
    return detected_walls

@app.post("/api/detect-walls", response_model=DetectionResponse)
async def detect_walls_endpoint(file: UploadFile = File(...), metersPerPixel: float = Form(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"walls": []}
        
    walls = detect_walls_opencv(image, metersPerPixel)
    return {"walls": walls}

@app.post("/api/detect-walls-base64", response_model=DetectionResponse)
async def detect_walls_base64_endpoint(data: dict):
    image_data = data.get("image")
    meters_per_pixel = data.get("metersPerPixel", 0.05)
    
    if not image_data:
        return {"walls": []}
        
    image = decode_image(image_data)
    walls = detect_walls_opencv(image, meters_per_pixel)
    return {"walls": walls}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
