# Implementation PRD: Wall Detection v2 (Server-Side, SOTA)

---

## 0. Context for Coding Agent

You are working in a Vite + React + FastAPI repo.

- Frontend: React/TypeScript under `src/`, with API helpers in `services/` (including `wallDetection`).
- Backend: FastAPI wall-detection service under `backend/` (`main.py` / `main_v2.py`, `requirements.txt`).

There is an **experimental ML detector** that uses HED + Hough, exposed via the same `/api/detect-walls-base64` route and selected via `detector: "ml"` in the request body.

We want to replace that baseline with **Wall Detection v2**, a significantly more accurate ML pipeline that outputs vector walls for Wi-Fi planning (AuraNet Planner).

---

## 1. Objectives & Scope

### 1.1 Goals

1. **Accuracy:** Achieve ≥ 98% **wall IoU** and ≥ 0.95 **structural F1** on a curated floor-plan validation set using the new pipeline and `wall_benchmark.py`.
2. **API-compatible:** Keep the existing `/api/detect-walls-base64` contract unchanged (request/response shape, including `detector: "ml"` behavior), while introducing a new detector name for v2.
3. **Pluggable models:** Implement a modular ML detector that can swap between:
    - Edge / wireframe models (e.g. DexiNed, HAWP) providing edge maps or line segments.
    - Segmentation models (e.g. SegFormer-B2, Mask R-CNN) providing wall probability maps.
4. **Reproducible deployment:** Use ONNX or TorchScript exports under `backend/models/` with documented provenance, hashes, and configuration.

### 1.2 Non-Goals

- Detecting furniture, labels, or non-wall objects (though the pipeline should robustly ignore them).
- Reworking the React UI or project management UX beyond what’s needed to call the new detector.
- Implementing full data labeling workflows; assume training datasets can be mounted at a given path.

---

## 2. User-Visible Behavior

### 2.1 API Contract (unchanged)

**Endpoint:**

```
POST /api/detect-walls-base64
Content-Type: application/json

```

**Request body (existing):**

```json
{
  "image": "data:image/png;base64,...",
  "metersPerPixel": 0.05,
  "detector": "ml"       // or "ml-v2", see below
}

```

**New behavior:**

- `detector` omitted or `"detector": "ml"`
    
    → Keep existing HED-based behavior *for now* (backwards compatibility).
    
- `detector: "ml-v2"`
    
    → Run the new high-accuracy pipeline described in this PRD.
    
- (Optional) `detector: "ml-dexined"` / `"ml-segformer"`
    
    → Allow direct selection of specific model variants for experimentation.
    

**Response (must remain compatible):**

```json
{
  "walls": [                // list of wall segments
    [x1, y1, x2, y2],
    ...
  ],
  "overlayPngBase64": "data:image/png;base64,...",  // debug overlay
  "diagnostics": {
    "detector": "ml-v2",
    "modelName": "segformer-b2-cubicasa5k",
    "edgeDensity": 0.23,             // optional
    "segmentCount": 415,
    "notes": "OK"                    // errors, model load status, etc.
  }
}

```

- Ensure `"walls"` has the same type/shape and semantics as before (pixel coordinates in image space).
- If the model/weights are missing or corrupted, return `walls: []` and set a meaningful `diagnostics.notes` message, matching current behavior.

---

## 3. Architecture Overview

### 3.1 Components

1. **Detector registry** (`backend/ml_wall_detection.py` – existing file, extend it):
    - A small abstraction that maps `detector` strings to concrete implementations.
2. **Model backends** (new) under `backend/ml_models/`:
    - `dexined_onnx.py` – edge detector.
    - `segformer_onnx.py` – wall segmentation model.
3. **Vectorizer** (existing or refactored):
    - Shared Hough + skeletonization logic that takes either:
        - an **edge map** (DexiNed/HAWP, HED baseline), or
        - a **wall mask / probability map** (SegFormer/Mask R-CNN).
4. **Benchmark harness** (`backend/tools/wall_benchmark.py` – new):
    - CLI script to run end-to-end detection on `{image}.png` + `{image}_mask.png` pairs and compute pixel IoU and structural F1.

### 3.2 Detector Types

- **Edge/wireframe detectors**:
    - Input: RGB image.
    - Output: `edge_map: np.ndarray[H,W]` (float32, 0–1).
    - Usage: Replace `hed_edges` in the existing vectorizer pipeline.
- **Segmentation detectors**:
    - Input: RGB image.
    - Output: `wall_prob_map: np.ndarray[H,W]` (float32, 0–1).
    - Usage: Threshold, then **skeletonize + Hough** to keep API compatible.

---

## 4. Detailed Implementation Steps

### 4.1 Extend Detector Registry

**File:** `backend/ml_wall_detection.py` (already referenced in research note).

### 4.1.1 Define shared types

Add at the top:

```python
from dataclasses import dataclass
from typing import Dict, Protocol, Tuple, List, Literal, Optional
import numpy as np

DetectorName = Literal["ml", "ml-v2", "ml-dexined", "ml-segformer"]

WallSegment = Tuple[float, float, float, float]

@dataclass
class WallDetectionResult:
    walls: List[WallSegment]
    overlay_png_base64: Optional[str]
    diagnostics: Dict[str, object]

```

Define a protocol/interface:

```python
class WallDetector(Protocol):
    def detect(self, image_bgr: np.ndarray, meters_per_pixel: float) -> WallDetectionResult:
        ...

```

### 4.1.2 Implement registry

Still in `ml_wall_detection.py`:

```python
_DETECTORS: Dict[DetectorName, WallDetector] = {}

def register_detector(name: DetectorName, detector: WallDetector) -> None:
    _DETECTORS[name] = detector

def get_detector(name: DetectorName) -> WallDetector:
    # Backwards compatibility: "ml" returns legacy HED detector
    if name not in _DETECTORS:
        raise ValueError(f"Unknown wall detector: {name}")
    return _DETECTORS[name]

```

Register detectors at module import time:

```python
from .hed_detector import HedWallDetector
from .dexined_detector import DexinedWallDetector
from .segformer_detector import SegformerWallDetector

register_detector("ml", HedWallDetector())             # existing path
register_detector("ml-dexined", DexinedWallDetector())
register_detector("ml-segformer", SegformerWallDetector())
register_detector("ml-v2", SegformerWallDetector())    # alias for default SOTA

```

> Note: Implement HedWallDetector wrapper around current HED + vectorizer logic so we can keep it as baseline.
> 

### 4.1.3 Wire FastAPI endpoint

**File:** `backend/main.py` or `backend/main_v2.py`.

Inside `/api/detect-walls-base64` handler:

```python
from .ml_wall_detection import get_detector, WallDetectionResult

@app.post("/api/detect-walls-base64")
async def detect_walls_base64(req: DetectWallsRequest) -> Dict[str, Any]:
    image_bgr = decode_base64_to_bgr(req.image)  # existing helper
    detector_name = req.detector or "ml"
    try:
        detector = get_detector(detector_name)
    except ValueError:
        detector = get_detector("ml")  # fallback to HED baseline

    try:
        result: WallDetectionResult = detector.detect(
            image_bgr=image_bgr,
            meters_per_pixel=req.metersPerPixel,
        )
    except Exception as exc:
        # Hard failure → empty walls, diagnostics
        return {
            "walls": [],
            "overlayPngBase64": None,
            "diagnostics": {
                "detector": detector_name,
                "error": str(exc),
                "notes": "Wall detector failed; see server logs."
            },
        }

    return {
        "walls": result.walls,
        "overlayPngBase64": result.overlay_png_base64,
        "diagnostics": result.diagnostics,
    }

```

Ensure the request model includes `detector: Optional[str]` (default `None`).

---

### 4.2 Implement SegFormer-based Detector (v2)

This is the recommended primary SOTA path (SegFormer-B2 fine-tuned on CubiCasa5k).

### 4.2.1 Model export (high-level)

Training/export details are in §5, but at runtime we expect an **ONNX** file:

- Path: `backend/models/segformer_b2_walls.onnx`
- Input tensor:
    - Name: `"input"`
    - Shape: `[1, 3, H, W]` (dynamic H,W allowed).
    - DType: `float32`.
- Output tensor:
    - Name: `"logits"`
    - Shape: `[1, 1, H, W]` (single wall class), values in logits or probabilities.

Validate and document this in a small `backend/models/README.md` with the model hash.

### 4.2.2 SegFormer detector class

**File:** `backend/segformer_detector.py`

Core responsibilities:

1. Load ONNX model using `onnxruntime`.
2. Preprocess incoming image (BGR np.ndarray) to model input.
3. Run inference, get `wall_prob_map`.
4. Threshold → wall mask.
5. Skeletonize + Hough to vector walls (via shared vectorizer).
6. Build `WallDetectionResult`.

Implementation outline:

```python
import base64
import cv2
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Tuple, Optional

from .ml_wall_detection import WallDetector, WallDetectionResult, WallSegment
from .vectorizer import vectorize_from_wall_mask

class SegformerWallDetector(WallDetector):
    def __init__(
        self,
        onnx_path: str = "backend/models/segformer_b2_walls.onnx",
        device: str = "cpu",
        prob_threshold: float = 0.5,
    ) -> None:
        providers = ["CPUExecutionProvider"]
        # TODO: add CUDA provider if available
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.prob_threshold = prob_threshold

    def detect(self, image_bgr: np.ndarray, meters_per_pixel: float) -> WallDetectionResult:
        h, w = image_bgr.shape[:2]

        # 1) Preprocess
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        # ImageNet normalization (if used during training)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_float - mean) / std
        chw = np.transpose(img_norm, (2, 0, 1))  # HWC → CHW
        inp = chw[np.newaxis, ...]  # [1,3,H,W]

        # 2) Inference
        logits = self.session.run(
            [self.output_name],
            {self.input_name: inp},
        )[0]  # [1,1,H,W]
        logits = logits[0, 0, :, :]
        wall_prob = 1 / (1 + np.exp(-logits))  # sigmoid if logits

        # 3) Threshold → mask
        wall_mask = (wall_prob >= self.prob_threshold).astype(np.uint8)  # 0/1

        # 4) Vectorize
        walls, debug_overlay = vectorize_from_wall_mask(
            wall_mask=wall_mask,
            original_bgr=image_bgr,
            meters_per_pixel=meters_per_pixel,
        )

        diagnostics: Dict[str, object] = {
            "detector": "ml-v2",
            "modelName": "segformer-b2-walls",
            "segmentCount": len(walls),
            "notes": "OK",
        }

        return WallDetectionResult(
            walls=walls,
            overlay_png_base64=debug_overlay,
            diagnostics=diagnostics,
        )

```

> Notes:
> 
> - Keep `prob_threshold` configurable (via env var or config module) to tune precision/recall later.
> - If inference throws, catch in endpoint (see 4.1.3).

---

### 4.3 Shared Vectorizer (Mask / Edge → Walls)

We need a **single** vectorization stack compatible with:

- HED baseline (`hed_edges`),
- DexiNed edges,
- SegFormer wall masks.

**File:** `backend/vectorizer.py`

### 4.3.1 From wall mask (SegFormer path)

Function signature:

```python
from typing import List, Tuple
import numpy as np

from .ml_wall_detection import WallSegment

def vectorize_from_wall_mask(
    wall_mask: np.ndarray,
    original_bgr: np.ndarray,
    meters_per_pixel: float,
) -> Tuple[List[WallSegment], str]:
    """
    Args:
        wall_mask: 2D uint8 array, values {0,1}, same HxW as original.
        original_bgr: original BGR image for overlay.
        meters_per_pixel: scaling factor; used to scale morphology and min lengths.
    Returns:
        walls: list of (x1,y1,x2,y2) in pixel coordinates (float).
        overlay_png_base64: debug overlay image with walls drawn.
    """

```

Implementation steps:

1. **Morphological cleanup (on mask):**
    - Compute kernel radius:
        - Example: `kernel_px = int(max(1, round(0.15 / meters_per_pixel)))`
            
            (i.e. ~15cm radius, clamped to [1, 7]).
            
    - Apply `cv2.morphologyEx` with `cv2.MORPH_CLOSE` to fill small gaps in walls.
    - Apply `cv2.morphologyEx` with `cv2.MORPH_OPEN` to remove tiny stray blobs.
2. **Skeletonization:**
    - Convert cleaned mask to 0/255 image.
    - Use `cv2.ximgproc.thinning` (`THINNING_ZHANGSUEN`) or an equivalent implementation to obtain a 1-pixel wide skeleton.
    - Remove spur pixels:
        - For each pixel, compute 8-connected neighbors.
        - Any skeleton pixel with exactly one neighbor **and** path length < `min_spur_px` (e.g. 0.3m) is removed.
3. **Line segment detection (Probabilistic Hough):**
    - Use `cv2.HoughLinesP` on skeleton with parameters based on `meters_per_pixel`:
        - `rho = 1`.
        - `theta = np.pi / 180`.
        - `threshold`: e.g. 20.
        - `minLineLength = max(5, int(0.5 / meters_per_pixel))` (min ~0.5m wall).
        - `maxLineGap = max(2, int(0.1 / meters_per_pixel))`.
    - For each Hough line `[x1,y1,x2,y2]`, cast to float and collect.
4. **Segment merging & snapping:**
    - Merge collinear, contiguous segments:
        - Two segments are mergeable if:
            - Angle difference < 3 degrees.
            - Distance between endpoints < `merge_gap_px` (e.g. 2–3 pixels).
    - Snap near-vertical / near-horizontal lines exactly to axis:
        - If angle in [−3°, +3°] → horizontal; snap `y` endpoints to their mean.
        - If angle in [87°, 93°] or [−93°, −87°] → vertical; snap `x` endpoints to mean.
    - Drop segments shorter than `min_wall_px` (e.g. 0.3m).
5. **Debug overlay:**
    - Create a copy of `original_bgr`.
    - Draw all walls as colored lines using `cv2.line`.
    - Encode overlay image to PNG + base64 as `"data:image/png;base64,..."`.
6. Return `walls` + `overlay_png_base64`.

### 4.3.2 From edge map (DexiNed / HED path)

Add:

```python
def vectorize_from_edge_map(
    edge_map: np.ndarray,       # float32 [0,1]
    original_bgr: np.ndarray,
    meters_per_pixel: float,
) -> Tuple[List[WallSegment], str]:
    """
    Steps:
    - Threshold edge_map → binary edges.
    - Morphologically close edges to wall blobs (similar kernel scheme as above).
    - Convert to wall_mask (e.g. by region growing / dilation).
    - Delegate to vectorize_from_wall_mask.
    """

```

- Implement logic similar to the existing HED pipeline:
    1. Threshold edges.
    2. Close edge gaps.
    3. Fill interiors to wall blobs (e.g. `cv2.dilate`, then `cv2.erode` to approximate thickness).
    4. Call `vectorize_from_wall_mask`.

This keeps **one** main vectorization implementation and avoids path divergence.

---

### 4.4 DexiNed-based Edge Detector (optional but recommended)

**Goal:** Implement a drop-in alternative edge detector that can reuse the vectorizer to improve thin-line recall.

**File:** `backend/dexined_detector.py`

- Expect ONNX model at `backend/models/dexined.onnx`, exported from the DexiNed weights.
- Preprocess:
    - Resize image to model’s input size if fixed (e.g. 512×512); otherwise support dynamic.
    - Normalize to [0,1] and apply any DexiNed-specific normalization (if known).
- Inference:
    - Run ONNX session, get edge logits or probabilities.
    - Upsample back to original H×W if input was resized.
- Postprocess:
    - Normalize to [0,1].
    - Call `vectorize_from_edge_map(edge_map, original_bgr, meters_per_pixel)`.

Detector skeleton:

```python
from .ml_wall_detection import WallDetector, WallDetectionResult
from .vectorizer import vectorize_from_edge_map

class DexinedWallDetector(WallDetector):
    def __init__(self, onnx_path: str = "backend/models/dexined.onnx") -> None:
        # init onnxruntime session, input/output names ...
        ...

    def detect(self, image_bgr: np.ndarray, meters_per_pixel: float) -> WallDetectionResult:
        edge_map = self._run_model(image_bgr)
        walls, overlay = vectorize_from_edge_map(edge_map, image_bgr, meters_per_pixel)
        diagnostics = {
            "detector": "ml-dexined",
            "modelName": "dexined",
            "segmentCount": len(walls),
            "notes": "OK",
        }
        return WallDetectionResult(walls=walls, overlay_png_base64=overlay, diagnostics=diagnostics)

```

---

### 4.5 HED Baseline Wrapper

Refactor existing HED logic (described in `ML_DETECTION.md`) into a class implementing `WallDetector`.

- Extract current steps:
    1. Lazy-download HED caffemodel to `backend/models`.
    2. Run HED → edge map.
    3. Morphologically close edges into blobs.
    4. Vectorize with probabilistic Hough.
- Replace the custom vectorization piece with `vectorize_from_edge_map` where possible.

**File:** `backend/hed_detector.py`

---

### 4.6 Config & Diagnostics

Add a simple config layer (can be just constants or environment-driven):

**File:** `backend/config_wall_detection.py`

```python
import os

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except ValueError:
        return default

SEGFORMER_PROB_THRESHOLD = env_float("SEGFORMER_PROB_THRESHOLD", 0.5)
MIN_WALL_METERS = env_float("WALL_MIN_WALL_METERS", 0.3)
MIN_SPUR_METERS = env_float("WALL_MIN_SPUR_METERS", 0.3)
MERGE_GAP_METERS = env_float("WALL_MERGE_GAP_METERS", 0.1)
CLOSE_RADIUS_METERS = env_float("WALL_CLOSE_RADIUS_METERS", 0.15)

```

Wire these constants into `vectorizer.py` and `SegformerWallDetector` so behavior can be tuned without code changes.

Diagnostics:

- Always populate:
    - `detector`, `modelName`, `segmentCount`.
- On failure (no walls, model not loaded, etc.), set:
    - `diagnostics.error`, `diagnostics.notes`.

---

## 5. Training & Export (Offline, but Code Included)

Even if training runs outside production, **include scripts** under `backend/training/` to make the process reproducible, aligned with MODEL_RESEARCH.

### 5.1 Directory Layout

- `backend/training/`
    - `datasets/`
        - `cubicasa5k.py` – dataset loader.
    - `train_segformer_b2_walls.py` – training loop.
    - `export_segformer_to_onnx.py` – export script.
    - `README.md` – how to run training & export.

### 5.2 Dataset Expectations

Follow `MODEL_RESEARCH` guidance: use CubiCasa5k or internal floorplan set with pixel-perfect wall masks.

Assume the dataset root looks like:

```
DATA_ROOT/
  images/
    sample_0001.png
    sample_0002.png
    ...
  masks/
    sample_0001_mask.png    # 0/255 wall mask
    sample_0002_mask.png

```

Where `_mask.png` corresponds to walls only.

`cubicasa5k.py` should:

- Accept `root`, `split` (“train”, “val”, “test”), `augment: bool`.
- Provide `__getitem__` returning:
    - `image: torch.FloatTensor [3,H,W]`
    - `mask: torch.LongTensor [H,W]` (0=background, 1=wall).

### 5.3 Training Script Skeleton

`train_segformer_b2_walls.py`:

- Use HuggingFace `SegformerForSemanticSegmentation` initialized from `segformer-b2` backbone.
- Hyperparameters (tunable, but provide defaults):
    - LR: `1e-4`, Optimizer: AdamW.
    - Batch size: 4.
    - Epochs: 60.
    - Loss: `CrossEntropyLoss(weight=[0.2, 0.8]) + DiceLoss` (or similar).
- Use standard augmentations:
    - Random horizontal/vertical flip.
    - Random rotation (multiples of 90°).
    - Color jitter (brightness/contrast).
- Split:
    - Train: 70%, Val: 15%, Test: 15% or follow dataset’s split.

Pseudo-outline:

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    ...
    args = parser.parse_args()

    train_ds = CubiCasaWalls(args.data_root, split="train", augment=True)
    val_ds = CubiCasaWalls(args.data_root, split="val", augment=False)
    ...

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=2,
    )
    ...

    for epoch in range(num_epochs):
        train_one_epoch(...)
        val_metrics = evaluate(...)
        # Save best checkpoint by val IoU

```

At the end, save final weights:

```python
torch.save(model.state_dict(), os.path.join(args.output_dir, "segformer_b2_walls.pt"))

```

### 5.4 Export to ONNX

`export_segformer_to_onnx.py`:

- Load model architecture & weights.
- Create dummy input `[1,3,512,512]`.
- Call `torch.onnx.export`:

```python
torch.onnx.export(
    model,
    dummy_input,
    "backend/models/segformer_b2_walls.onnx",
    input_names=["input"],
    output_names=["logits"],
    opset_version=13,
    dynamic_axes={"input": {2: "height", 3: "width"}, "logits": {2: "height", 3: "width"}},
)

```

- After export, validate with a few sample images and compare output with PyTorch to ensure parity.

Document this process in `backend/models/README.md`.

---

## 6. Benchmark Harness (`wall_benchmark.py`)

Implement the benchmark tool described in `MODEL_RESEARCH` as an actual script.

**File:** `backend/tools/wall_benchmark.py`

### 6.1 CLI Arguments

Use `argparse`:

- `-images-dir` (required): folder with `.png` images.
- `-masks-dir` (required): folder with `{name}_mask.png` wall masks.
- `-detector` (default `"ml-v2"`).
- `-meters-per-pixel` (default `0.05`).
- `-output-json` (default `wall_benchmark_results.json`).

### 6.2 Logic

For each `image_path`:

1. Derive `mask_path` as `masks_dir / (name + "_mask.png")`.
2. Load image and mask.
3. Call the same detection pipeline as the API, but locally:
    - Use `get_detector(detector_name)` and call `.detect(image_bgr, meters_per_pixel)`.
4. Rasterize predicted walls into a mask:
    - Create blank mask.
    - Draw each predicted segment as a line with width corresponding to wall thickness (e.g., 2–4 pixels).
5. Compute metrics vs ground truth mask:
    - Pixel TP/FP/FN.
    - `precision = TP / (TP+FP)`, `recall = TP / (TP+FN)`.
    - `IoU = TP / (TP+FP+FN)`.
    - F1 = `2 * precision * recall / (precision + recall)`.
6. Structural F1:
    - Vectorize **ground truth** mask using `vectorize_from_wall_mask`.
    - Match predicted vs ground truth segments:
        - Two segments match if:
            - Angle difference < 5°.
            - Projected overlap along their main axis > 50% of the shorter segment.
    - Compute structural precision/recall/F1 on segment level.
7. Accumulate per-image metrics and overall averages.

### 6.3 Output

Write JSON:

```json
{
  "detector": "ml-v2",
  "metersPerPixel": 0.05,
  "global": {
    "meanIoU": 0.983,
    "meanPixelF1": 0.991,
    "meanStructuralF1": 0.955
  },
  "perSample": [
    {
      "name": "sample_0001",
      "iou": 0.982,
      "pixelPrecision": 0.993,
      "pixelRecall": 0.989,
      "pixelF1": 0.991,
      "structuralF1": 0.953
    },
    ...
  ]
}

```

Use this to assert the **≥ 98% wall IoU and ≥ 0.95 structural F1** before promoting a model to production.

---

## 7. Testing & Definition of Done

### 7.1 Unit / Integration Tests

Even though there’s no Jest/pytest suite yet, add **Python tests** as a starting point:

- `backend/tests/test_vectorizer.py`
    - Synthetic images with known walls (e.g. simple rectangles), ensure the vectorizer recovers expected segments within a small tolerance.
- `backend/tests/test_segformer_detector.py`
    - Mock ONNX session returning known masks; verify `detect()` returns expected number of walls and diagnostics.
- `backend/tests/test_wall_benchmark.py`
    - Tiny dataset with trivial masks where expected metrics are known.

Update `README` (or AGENTS docs) to mention `pytest` as a way to run tests.

### 7.2 Manual QA

- Run backend locally (`uvicorn backend.main_v2:app --reload`).
- From the React app, test:
    1. Upload various floorplans (simple, complex, with text/scale bars).
    2. Toggle detectors via dev UI or by modifying the client to send `detector: "ml-v2"`.
    3. Visually inspect overlay & walls.
    4. Run Wi-Fi planning (AP placement & heatmap) to ensure walls are respected in ray tracing (as per AuraNet PRD wall attenuation logic).

### 7.3 Definition of Done

Wall Detection v2 is **done** when:

1. `detector: "ml-v2"` is wired through `/api/detect-walls-base64` and returns correct shape + diagnostics.
2. `SegformerWallDetector` + shared `vectorizer` are implemented, tested on synthetic inputs.
3. `DexinedWallDetector` + `HedWallDetector` wrappers exist and use the same vectorizer interface.
4. `wall_benchmark.py` runs and reports:
    - `global.meanIoU ≥ 0.98`,
    - `global.meanStructuralF1 ≥ 0.95`
        
        on the curated validation set configured by engineering.
        
5. Overlay visualization in the React app shows accurate, clean walls across a sample of real floorplans.
6. No regressions in existing HED-based `"ml"` behavior when explicitly selected.

---