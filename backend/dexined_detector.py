import os
import logging
from typing import Dict, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None

from backend.config_wall_detection import (
    DEFAULT_DEXINED_MODEL_PATH,
    DEFAULT_DEVICE,
)
from backend.ml_wall_detection import WallDetector, WallDetectionResult
from backend.vectorizer import vectorize_from_edge_map

logger = logging.getLogger(__name__)


class DexinedWallDetector(WallDetector):
    def __init__(
        self,
        onnx_path: str = DEFAULT_DEXINED_MODEL_PATH,
        device: str = DEFAULT_DEVICE,
    ) -> None:
        self.onnx_path = onnx_path
        self.device = device

        self.session = None
        self.input_name = None
        self.output_name = None
        self.fixed_hw: Tuple[int, int] | None = None
        self.model_loaded = False
        self.load_error = None

        if ort is None:
            self.load_error = "onnxruntime not installed"
            return

        if not os.path.exists(self.onnx_path):
            self.load_error = f"DexiNed ONNX model not found at {self.onnx_path}"
            return

        providers = ["CPUExecutionProvider"]
        if self.device.lower() == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        try:
            self.session = ort.InferenceSession(self.onnx_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            shape = self.session.get_inputs()[0].shape
            if len(shape) == 4 and all(isinstance(v, int) for v in shape[2:]):
                self.fixed_hw = (int(shape[2]), int(shape[3]))
            self.model_loaded = True
        except Exception as exc:  # pragma: no cover - defensive
            self.load_error = str(exc)
            logger.warning("Failed to load DexiNed ONNX session: %s", exc)

    def _preprocess(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        h, w = image_bgr.shape[:2]
        target_h, target_w = self.fixed_hw or (h, w)
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if (target_w, target_h) != (w, h):
            img_rgb = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img_float = img_rgb.astype(np.float32) / 255.0
        chw = np.transpose(img_float, (2, 0, 1))
        return chw[np.newaxis, ...], (h, w)

    def detect(self, image_bgr: np.ndarray, meters_per_pixel: float) -> WallDetectionResult:
        if not self.model_loaded or self.session is None:
            diagnostics: Dict[str, object] = {
                "detector": "ml-dexined",
                "modelName": "dexined",
                "segmentCount": 0,
                "notes": self.load_error or "DexiNed model not loaded",
            }
            return WallDetectionResult(walls=[], overlay_png_base64=None, diagnostics=diagnostics)

        try:
            input_tensor, original_hw = self._preprocess(image_bgr)
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            edge_map = outputs[0]
            edge_map = edge_map.squeeze()
            if edge_map.ndim == 3:
                edge_map = edge_map[0, :, :]

            if edge_map.shape != original_hw:
                edge_map = cv2.resize(edge_map, (original_hw[1], original_hw[0]), interpolation=cv2.INTER_CUBIC)

            edge_map = edge_map.astype(np.float32)
            edge_map -= edge_map.min()
            max_val = edge_map.max() if edge_map.max() > 0 else 1.0
            edge_map /= max_val

            walls, overlay = vectorize_from_edge_map(
                edge_map=edge_map,
                original_bgr=image_bgr,
                meters_per_pixel=meters_per_pixel,
            )

            diagnostics: Dict[str, object] = {
                "detector": "ml-dexined",
                "modelName": "dexined",
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
            diagnostics = {
                "detector": "ml-dexined",
                "modelName": "dexined",
                "segmentCount": 0,
                "error": str(exc),
                "notes": "DexiNed detector failed",
            }
            logger.exception("DexiNed detection failed: %s", exc)
            return WallDetectionResult(walls=[], overlay_png_base64=None, diagnostics=diagnostics)
