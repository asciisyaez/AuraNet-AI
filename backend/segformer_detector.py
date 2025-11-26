import os
import logging
from typing import Dict

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - optional dependency
    ort = None

from backend.config_wall_detection import (
    DEFAULT_DEVICE,
    DEFAULT_SEGFORMER_MODEL_PATH,
    SEGFORMER_PROB_THRESHOLD,
)
from backend.ml_wall_detection import WallDetector, WallDetectionResult
from backend.vectorizer import vectorize_from_wall_mask

logger = logging.getLogger(__name__)


class SegformerWallDetector(WallDetector):
    def __init__(
        self,
        onnx_path: str = DEFAULT_SEGFORMER_MODEL_PATH,
        device: str = DEFAULT_DEVICE,
        prob_threshold: float = SEGFORMER_PROB_THRESHOLD,
    ) -> None:
        self.onnx_path = onnx_path
        self.device = device
        self.prob_threshold = prob_threshold

        self.session = None
        self.input_name = None
        self.output_name = None
        self.model_loaded = False
        self.load_error = None

        if ort is None:
            self.load_error = "onnxruntime not installed"
            return

        if not os.path.exists(self.onnx_path):
            self.load_error = f"SegFormer ONNX model not found at {self.onnx_path}"
            return

        providers = ["CPUExecutionProvider"]
        if self.device.lower() == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        try:
            self.session = ort.InferenceSession(self.onnx_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.model_loaded = True
        except Exception as exc:  # pragma: no cover - defensive
            self.load_error = str(exc)
            logger.warning("Failed to load SegFormer ONNX session: %s", exc)

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_float - mean) / std
        chw = np.transpose(img_norm, (2, 0, 1))
        return chw[np.newaxis, ...]

    def detect(self, image_bgr: np.ndarray, meters_per_pixel: float) -> WallDetectionResult:
        if not self.model_loaded or self.session is None:
            diagnostics: Dict[str, object] = {
                "detector": "ml-v2",
                "modelName": "segformer-b2-walls",
                "segmentCount": 0,
                "notes": self.load_error or "SegFormer model not loaded",
            }
            return WallDetectionResult(walls=[], overlay_png_base64=None, diagnostics=diagnostics)

        try:
            input_tensor = self._preprocess(image_bgr)
            logits = self.session.run(
                [self.output_name],
                {self.input_name: input_tensor},
            )[0]

            logits = logits[0, 0, :, :]
            wall_prob = 1.0 / (1.0 + np.exp(-logits))
            wall_mask = (wall_prob >= self.prob_threshold).astype(np.uint8)

            walls, overlay = vectorize_from_wall_mask(
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
                overlay_png_base64=overlay,
                diagnostics=diagnostics,
            )
        except Exception as exc:  # pragma: no cover - safety
            diagnostics = {
                "detector": "ml-v2",
                "modelName": "segformer-b2-walls",
                "segmentCount": 0,
                "error": str(exc),
                "notes": "SegFormer detector failed",
            }
            logger.exception("SegFormer detection failed: %s", exc)
            return WallDetectionResult(walls=[], overlay_png_base64=None, diagnostics=diagnostics)
