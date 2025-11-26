import numpy as np

from backend.segformer_detector import SegformerWallDetector


def test_segformer_detector_handles_missing_model(tmp_path):
    # Point to a non-existent model to ensure graceful diagnostics
    detector = SegformerWallDetector(onnx_path=tmp_path / "missing.onnx")
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    result = detector.detect(image_bgr=image, meters_per_pixel=0.05)

    assert result.walls == []
    assert result.diagnostics.get("detector") == "ml-v2"
    assert "notes" in result.diagnostics
