import numpy as np
import cv2

from backend.ml_wall_detection import WallDetectionResult
from backend.tools import wall_benchmark


class _StubDetector:
    def __init__(self, walls):
        self._walls = walls

    def detect(self, image_bgr, meters_per_pixel):
        return WallDetectionResult(
            walls=self._walls,
            overlay_png_base64=None,
            diagnostics={"detector": "stub"},
        )


def test_wall_benchmark_runs_with_stub_detector(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    image_path = images_dir / "sample_0001.png"
    mask_path = masks_dir / "sample_0001_mask.png"

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.line(image, (5, 5), (58, 5), (255, 255, 255), 2)
    cv2.imwrite(str(image_path), image)

    mask = np.zeros((64, 64), dtype=np.uint8)
    cv2.line(mask, (5, 5), (58, 5), 255, 3)
    cv2.imwrite(str(mask_path), mask)

    stub_detector = _StubDetector([(5.0, 5.0, 58.0, 5.0)])
    monkeypatch.setattr(wall_benchmark, "get_detector", lambda name: stub_detector)

    results = wall_benchmark.run_benchmark(
        images_dir=images_dir,
        masks_dir=masks_dir,
        detector_name="stub",
        meters_per_pixel=0.05,
    )

    assert results["perSample"]
    assert results["global"]["meanIoU"] >= 0
