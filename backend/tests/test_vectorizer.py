import numpy as np
import cv2

from backend.vectorizer import vectorize_from_wall_mask


def test_vectorize_from_wall_mask_simple_rect():
    mask = np.zeros((120, 120), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (100, 100), color=1, thickness=3)
    image = np.zeros((120, 120, 3), dtype=np.uint8)

    walls, overlay = vectorize_from_wall_mask(mask, image, meters_per_pixel=0.05)

    assert walls, "Expected at least one wall segment from synthetic mask"
    assert isinstance(overlay, str)
    assert overlay.startswith("data:image/png;base64")
