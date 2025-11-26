import numpy as np
import cv2

from backend.vectorizer import vectorize_from_wall_mask


def test_vectorize_from_wall_mask_simple_rect():
    """
    Test vectorization with a larger synthetic mask that survives morphological
    operations and produces segments above the minimum wall length threshold.
    """
    # Create a larger image with thick wall lines to survive morphology
    mask = np.zeros((400, 400), dtype=np.uint8)
    # Draw thick horizontal and vertical lines representing walls
    cv2.line(mask, (50, 50), (350, 50), color=1, thickness=8)   # top wall
    cv2.line(mask, (50, 350), (350, 350), color=1, thickness=8) # bottom wall
    cv2.line(mask, (50, 50), (50, 350), color=1, thickness=8)   # left wall
    cv2.line(mask, (350, 50), (350, 350), color=1, thickness=8) # right wall
    
    image = np.zeros((400, 400, 3), dtype=np.uint8)

    # Use a higher meters_per_pixel so min wall length is fewer pixels
    walls, overlay = vectorize_from_wall_mask(mask, image, meters_per_pixel=0.02)

    assert walls, "Expected at least one wall segment from synthetic mask"
    assert isinstance(overlay, str)
    assert overlay.startswith("data:image/png;base64")
