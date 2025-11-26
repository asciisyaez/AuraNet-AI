import os


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Tunable parameters for morphology/vectorization
SEGFORMER_PROB_THRESHOLD = _env_float("SEGFORMER_PROB_THRESHOLD", 0.5)
CLOSE_RADIUS_METERS = _env_float("WALL_CLOSE_RADIUS_METERS", 0.15)
MERGE_GAP_METERS = _env_float("WALL_MERGE_GAP_METERS", 0.1)
MIN_WALL_METERS = _env_float("WALL_MIN_WALL_METERS", 0.3)
MIN_SPUR_METERS = _env_float("WALL_MIN_SPUR_METERS", 0.3)

# Runtime configuration
DEFAULT_SEGFORMER_MODEL_PATH = os.getenv(
    "SEGFORMER_MODEL_PATH", "backend/models/segformer_b2_walls.onnx"
)
DEFAULT_DEXINED_MODEL_PATH = os.getenv(
    "DEXINED_MODEL_PATH", "backend/models/dexined.onnx"
)
DEFAULT_DEVICE = os.getenv("WALL_DETECTOR_DEVICE", "cpu")

# Debug overlay
OVERLAY_ALPHA = _env_float("WALL_OVERLAY_ALPHA", 0.6)

# Hough parameters
HOUGH_THRESHOLD = _env_int("WALL_HOUGH_THRESHOLD", 20)
