# ML-driven wall detection (experimental)

This backend now exposes an experimental ML detector that runs a Holistically-Nested Edge Detection (HED) model before vectorizing walls. The endpoint shares the same contract as the existing `/api/detect-walls-base64` route and can be selected by passing `detector: "ml"` in the request body.

## How it works
1. Download a pre-trained HED Caffe model (lazy-downloaded on first run to `backend/models`).
2. Run HED to obtain a dense edge map that preserves faint blueprint strokes better than Canny.
3. Suppress thin strokes (text, rulers, dimension lines) via stroke-width filtering and connected-component pruning.
4. Morphologically close edges into wall-like blobs and vectorize them with a probabilistic Hough transform.
5. Return vector walls, a glowing overlay preview, and diagnostics showing edge density, suppressed pixels, and segment counts.

## Usage
```
POST /api/detect-walls-base64
{
  "image": "data:image/png;base64,....",
  "metersPerPixel": 0.05,
  "detector": "ml"
}
```

If the weights are missing or corrupted, the API responds with empty walls and a diagnostic note describing the issue. Delete the `backend/models` folder to trigger a re-download.

## Notes
- The HED caffemodel (~50 MB) is downloaded from the OpenCV extra testdata repository on demand; keep it cached between runs for speed.
- Tune `metersPerPixel` accurately—the morphology kernel, thin-stroke filters, and minimum segment length scale with it.
- This path is intentionally simple and should be treated as a baseline for further ML experiments (e.g., DexiNed, wireframe detectors, or custom fine-tuning on your floor plan corpus).

## Researching better weights & validation
- See `backend/MODEL_RESEARCH.md` for a curated list of candidate edge/segmentation models (DexiNed, HAWP, SegFormer) and how to slot their weights into this pipeline.
- Use `backend/wall_benchmark.py` to validate any candidate weights against a labeled floor-plan set (image + wall-mask pairs). The script reports pixel IoU/precision/recall and a structural F1 score derived from coverage with a tolerance band.
- Do not adopt a model unless it clears ≥98% wall IoU on a held-out set representative of your production scans.
