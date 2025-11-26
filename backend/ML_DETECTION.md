# ML-driven wall detection (experimental)

This backend now exposes an experimental ML detector that runs a Holistically-Nested Edge Detection (HED) model before vectorizing walls. The endpoint shares the same contract as the existing `/api/detect-walls-base64` route and can be selected by passing `detector: "ml"` in the request body.

## How it works
1. Download a pre-trained HED Caffe model (lazy-downloaded on first run to `backend/models`).
2. Run HED to obtain a dense edge map that preserves faint blueprint strokes better than Canny.
3. Morphologically close edges into wall-like blobs and vectorize them with a probabilistic Hough transform.
4. Return vector walls, an overlay preview, and diagnostics showing edge density and segment counts.

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
- Tune `metersPerPixel` accurately—the morphology kernel and minimum segment length scale with it.
- This path is intentionally simple and should be treated as a baseline for further ML experiments (e.g., DexiNed, wireframe detectors, or custom fine-tuning on your floor plan corpus).
