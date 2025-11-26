# Wall Detection Models

This folder stores inference-ready artifacts for the wall detection pipeline. Keep large binaries out of Git; download or mount them at runtime.

## Expected Files

- `segformer_b2_walls.onnx` – SegFormer-B2 fine-tuned for wall segmentation (input `[1,3,H,W]`, output `[1,1,H,W]` named `logits`). Intended as the default `ml-v2` path.
- `dexined.onnx` – DexiNed edge detector used by the `ml-dexined` variant.

## Provenance & Validation

- SegFormer export command (see `../training/export_segformer_to_onnx.py`):
  - Opset: 13
  - Dynamic axes: height/width
  - Example hash placeholder: `sha256: <fill-after-download>`
- Validate each export with a quick forward pass and ensure mean IoU/structural F1 meets the PRD thresholds via `backend/tools/wall_benchmark.py`.

## Configuration

- Override paths with env vars:
  - `SEGFORMER_MODEL_PATH`
  - `DEXINED_MODEL_PATH`
- Runtime device can be set with `WALL_DETECTOR_DEVICE` (`cpu` or `cuda`).

If models are missing, the backend will return an empty `walls` array with a diagnostics note instead of crashing.
