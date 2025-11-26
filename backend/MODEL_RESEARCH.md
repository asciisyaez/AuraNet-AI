# Floor plan wall-detection model research

This note curates open weights worth evaluating and outlines how to validate them locally. The goal is to reach **≥98% wall IoU** on curated floor-plan validation data, using reproducible, self-contained checkpoints.

## Recommended candidate weights

| Model | Why evaluate | Pretrained weights | Notes |
| --- | --- | --- | --- |
| DexiNed (edge detector) | Recent edge detector with strong thin-line recall; robust to blueprint noise. | https://huggingface.co/iszlai/dependencies/resolve/main/dexined.pth | Convert to ONNX for CPU-only serving; pair with the existing Hough vectorizer. |
| HAWP (wireframe parser) | Produces structured line segments directly; good for geometric layouts. | https://github.com/cherubicxn/hawp (pretrained on wireframes) | Requires PyTorch; keep as a benchmarking baseline because it emits segments instead of masks. |
| SegFormer-B2 wall segmentation (fine-tuned on CubiCasa5k) | Strong semantic segmentation backbone; reported ≈mIoU>0.95 on structural classes when fine-tuned. | https://huggingface.co/datasets/CubiCasa5k for training data; initialize from https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512 | Best candidate to push toward ≥98% wall IoU after fine-tuning on blueprint crops. |
| Mask R-CNN (ResNet-50) wall/headroom segmentation | Mature instance/semantic segmentation baseline; easy to fine-tune on custom wall masks. | https://github.com/facebookresearch/detectron2 model zoo ("mask_rcnn_R_50_FPN_3x") | Simple to train; export to ONNX or TorchScript for serving. |

## Validation strategy

1. Assemble a held-out set of floor-plan images with pixel-perfect wall masks (e.g., 200–500 CubiCasa5k samples or your internal set).
2. Use `wall_benchmark.py` (see below) to run each candidate model end-to-end and compute per-image precision/recall/IoU along with structural F1 on vectorized walls.
3. Target wall IoU ≥98% and segment F1 >0.95 before adopting a model. Track failure cases (text bleed-through, scale rulers, curved walls) separately.

## Converting and dropping in new weights

- Edge/wireframe models (DexiNed/HAWP) can feed the existing vectorizer by replacing `hed_edges` in `ml_wall_detection.py` with a model-specific edge map.
- Segmentation models (SegFormer/Mask R-CNN) should emit a wall probability map; threshold it and vectorize via skeletonization + Hough to stay compatible with the API response.
- Prefer ONNX exports for reproducibility and CPU inference; keep exports in `backend/models/` with a short README describing provenance and hashes.

## Benchmark tooling (new)

See `wall_benchmark.py` for a reproducible harness that:
- Iterates over a folder of `{image}.png` and `{image}_mask.png` pairs.
- Runs the ML detector end-to-end (including text suppression and vectorization).
- Computes pixel precision/recall/IoU plus a structural F1 score by rasterizing predicted/true walls.
- Outputs a JSON summary and per-sample diagnostics so you can spot regressions when swapping weights.

With these pieces in place, you can experiment rapidly, measure progress toward the 98% IoU goal, and justify model/weight choices with real metrics.
