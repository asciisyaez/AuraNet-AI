# Wall Detection Training (SegFormer-B2)

Reference scripts to reproduce the `ml-v2` SegFormer model. Training is offline; artifacts should be exported to ONNX and placed under `backend/models/`.

## Layout

- `datasets/cubicasa5k.py` – PyTorch dataset for CubiCasa5k-style images + wall masks.
- `datasets/roboflow_walls.py` – PyTorch dataset for Roboflow Floor Plan Walls (YOLO format).
- `datasets/combined.py` – Combines multiple datasets for SOTA training.
- `train_segformer_b2_walls.py` – Training loop using HuggingFace `SegformerForSemanticSegmentation`.
- `export_segformer_to_onnx.py` – Export a fine-tuned checkpoint to ONNX with dynamic height/width axes.

## Supported Datasets

### 1. CubiCasa5k (Segmentation Masks)

Standard format with image/mask pairs:

```
CUBICASA_ROOT/
  images/
    sample_0001.png
  masks/
    sample_0001_mask.png  # 0/255 wall mask
```

### 2. Roboflow Floor Plan Walls (YOLO Object Detection)

Download from: https://universe.roboflow.com/newaguss/floor-plan-walls-pdiqq

```python
# Download using Roboflow Python SDK
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("newaguss").project("floor-plan-walls-pdiqq")
dataset = project.version(1).download("yolov8")
```

Expected structure after download:

```
ROBOFLOW_ROOT/
  train/
    images/
      img001.jpg
    labels/
      img001.txt  # YOLO: class x_center y_center width height
  valid/
    images/
    labels/
  test/
    images/
    labels/
  data.yaml
```

The loader converts wall bounding boxes (class 1) to segmentation masks.

## Training

### Single Dataset (CubiCasa5k only)

```bash
python backend/training/train_segformer_b2_walls.py \
  --cubicasa-root /path/to/cubicasa5k \
  --output-dir /tmp/wall_runs/segformer_b2 \
  --epochs 60
```

### Single Dataset (Roboflow only)

```bash
python backend/training/train_segformer_b2_walls.py \
  --roboflow-root /path/to/floor-plan-walls-pdiqq \
  --output-dir /tmp/wall_runs/segformer_b2 \
  --epochs 60
```

### Combined Datasets (SOTA - Recommended)

Training on multiple datasets improves generalization:

```bash
python backend/training/train_segformer_b2_walls.py \
  --cubicasa-root /path/to/cubicasa5k \
  --roboflow-root /path/to/floor-plan-walls-pdiqq \
  --output-dir /tmp/wall_runs/segformer_b2_sota \
  --epochs 60 \
  --batch-size 4 \
  --image-size 512
```

## Export to ONNX

```bash
python backend/training/export_segformer_to_onnx.py \
  --checkpoint /tmp/wall_runs/segformer_b2_sota/best.pt \
  --output backend/models/segformer_b2_walls.onnx
```

## Validation

After export, validate with the benchmark tool:

```bash
python backend/tools/wall_benchmark.py \
  --images-dir /path/to/test/images \
  --masks-dir /path/to/test/masks \
  --detector ml-v2 \
  --output-json wall_benchmark_results.json
```

Target metrics for SOTA:
- Wall IoU ≥ 98%
- Structural F1 ≥ 0.95

## Dataset Citations

### Roboflow Floor Plan Walls

```bibtex
@misc{floor-plan-walls-pdiqq_dataset,
  title = { Floor Plan Walls Dataset },
  type = { Open Source Dataset },
  author = { NewAguss },
  howpublished = { \url{ https://universe.roboflow.com/newaguss/floor-plan-walls-pdiqq } },
  url = { https://universe.roboflow.com/newaguss/floor-plan-walls-pdiqq },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year = { 2025 },
  month = { apr },
}
```
