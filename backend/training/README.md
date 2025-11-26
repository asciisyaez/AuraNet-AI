# Wall Detection Training (SegFormer-B2)

Reference scripts to reproduce the `ml-v2` SegFormer model. Training is offline; artifacts should be exported to ONNX and placed under `backend/models/`.

## Layout

- `datasets/cubicasa5k.py` – PyTorch dataset for CubiCasa5k-style images + wall masks.
- `train_segformer_b2_walls.py` – Minimal training loop using HuggingFace `SegformerForSemanticSegmentation`.
- `export_segformer_to_onnx.py` – Export a fine-tuned checkpoint to ONNX with dynamic height/width axes.

## Dataset Assumptions

```
DATA_ROOT/
  images/
    sample_0001.png
  masks/
    sample_0001_mask.png  # 0/255 wall mask
```

## Quickstart

```bash
python -m pip install -r backend/training/requirements.txt
python backend/training/train_segformer_b2_walls.py \
  --data-root /path/to/data \
  --output-dir /tmp/wall_runs/segformer_b2 \
  --epochs 10

python backend/training/export_segformer_to_onnx.py \
  --checkpoint /tmp/wall_runs/segformer_b2/best.pt \
  --output backend/models/segformer_b2_walls.onnx
```

After export, validate with `backend/tools/wall_benchmark.py` before shipping.
