import argparse
import os
from pathlib import Path
from typing import Tuple, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerConfig

from backend.training.datasets.cubicasa5k import CubiCasaWalls
from backend.training.datasets.roboflow_walls import RoboflowWalls
from backend.training.datasets.combined import CombinedWallsDataset


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
    intersect = (probs * targets_one_hot).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
    dice = (2 * intersect + eps) / (union + eps)
    return 1 - dice.mean()


def evaluate(model: SegformerForSemanticSegmentation, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    ious = []
    losses = []
    ce_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8], device=device))
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(pixel_values=images)
            logits = outputs.logits
            loss = ce_loss(logits, masks) + dice_loss(logits, masks)
            losses.append(loss.item())

            preds = logits.argmax(dim=1)
            intersection = torch.logical_and(preds == 1, masks == 1).sum().item()
            union = torch.logical_or(preds == 1, masks == 1).sum().item()
            iou = intersection / union if union else 0.0
            ious.append(iou)
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    mean_loss = sum(losses) / len(losses) if losses else 0.0
    return mean_loss, mean_iou


def build_datasets(
    cubicasa_root: Optional[str],
    roboflow_root: Optional[str],
    split: str,
    augment: bool,
    target_size: Tuple[int, int] = (512, 512),
) -> Dataset:
    """
    Build a combined dataset from available sources.
    
    Args:
        cubicasa_root: Path to CubiCasa5k dataset (optional)
        roboflow_root: Path to Roboflow floor-plan-walls dataset (optional)
        split: 'train', 'val', or 'test'
        augment: Whether to apply data augmentation
        target_size: Target image size for all samples
    
    Returns:
        Combined Dataset ready for training
    """
    datasets = []
    
    if cubicasa_root and Path(cubicasa_root).exists():
        try:
            ds = CubiCasaWalls(cubicasa_root, split=split, augment=augment)
            datasets.append(ds)
            print(f"  Loaded CubiCasa5k {split}: {len(ds)} samples")
        except Exception as e:
            print(f"  Warning: Could not load CubiCasa5k: {e}")
    
    if roboflow_root and Path(roboflow_root).exists():
        try:
            ds = RoboflowWalls(roboflow_root, split=split, augment=augment)
            datasets.append(ds)
            print(f"  Loaded Roboflow Walls {split}: {len(ds)} samples")
        except Exception as e:
            print(f"  Warning: Could not load Roboflow Walls: {e}")
    
    if not datasets:
        raise ValueError("No valid datasets found. Provide at least one of --cubicasa-root or --roboflow-root")
    
    if len(datasets) == 1:
        # Single dataset - wrap with resize only
        return CombinedWallsDataset(datasets, target_size=target_size)
    
    # Multiple datasets - combine
    combined = CombinedWallsDataset(datasets, target_size=target_size)
    stats = combined.get_dataset_stats()
    print(f"  Combined dataset: {stats['total_samples']} total samples from {stats['num_datasets']} sources")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SegFormer-B2 wall segmentation model.")
    parser.add_argument("--cubicasa-root", type=str, default=None,
                        help="CubiCasa5k dataset root (images/ and masks/ folders)")
    parser.add_argument("--roboflow-root", type=str, default=None,
                        help="Roboflow floor-plan-walls dataset root (train/valid/test folders)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Legacy: same as --cubicasa-root for backwards compatibility")
    parser.add_argument("--output-dir", required=True, help="Directory to store checkpoints")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=512, help="Target image size (square)")
    args = parser.parse_args()

    # Handle legacy --data-root argument
    cubicasa_root = args.cubicasa_root or args.data_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    target_size = (args.image_size, args.image_size)
    
    print("Loading training datasets...")
    train_ds = build_datasets(
        cubicasa_root=cubicasa_root,
        roboflow_root=args.roboflow_root,
        split="train",
        augment=True,
        target_size=target_size,
    )
    
    print("Loading validation datasets...")
    val_ds = build_datasets(
        cubicasa_root=cubicasa_root,
        roboflow_root=args.roboflow_root,
        split="val",
        augment=False,
        target_size=target_size,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    config = SegformerConfig.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=2,
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        config=config,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8], device=device))

    best_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, masks in progress:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=images)
            logits = outputs.logits
            loss = ce_loss(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()

            progress.set_postfix({"loss": loss.item()})

        val_loss, val_iou = evaluate(model, val_loader, device)
        tqdm.write(f"Validation loss: {val_loss:.4f}, IoU: {val_iou:.4f}")

        ckpt_path = Path(args.output_dir) / f"epoch_{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)

        if val_iou > best_iou:
            best_iou = val_iou
            best_path = Path(args.output_dir) / "best.pt"
            torch.save(model.state_dict(), best_path)
            tqdm.write(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
