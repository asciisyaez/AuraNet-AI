from pathlib import Path
import random
from typing import Tuple, List

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter


class CubiCasaWalls(Dataset):
    """
    Minimal dataset for floorplan wall segmentation.
    Expects PNG images and corresponding *_mask.png files with wall pixels = 255.
    """

    def __init__(self, root: str, split: str = "train", augment: bool = False) -> None:
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        self.augment = augment
        self.jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

        all_images = sorted(self.images_dir.glob("*.png"))
        if not all_images:
            raise ValueError(f"No images found under {self.images_dir}")

        # Simple deterministic split
        train_cut = int(0.7 * len(all_images))
        val_cut = int(0.85 * len(all_images))
        if split == "train":
            self.image_paths = all_images[:train_cut]
        elif split == "val":
            self.image_paths = all_images[train_cut:val_cut]
        elif split == "test":
            self.image_paths = all_images[val_cut:]
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_pair(self, image_path: Path) -> Tuple[Image.Image, Image.Image]:
        mask_path = self.masks_dir / f"{image_path.stem}_mask.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for {image_path.name}")
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        return image, mask

    def _apply_transforms(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if not self.augment:
            return image, mask

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        k = random.randint(0, 3)
        if k:
            angle = 90 * k
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        image = self.jitter(image)
        return image, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        image, mask = self._load_pair(image_path)
        image, mask = self._apply_transforms(image, mask)

        image_tensor = TF.to_tensor(image)  # [3,H,W], float32 in [0,1]
        mask_tensor = (TF.to_tensor(mask) > 0.5).long().squeeze(0)

        return image_tensor, mask_tensor
