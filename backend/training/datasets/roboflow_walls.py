"""
Roboflow Floor Plan Walls dataset loader.

This loader supports the Roboflow "Floor Plan Walls" dataset format:
https://universe.roboflow.com/newaguss/floor-plan-walls-pdiqq

The dataset uses YOLO format with bounding boxes for wall, door, and window classes.
We convert wall bounding boxes to segmentation masks for training.

To download:
    pip install roboflow
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace("newaguss").project("floor-plan-walls-pdiqq")
    dataset = project.version(1).download("yolov8")
"""

import random
from pathlib import Path
from typing import Tuple, Optional, List

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter


class RoboflowWalls(Dataset):
    """
    Dataset for Roboflow Floor Plan Walls in YOLO format.
    
    Expects the standard Roboflow YOLO export structure:
        root/
            train/
                images/
                    img001.jpg
                labels/
                    img001.txt  (YOLO format: class x_center y_center width height)
            valid/
                images/
                labels/
            test/
                images/
                labels/
            data.yaml
    
    The 'wall' class (typically class 1 in this dataset) is converted to a 
    segmentation mask by drawing filled rectangles.
    """

    WALL_CLASS_ID = 1  # 'wall' class in the dataset (0=door, 1=wall, 2=window)

    def __init__(
        self,
        root: str,
        split: str = "train",
        augment: bool = False,
        wall_thickness: int = 8,
    ) -> None:
        """
        Args:
            root: Path to extracted Roboflow dataset root
            split: One of 'train', 'val' (maps to 'valid'), or 'test'
            augment: Whether to apply data augmentation
            wall_thickness: Pixel thickness to use when rasterizing wall boxes
        """
        self.root = Path(root)
        self.augment = augment
        self.wall_thickness = wall_thickness
        self.jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

        # Roboflow uses 'valid' folder name
        split_folder = "valid" if split == "val" else split
        self.images_dir = self.root / split_folder / "images"
        self.labels_dir = self.root / split_folder / "labels"

        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")

        # Find all images (Roboflow exports as jpg typically)
        self.image_paths = sorted(
            list(self.images_dir.glob("*.jpg")) +
            list(self.images_dir.glob("*.jpeg")) +
            list(self.images_dir.glob("*.png"))
        )

        if not self.image_paths:
            raise ValueError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _parse_yolo_label(self, label_path: Path, img_width: int, img_height: int) -> List[Tuple[int, int, int, int]]:
        """
        Parse YOLO format labels and return wall bounding boxes as pixel coordinates.
        
        YOLO format: class_id x_center y_center width height (all normalized 0-1)
        
        Returns:
            List of (x1, y1, x2, y2) tuples for wall class only
        """
        boxes = []
        if not label_path.exists():
            return boxes

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                class_id = int(parts[0])
                if class_id != self.WALL_CLASS_ID:
                    continue

                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height

                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)

                boxes.append((x1, y1, x2, y2))

        return boxes

    def _boxes_to_mask(
        self, boxes: List[Tuple[int, int, int, int]], width: int, height: int
    ) -> Image.Image:
        """
        Convert bounding boxes to a binary mask image.
        
        For elongated boxes (aspect ratio > 3), we draw them as thick lines
        (representing walls). For square-ish boxes, we fill them completely.
        """
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        for x1, y1, x2, y2 in boxes:
            box_width = x2 - x1
            box_height = y2 - y1
            
            if box_width == 0 or box_height == 0:
                continue

            aspect = max(box_width, box_height) / max(min(box_width, box_height), 1)

            if aspect > 3:
                # Elongated box - draw as thick line (wall)
                if box_width > box_height:
                    # Horizontal wall
                    cy = (y1 + y2) // 2
                    thickness = min(self.wall_thickness, box_height)
                    draw.rectangle(
                        [x1, cy - thickness // 2, x2, cy + thickness // 2],
                        fill=255
                    )
                else:
                    # Vertical wall
                    cx = (x1 + x2) // 2
                    thickness = min(self.wall_thickness, box_width)
                    draw.rectangle(
                        [cx - thickness // 2, y1, cx + thickness // 2, y2],
                        fill=255
                    )
            else:
                # Square-ish - fill the whole box (could be a wall section)
                draw.rectangle([x1, y1, x2, y2], fill=255)

        return mask

    def _load_pair(self, image_path: Path) -> Tuple[Image.Image, Image.Image]:
        """Load image and generate wall mask from YOLO labels."""
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        label_path = self.labels_dir / f"{image_path.stem}.txt"
        boxes = self._parse_yolo_label(label_path, width, height)
        mask = self._boxes_to_mask(boxes, width, height)

        return image, mask

    def _apply_transforms(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Apply data augmentation."""
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
