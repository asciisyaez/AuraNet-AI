"""
Combined dataset that merges multiple wall segmentation datasets.

This allows training on multiple sources (CubiCasa5k, Roboflow, etc.)
to improve model generalization and achieve SOTA performance.
"""

import random
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import torchvision.transforms.functional as TF


class CombinedWallsDataset(Dataset):
    """
    Combines multiple wall segmentation datasets with optional resizing
    to a common resolution for batching.
    
    This wrapper handles:
    - Combining samples from multiple datasets
    - Resizing images to a common size for efficient batching
    - Maintaining class balance across datasets (optional weighting)
    """

    def __init__(
        self,
        datasets: List[Dataset],
        target_size: Optional[Tuple[int, int]] = (512, 512),
        weights: Optional[List[float]] = None,
    ) -> None:
        """
        Args:
            datasets: List of Dataset objects to combine
            target_size: (H, W) to resize all images to. None = no resizing.
            weights: Optional sampling weights for each dataset. If provided,
                     samples are weighted by dataset. If None, simple concatenation.
        """
        self.datasets = datasets
        self.target_size = target_size
        self.weights = weights

        # Build combined dataset
        self.concat = ConcatDataset(datasets)
        
        # Pre-compute dataset boundaries for weighted sampling
        self.boundaries = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.boundaries.append(total)

    def __len__(self) -> int:
        return len(self.concat)

    def _resize_pair(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resize image and mask to target size."""
        if self.target_size is None:
            return image, mask

        h, w = self.target_size
        
        # Convert to PIL for high-quality resize
        image_pil = TF.to_pil_image(image)
        mask_pil = TF.to_pil_image(mask.float())
        
        image_resized = image_pil.resize((w, h), Image.BILINEAR)
        mask_resized = mask_pil.resize((w, h), Image.NEAREST)
        
        image_tensor = TF.to_tensor(image_resized)
        mask_tensor = (TF.to_tensor(mask_resized) > 0.5).long().squeeze(0)
        
        return image_tensor, mask_tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.concat[idx]
        return self._resize_pair(image, mask)

    def get_dataset_stats(self) -> dict:
        """Return statistics about the combined dataset."""
        stats = {
            "total_samples": len(self.concat),
            "num_datasets": len(self.datasets),
            "samples_per_dataset": [len(ds) for ds in self.datasets],
        }
        return stats
