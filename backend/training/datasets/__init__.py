# Dataset loaders for wall segmentation training

from backend.training.datasets.cubicasa5k import CubiCasaWalls
from backend.training.datasets.roboflow_walls import RoboflowWalls
from backend.training.datasets.combined import CombinedWallsDataset

__all__ = ["CubiCasaWalls", "RoboflowWalls", "CombinedWallsDataset"]
