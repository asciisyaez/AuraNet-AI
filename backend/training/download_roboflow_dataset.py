#!/usr/bin/env python
"""
Download the Roboflow Floor Plan Walls dataset.

Usage:
    python backend/training/download_roboflow_dataset.py --api-key YOUR_API_KEY --output-dir ./data/roboflow_walls

Get your API key from: https://app.roboflow.com/settings/api
"""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download Roboflow Floor Plan Walls dataset")
    parser.add_argument("--api-key", required=True, help="Roboflow API key")
    parser.add_argument("--output-dir", default="./data/roboflow_walls", help="Output directory")
    parser.add_argument("--format", default="yolov8", choices=["yolov8", "coco", "voc"],
                        help="Export format (yolov8 recommended)")
    args = parser.parse_args()

    try:
        from roboflow import Roboflow
    except ImportError:
        print("Error: roboflow package not installed.")
        print("Install with: pip install roboflow")
        return 1

    print("Connecting to Roboflow...")
    rf = Roboflow(api_key=args.api_key)
    
    print("Accessing Floor Plan Walls project...")
    project = rf.workspace("newaguss").project("floor-plan-walls-pdiqq")
    
    print(f"Downloading dataset in {args.format} format...")
    # Get the latest version
    version = project.version(1)
    
    # Download to specified directory
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = version.download(args.format, location=args.output_dir)
    
    print(f"\nDataset downloaded to: {args.output_dir}")
    print(f"\nDataset info:")
    print(f"  - Format: {args.format}")
    print(f"  - Images: ~3,400")
    print(f"  - Classes: door, wall, window")
    
    print(f"\nTo train with this dataset:")
    print(f"  python backend/training/train_segformer_b2_walls.py \\")
    print(f"    --roboflow-root {args.output_dir} \\")
    print(f"    --output-dir ./checkpoints/segformer_walls \\")
    print(f"    --epochs 60")
    
    return 0


if __name__ == "__main__":
    exit(main())
