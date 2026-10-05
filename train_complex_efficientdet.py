"""Train complex efficientdet (KITTI track: backbone + neck with the YOLOv3 layers), then evaluate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.runner import main
from detectors.efficientdet.build import build_model

if __name__ == "__main__":
    main("train", task="complex", version="efficientdet", build_model=build_model)
