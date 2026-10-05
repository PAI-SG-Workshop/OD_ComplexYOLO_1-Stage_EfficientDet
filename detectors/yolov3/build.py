"""
YOLOv3 builders for the three tasks.

Everything here is algorithm-specific:
    - which .cfg (backbone + neck + head layout)
    - how [yolo] anchors are parsed
    - which detection layer (loss / target assignment) is used
    - how many box attributes the head regresses

Everything else (Darknet parsing, training, evaluation, demo) is common.
"""

import math
from pathlib import Path

from detectors.yolov3.complex_yolo_layer import YoloLayer as ComplexYoloLayer

CFG_DIR = Path(__file__).resolve().parent / "cfg"


def _floats(text):
    return [float(v) for v in text.split(",")]


def _mask(block):
    return [int(v) for v in block["mask"].split(",")]


def complex_factory(block, hyperparams):
    """Complex-YOLO: (w, l, yaw) anchors -> (w, l, sin, cos) (legacy)."""
    v = _floats(block["anchors"])
    anchors = [(v[i], v[i + 1], math.sin(v[i + 2]), math.cos(v[i + 2])) for i in range(0, len(v), 3)]
    return ComplexYoloLayer([anchors[i] for i in _mask(block)], int(block["classes"]),
                            int(hyperparams["height"]))


SPECS = {
    #          cfg file                 factory          box attributes regressed
    "complex": ("complex_yolov3.cfg", complex_factory, 7),   # x y w l im re conf
}
