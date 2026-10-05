"""
EfficientDet-D0 builder (Image task; KITTI complex / yolo3d track: backbone + BiFPN with the YOLOv3 layers).

Paper: Tan, Pang, Le, "EfficientDet: Scalable and Efficient Object Detection" (CVPR 2020, arXiv 1911.09070).
Code : official google/automl (TensorFlow); approved official-equivalent source (2026-10-04): rwightman
       efficientdet-pytorch (pip package effdet 0.4.1, reproduces the official TF weights), used directly:
       effdet.EfficientDet (tf_efficientnet_b0 backbone via timm, BiFPN x3 with fast normalised fusion, 64 ch,
       class / box nets x3 with separable convs and per-level BN), effdet.anchors.Anchors (3 scales x 3 ratios,
       anchor_scale 4, P3-P7), AnchorLabeler (IoU 0.5), effdet.loss.DetectionLoss (focal alpha 0.25 gamma 1.5,
       Huber delta 0.1, box_loss_weight 50, / (#pos + 1)), decode_box_outputs.
       Requirements therefore include effdet (and with it timm).

effdet works in YXYX boxes and 1-based classes; the wrapper converts the framework's XYXY / 0-based targets.
Raw contract: sigmoid class probabilities, objectness = 1.

[Unified] no ImageNet backbone weights (pretrained_backbone=False); common init (nn.Conv2d incl. timm Conv2dSame,
BN; class-net prior bias kept); no mean / std; 608 x 608 instead of 512 x 512 (feature maps 76/38/19/10/5);
common augmentation instead of scale jitter; Adam 1e-4 instead of SGD 0.08 cosine + EMA; common post-processing
(instead of top-5000 + NMS 0.5, max 100). PreciseBN (BN everywhere, micro batch 2).
"""

import torch.nn as nn
from effdet import EfficientDet as EffDetNet
from effdet import get_efficientdet_config


class EfficientDetKitti(nn.Module):
    """KITTI track: EfficientNet-B0 + BiFPN (P3-P5 of the 5 BiFPN outputs) feeding the YOLOv3 layers."""

    def __init__(self, task, num_classes, name="tf_efficientdet_d0", image_size=608):
        super().__init__()
        from detectors.retinanet.kitti import KittiV3Head
        config = get_efficientdet_config(name)
        config.num_classes = num_classes
        config.image_size = (image_size, image_size)
        net = EffDetNet(config, pretrained_backbone=False)
        self.backbone, self.fpn = net.backbone, net.fpn          # class / box nets are not used on KITTI
        c = config.fpn_channels
        self.head = KittiV3Head(task, num_classes, {8: c, 16: c, 32: c}, self._features)
        self.yolo_layers = self.head.yolo_layers
        self.precise_bn, self.precise_bn_images, self.precise_bn_batch_size = True, 1600, 16

    def _features(self, x):
        p = self.fpn(self.backbone(x))
        return {8: p[0], 16: p[1], 32: p[2]}

    def forward(self, x, targets=None):
        return self.head(x, targets)

    def load_darknet_weights(self, path):
        raise NotImplementedError("EfficientDet has no Darknet weights; train from scratch.")


def build_model(task, num_classes):
    if task in ("complex", "yolo3d"):
        return EfficientDetKitti(task, num_classes)
    raise ValueError(f"unknown task {task}")
