# OD_ComplexYOLO_1-Stage_EfficientDet

**EfficientDet - ComplexYOLO track - training and test evaluation**

Part of the *Physical AI Object Detection* reference series of the **Statera Working Group**:
one algorithm and one task per repository, the same data, protocol and metrics for every algorithm -
only the algorithm differs.

## Model

- Paper: Tan, Pang & Le, *EfficientDet: Scalable and Efficient Object Detection*, CVPR 2020 (arXiv:1911.09070)
- Implementation source: rwightman efficientdet-pytorch (pip effdet 0.4.1, reproduces the official google/automl weights)
- Parameters: 3.80 M (3 classes)

- EfficientNet-B0 (timm) + 3 x BiFPN (64 channels)
- Class / box nets with separable convs and per-level BN

Detection layer / assignment / loss: Complex-YOLO layer of the YOLOv3 lineage: x, y, w, l + (im, re) with E-RPN, fixed z / h - identical in every model of this track; EfficientDet changes the backbone / neck only

Differences from the original (common protocol):

- No ImageNet weights, common init, no mean / std, 608 input, Adam with the common schedule, common augmentation and post-processing (see [Unified] markers)
- PreciseBN before every evaluation and the final save (micro batch 2 BN statistics; RetinaNet test mAP50 0.148 -> 0.849 for the same weights)
- 30 epochs are short for EfficientDet (official 300, pretrained backbone)
- KITTI: EfficientNet-B0 + BiFPN (P3-P5) with the YOLOv3 Complex-YOLO / YOLO3D layers

## Data: KITTI_5k (LiDAR bird's-eye-view, 3 classes)

5,000 KITTI object frames with LiDAR, split into train / test by the fixed list in `data_splits/KITTI_5k/`; classes Car, Pedestrian, Cyclist; 608 x 608 BEV maps.

The KITTI data must be obtained from http://www.cvlibs.net/datasets/kitti/ under its own license.

## Results

Test set: KITTI_5k test (2,470 frames), IoU 0.5, after 10 epochs (0.8 h on an RTX 4070 Laptop GPU).

| AP50 | mean | Car | Pedestrian | Cyclist |
| --- | --- | --- | --- | --- |
| BEV | **0.3749** | 0.8412 | 0.1014 | 0.1822 |
| 3D | **0.2096** | 0.5694 | 0.0407 | 0.0186 |

Complex-YOLO does not regress z / h (paper); the 3D AP uses fixed class heights.

Protocol: 10 epochs, Adam lr 1e-3 (constant), effective batch 8 (micro batch 2 x 4 accumulation), BEV 608, seed 777, no pretraining; point-cloud augmentation of the Legacy code.

## Usage

```bash
pip install -r requirements.txt
python data_processing_complex.py   # extracts KITTI_5k (see the script header for the source zip)
python train_complex_efficientdet.py               # training, then test evaluation (runs/complex_efficientdet/)
python evaluate_complex_efficientdet.py            # test evaluation of runs/complex_efficientdet/checkpoints/ (or --weights <file>)
```

Data are expected under `$PHYSICAL_AI_ROOT/datasets` (default `~/Physical_AI/datasets`).
Smoke test: `python train_complex_efficientdet.py --epochs 1 --max-train 8 --max-test 4`.

Trained weights: the GitHub Release of this repository contains `complex_efficientdet_kitti5k_e010.pth` (the run reported above).
Put it into `runs/complex_efficientdet/checkpoints/` or pass `--weights <file>` to the evaluation script.
Demo scripts using these weights will be added in a later release.

## Files

```
data_processing_*.py      step 1  data
train_complex_efficientdet.py         step 2  training + test evaluation
evaluate_complex_efficientdet.py      step 3  test evaluation
common/                   protocol (config), data loading, trainer, metrics, utilities
tasks/                    task definition (data sets, post-processing, evaluation)
detectors/                EfficientDet (efficientdet, retinanet, anchor_free, yolov3)
THIRD_PARTY.md            sources and licenses of the reused code
```

Licenses: see `THIRD_PARTY.md`.

## Global Collaboration

Hills Robotics is currently participating in multiple projects in Korea and internationally.

We are actively interested in building relationships with organizations that can complement our technology with global sales networks, distribution channels, system integration capabilities, manufacturing ecosystems, AI infrastructure, robotics technologies, research capabilities, or strategic investment.

Several technology and business collaborations are already underway. Some partners, projects, system architectures, and commercial programs cannot be publicly disclosed because they are subject to confidentiality and business agreements.

We believe that the emerging Physical AI ecosystem cannot be built by a single company.
It requires a global network connecting robotics companies, AI researchers, semiconductor and computing companies, sensor manufacturers, software developers, universities, system integrators, industrial customers, investors, and global distribution partners.

We have now begun our serious exploration of the Physical AI frontier, and we welcome connections with people and organizations who want to build this ecosystem together.

## Research, Contribution, and Donation

The repositories of the Statera Working Group are intended to contribute to the broader technical discussion around Physical AI.

Researchers, engineers, developers, companies, universities, and independent contributors interested in these subjects are welcome to connect with us.

If you would like to collaborate on Physical AI or autonomous robotics, participate in research or technology development, discuss international business cooperation, propose technology or platform integration, contribute technical knowledge, or support this GitHub knowledge initiative through sponsorship or donation, please contact us.

## Contact

**Richard Minsoo Go**
Head of AI, Hills Robotics
Statera Working Group

Personal / Research Contact:
[gominsoo888@gmail.com](mailto:gominsoo888@gmail.com)

Hills Robotics / Business Contact:
[minsoo.go@hillslogis.com](mailto:minsoo.go@hillslogis.com)

Hills Robotics:
https://hillslogis.com/

For inquiries regarding next-generation outdoor autonomous robotic platforms, Physical AI Spider Grid, Intelligence Cabinet, Edge AI, or On-Premise AI architectures that are not yet publicly presented on the company website, please contact us directly.
