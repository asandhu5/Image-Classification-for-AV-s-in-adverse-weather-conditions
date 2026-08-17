# Visual Object Classification for Autonomous Vehicles in Adverse Weather

---

## What this repository contains

This is the complete codebase for my Master's thesis on real-time object
classification for autonomous vehicles under adverse weather (rain, fog, snow).
It has two self-contained parts:

| Directory | What it is |
|-----------|-----------|
| [`proposed/`](proposed/) | The **proposed model**: a two-stage YOLOv8m + CustomViT pipeline (7.6M params) — training, evaluation, cross-dataset testing, and image/video/webcam inference |
| [`benchmark/`](benchmark/) | The **benchmark suite**: nine baseline models (VGG16, ResNet-101, InceptionV3, Xception, and five ViT variants) trained and evaluated under identical conditions for comparison |

Each part has its own `requirements.txt` and README and runs independently.

---

## Key results (BDD100K test set)

| Model | Params (M) | FLOPs (G) | Eval time (s) | Accuracy |
|-------|-----------|-----------|---------------|----------|
| **CustomViT (ours)** | **7.60** | **4.19** | **0.5046** | **91.47%** |
| Xception | 22.95 | 8.4 | 0.9509 | 87.23% |
| VGG16 | 136.90 | 15.5 | 0.9935 | 86.75% |
| InceptionV3 | 29.58 | 5.7 | 0.8067 | 85.16% |
| ResNet-101 | 47.16 | 7.6 | 0.8413 | 84.23% |
| ViT-B/16 | 76.41 | 14.87 | 0.8307 | 81.61% |
| ViT-B/32 | 81.74 | 37.89 | 0.8874 | 80.41% |
| ViT-L/16 | 273.74 | 12.75 | 1.0780 | 78.14% |
| ViT-L/32 | 287.11 | 44.11 | 1.1110 | 81.07% |
| ViT-H/14 | 518.74 | 111.02 | 1.4360 | 82.46% |

The proposed model is the smallest, fastest, and most accurate of the set —
outperforming ViT-H/14 (68× larger) by ~9 percentage points while running
nearly 3× faster.

**Cross-dataset generalization** (trained on BDD100K only):

| Dataset | Accuracy |
|---------|----------|
| ACDC | 88.37% |
| CADC | 89.89% |
| Cityscapes | 87.00% |
| ONCE | 89.20% |

---

## Approach

Stage 1 uses YOLOv8m purely as a localizer to extract object ROIs from the
driving scene. Stage 2 classifies each ROI with a compact custom Vision
Transformer (256-dim embeddings, 8 encoder layers, 8 attention heads,
512-dim MLP) into 10 traffic-relevant classes (car, bus, truck, person,
rider, motorcycle, cycle, train, traffic light, traffic sign). Decoupling
localization from classification lets the classifier stay small enough for
real-time use while the transformer's global attention provides robustness
to the visual noise introduced by rain, fog, and snow. Training data comes
from BDD100K (70/10/20 split), with cross-dataset evaluation on ACDC, CADC,
Cityscapes, and ONCE.

---

## Quick start

```bash
# Proposed model
cd proposed
pip install -r requirements.txt
python train.py                          # train CustomViT
python evaluate.py --all-datasets        # evaluate on all 5 datasets
python inference.py --source dashcam.mp4 # run the full YOLO+ViT pipeline

# Benchmarks
cd benchmark
pip install -r requirements.txt
python vgg16/train.py                    # each baseline trains the same way
python vgg16/evaluate.py
```

See [`proposed/README.md`](proposed/README.md) and
[`benchmark/README.md`](benchmark/README.md) for full dataset setup,
training options, and per-model details.

---

## Repository layout

```
adverse-weather-object-classification/
├── proposed/                  # Thesis model (YOLOv8m + CustomViT)
│   ├── config.py              # All hyperparameters (thesis Tables 4.2–4.5, 5.2)
│   ├── train.py / evaluate.py / inference.py
│   ├── models/                # vit.py, detector.py, pipeline.py
│   ├── dataset/  engine/  utils/
│   └── scripts/               # ACDC / CADC / Cityscapes / ONCE preparation
│
└── benchmark/                 # Nine baselines under identical training conditions
    ├── shared/                # Common dataset / trainer / evaluator / metrics
    ├── vgg16/  resnet101/  inceptionv3/  xception/
    └── vit_b16/  vit_b32/  vit_l16/  vit_l32/  vit_h14/
```

---

## Environment

- Python 3.11.9 | PyTorch 2.1+ | CUDA 12.x
- NVIDIA RTX 4090 (24 GB) | Ubuntu 22.04

## License

Research use only. BDD100K, ACDC, CADC, Cityscapes, and ONCE are subject to
their respective licenses. YOLOv8 is AGPL-3.0 (Ultralytics).
