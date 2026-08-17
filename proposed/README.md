# Visual Object Classification for Autonomous Vehicles in Adverse Weather

M.Sc. Thesis — NUST, Islamabad (2025)  
Author: Muhammad Ahmed | Supervisor: Dr. Tahir Habib Nawaz

---

## Overview

> Part of the [adverse-weather-object-classification](../) monorepo — the nine baseline models used for comparison live in [`../benchmark/`](../benchmark/).

A two-stage hybrid pipeline for real-time object classification under adverse weather (rain, fog, snow):

- **Stage 1 — YOLOv8m** (Ultralytics): bounding-box detection / localisation
- **Stage 2 — CustomViT**: lightweight Vision Transformer for 10-class classification of each detected ROI

Trained and evaluated on BDD100K; cross-tested on ACDC, CADC, Cityscapes, and ONCE.

---

## Results

| Dataset     | Accuracy  | Inference |
|-------------|-----------|-----------|
| BDD100K     | **91.47%**| 0.4156 s  |
| ACDC        | 88.37%    | —         |
| CADC        | 89.89%    | —         |
| Cityscapes  | 87.00%    | —         |
| ONCE        | 89.20%    | —         |

Comparison (BDD100K test set):

| Model         | Accuracy  |
|---------------|-----------|
| **CustomViT** | **91.47%**|
| Xception      | 87.23%    |
| VGG16         | 86.75%    |
| InceptionV3   | 85.16%    |
| ResNet-101    | 84.23%    |

---

## Architecture

```
Input ROI (3 × 224 × 224)
  ↓
PatchEmbedding  — Conv2d(3, 256, k=16, s=16) → (B, 196, 256)
  ↓
[CLS] prepend                                 → (B, 197, 256)
  ↓
Positional Embedding                          → (B, 197, 256)
  ↓
8 × TransformerEncoderBlock
    └── Pre-norm → MultiheadAttn(8 heads) → residual
    └── Pre-norm → FFN(256→512→256, GELU)  → residual
  ↓
LayerNorm
  ↓
CLS token [:, 0]                              → (B, 256)
  ↓
MLP Head: Linear(256→512) → GELU → Dropout → Linear(512→10)
  ↓
Logits (B, 10)
```

**Note on attention heads:** The thesis Table 5.2 specifies 6 heads, but
`256 % 6 ≠ 0`. The implementation uses **8 heads** (256 / 8 = 32 per head)
as the nearest valid configuration.

---

## 10 Object Classes

| Index | Class Name     | BDD100K JSON key |
|-------|---------------|-----------------|
| 0     | Traffic Light  | `traffic light` |
| 1     | Traffic Sign   | `traffic sign`  |
| 2     | Car            | `car`           |
| 3     | Bus            | `bus`           |
| 4     | Person         | `person`        |
| 5     | Train          | `train`         |
| 6     | Truck          | `truck`         |
| 7     | Rider          | `rider`         |
| 8     | Motorcycle     | `motor`         |
| 9     | Cycle          | `bicycle`       |

---

## Project Structure

```
thesis-voc/
├── config.py              # All hyperparameters and paths
├── requirements.txt
├── train.py               # Main training entry point
├── evaluate.py            # Evaluation (BDD100K + cross-datasets)
├── inference.py           # Inference: image / video / webcam
│
├── dataset/
│   ├── bdd100k.py         # BDD100KDataset, CrossDatasetROI, get_dataloaders
│   └── transforms.py      # Train (augmented) and val/test transforms
│
├── models/
│   ├── vit.py             # PatchEmbedding, TransformerEncoderBlock, CustomViT
│   ├── detector.py        # YOLOv8Detector wrapper
│   └── pipeline.py        # ObjectClassificationPipeline (image + video)
│
├── engine/
│   ├── trainer.py         # Training loop — AdamW, warmup + cosine LR
│   └── evaluator.py       # Evaluation loop — accuracy, timing
│
├── utils/
│   ├── metrics.py         # Accuracy, confusion matrix, per-class P/R/F1
│   ├── visualization.py   # Training curves, confusion matrix, bar charts
│   └── checkpoint.py      # Save / load / resume checkpoints
│
├── scripts/
│   ├── prepare_acdc.py       # Convert ACDC → rois.json
│   ├── prepare_cadc.py       # Convert CADC → rois.json
│   ├── prepare_cityscapes.py # Convert Cityscapes → rois.json
│   └── prepare_once.py       # Convert ONCE → rois.json
│
├── checkpoints/           # Saved model weights (created at runtime)
├── logs/                  # Training logs (created at runtime)
└── outputs/               # Plots and result JSONs (created at runtime)
```

---

## Installation

```bash
# Python 3.11.9 (matches thesis environment)
pip install -r requirements.txt
```

Tested on:
- Python 3.11.9 | PyTorch 2.1+ | CUDA 12.x
- NVIDIA RTX 3090 (24 GB) | Ubuntu 22.04

---

## Dataset Setup

### BDD100K (primary)

```
data/bdd100k/
├── images/100k/
│   ├── train/        ← 70,000 images
│   └── val/          ← 10,000 images
└── labels/det_20/
    ├── det_train.json
    └── det_val.json
```

Download from: https://bdd-data.berkeley.edu/

### Cross-datasets (optional)

Each must be prepared before running `evaluate.py --all-datasets`:

```bash
python scripts/prepare_acdc.py        # data/acdc/
python scripts/prepare_cadc.py        # data/cadc/
python scripts/prepare_cityscapes.py  # data/cityscapes/
python scripts/prepare_once.py        # data/once/
```

Each script generates `data/<dataset>/rois.json` consumed by `CrossDatasetROI`.

---

## Training

```bash
# Full training (200 epochs, uses GPU if available)
python train.py

# Custom number of epochs (useful for testing)
python train.py --epochs 50

# Resume from a checkpoint
python train.py --resume checkpoints/epoch_050.pth

# Custom data path
python train.py --data-root /path/to/bdd100k
```

Training outputs:
- `checkpoints/best.pth` — best validation-loss checkpoint
- `checkpoints/epoch_NNN.pth` — periodic checkpoints (every 10 epochs)
- `outputs/training_history.json` — loss/accuracy per epoch
- `outputs/plots/training_curves.png`
- `outputs/plots/lr_schedule.png`
- `outputs/plots/confusion_matrix_bdd100k.png`
- `outputs/plots/per_class_metrics_bdd100k.png`
- `outputs/test_results.json`
- `logs/train.log`

---

## Evaluation

```bash
# BDD100K test split
python evaluate.py

# All five datasets
python evaluate.py --all-datasets

# Specific checkpoint
python evaluate.py --ckpt checkpoints/epoch_100.pth
```

---

## Inference

```bash
# Single image (print JSON results)
python inference.py --source image.jpg

# Single image with annotated output
python inference.py --source image.jpg --draw --save annotated.jpg

# Video file
python inference.py --source dashcam.mp4 --output out.mp4

# Live webcam
python inference.py --source 0

# CPU-only
python inference.py --source image.jpg --device cpu
```

---

## Hyperparameters

From thesis Tables 4.2 / 4.3 / 4.4 / 5.2:

| Parameter       | Value                 |
|----------------|----------------------|
| Epochs          | 200                  |
| Batch size      | 16                   |
| Learning rate   | 0.001                |
| Weight decay    | 1e-6                 |
| Optimizer       | AdamW                |
| Activation      | GELU                 |
| Embed dim       | 256                  |
| Attention heads | 8 (adjusted from 6)  |
| Encoder layers  | 8                    |
| MLP dim         | 512                  |
| Dropout         | 0.10                 |
| Patch size      | 16                   |
| Image size      | 224 × 224            |

LR schedule: Linear warmup (10 epochs) → CosineAnnealing (190 epochs, η_min = 1e-6)

---

## License

Research use only. BDD100K and cross-datasets are subject to their
respective licenses. YOLOv8 is licensed under AGPL-3.0 by Ultralytics.
