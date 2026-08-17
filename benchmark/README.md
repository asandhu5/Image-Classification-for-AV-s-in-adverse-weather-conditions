# Benchmark Suite — Baseline Models for Adverse-Weather Object Classification

Nine state-of-the-art classifiers trained and evaluated under **identical
conditions** to benchmark the proposed CustomViT model from my M.Sc. thesis
(NUST, 2025): same BDD100K ROI dataset, same 70/10/20 split, same optimizer,
schedule, and 200-epoch budget.

Companion to the proposed model in [`../proposed/`](../proposed/)
(standalone repo: `Classification-on-adverse-weather-conditions`).

---

## Models

| Directory | Model | Params (M) | FLOPs (G) | Input |
|-----------|-------|-----------|-----------|-------|
| `vgg16/` | VGG16 | 136.90 | 15.5 | 224² |
| `resnet101/` | ResNet-101 | 47.16 | 7.6 | 224² |
| `inceptionv3/` | InceptionV3 | 29.58 | 5.7 | 299² |
| `xception/` | Xception | 22.95 | 8.4 | 299² |
| `vit_b16/` | ViT-B/16 | 76.41 | 14.87 | 224² |
| `vit_b32/` | ViT-B/32 | 81.74 | 37.89 | 224² |
| `vit_l16/` | ViT-L/16 | 273.74 | 12.75 | 224² |
| `vit_l32/` | ViT-L/32 | 287.11 | 44.11 | 224² |
| `vit_h14/` | ViT-H/14 | 518.74 | 111.02 | 224² |

ViT variants follow the architectural parameters of Dosovitskiy et al.
(thesis Table 5.3); CNN baselines are implemented per their original papers.

---

## Results (BDD100K test set, thesis Table 5.4)

| Model | Eval time (s) | Test accuracy |
|-------|---------------|---------------|
| VGG16 | 0.9935 | 86.75% |
| ResNet-101 | 0.8413 | 84.23% |
| InceptionV3 | 0.8067 | 85.16% |
| Xception | 0.9509 | 87.23% |
| ViT-B/16 | 0.8307 | 81.61% |
| ViT-B/32 | 0.8874 | 80.41% |
| ViT-L/16 | 1.0780 | 78.14% |
| ViT-L/32 | 1.1110 | 81.07% |
| ViT-H/14 | 1.4360 | 82.46% |
| **CustomViT (proposed)** | **0.5046** | **91.47%** |

---

## Structure

```
benchmark/
├── shared/                # Code common to all nine models
│   ├── config.py          # SharedConfig — dataset, splits, training HPs
│   ├── dataset/           # BDD100K ROI dataset + transforms
│   ├── engine/            # Trainer (AdamW, warmup + cosine) and evaluator
│   ├── models/vit_base.py # Parameterized ViT used by all five ViT variants
│   └── utils/             # Metrics, visualization, checkpointing
│
└── <model>/               # One folder per baseline, all with the same API:
    ├── config.py          # ModelConfig(SharedConfig) — overrides per model
    ├── model.py           # get_model() → nn.Module
    ├── train.py           # Training entry point
    ├── evaluate.py        # Test-set evaluation
    └── inference.py       # Single-image inference
```

Every model folder is interchangeable: `python <model>/train.py`,
`python <model>/evaluate.py --ckpt ...`, `python <model>/inference.py --source img.jpg`.

---

## Setup

```bash
# Python 3.11.9
pip install -r requirements.txt
```

### Dataset (BDD100K)

```
data/bdd100k/
├── images/100k/
│   ├── train/        ← 70,000 images
│   └── val/          ← 10,000 images
└── labels/det_20/
    ├── det_train.json
    └── det_val.json
```

Download from https://bdd-data.berkeley.edu/. ROIs are cropped from
detection boxes into 10 classes (traffic light, traffic sign, car, bus,
person, train, truck, rider, motorcycle, cycle) with the same filtering as
the proposed model (min box area 400 px², max aspect 10, 5% padding).

---

## Training

```bash
python vit_b16/train.py                # any model, same interface
python vit_b16/train.py --epochs 50    # shorter run
python vit_b16/train.py --resume checkpoints/vit_b16/epoch_050.pth
```

Shared hyperparameters (thesis Tables 4.2–4.4): 200 epochs, batch 16,
AdamW (lr 1e-3, wd 1e-6), 10-epoch linear warmup → cosine annealing,
label smoothing 0.1, gradient clipping 1.0, seed 42.

Outputs land in `checkpoints/<model>/` and `outputs/<model>/`
(training curves, confusion matrix, per-class metrics, results JSON).

Note on VRAM: ViT-L and ViT-H variants are large; on GPUs below 24 GB,
reduce `BATCH_SIZE` in the model's `config.py` (gradient checkpointing is
available in `shared/models/vit_base.py`).

---

## Evaluation

```bash
python xception/evaluate.py                          # best checkpoint
python xception/evaluate.py --ckpt path/to/ckpt.pth  # specific checkpoint
```

Reports test accuracy, per-class precision/recall/F1, confusion matrix,
and average evaluation time — the numbers reported in thesis Table 5.4.

---

## Environment

- Python 3.11.9 | PyTorch 2.1+ | CUDA 12.x
- NVIDIA RTX 3090 (24 GB) | Ubuntu 22.04

## License

Research use only. BDD100K is subject to its own license.
