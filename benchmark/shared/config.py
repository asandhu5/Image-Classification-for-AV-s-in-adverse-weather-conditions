"""
shared/config.py
────────────────
Base configuration shared across all models in the benchmark.
Each model sub-package defines its own ModelConfig that inherits from here
and overrides only what it needs (IMG_SIZE, MODEL_NAME, dirs, etc.).
"""
from pathlib import Path
import torch


class SharedConfig:
    # ── Dataset ───────────────────────────────────────────────────────
    DATA_ROOT   = Path("data/bdd100k")
    IMG_DIR     = DATA_ROOT / "images" / "100k"
    LABEL_DIR   = DATA_ROOT / "labels" / "det_20"
    TRAIN_JSON  = LABEL_DIR / "det_train.json"

    # BDD100K annotation category → class index
    BDD_CLASSES = [
        "traffic light", "traffic sign", "car", "bus",
        "person", "train", "truck", "rider", "motor", "bicycle",
    ]
    CLASS_NAMES = [
        "Traffic Light", "Traffic Sign", "Car", "Bus",
        "Person", "Train", "Truck", "Rider", "Motorcycle", "Cycle",
    ]
    NUM_CLASSES = 10
    CLASS2IDX   = {c: i for i, c in enumerate(BDD_CLASSES)}
    IDX2CLASS   = {i: n for i, n in enumerate(CLASS_NAMES)}

    # ── Dataset splits ────────────────────────────────────────────────
    TRAIN_SPLIT  = 0.70
    VAL_SPLIT    = 0.10
    TEST_SPLIT   = 0.20
    MIN_BOX_AREA = 400
    MAX_ASPECT   = 10.0
    BOX_PADDING  = 0.05

    # ── Training hyperparameters (thesis Table 4.2 / 4.3 / 4.4) ─────
    EPOCHS        = 200
    BATCH_SIZE    = 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY  = 1e-6
    GRAD_CLIP     = 1.0
    WARMUP_EPOCHS = 10
    LABEL_SMOOTH  = 0.1
    SEED          = 42
    SAVE_EVERY    = 10

    # ── Hardware ──────────────────────────────────────────────────────
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY  = True

    IN_CHANNELS    = 3

    # ── Must be set by each model's ModelConfig ───────────────────────
    MODEL_NAME     = "base"
    IMG_SIZE       = 224
    CHECKPOINT_DIR = Path("checkpoints/base")
    OUTPUT_DIR     = Path("outputs/base")

    @classmethod
    def make_dirs(cls) -> None:
        for d in [cls.CHECKPOINT_DIR, cls.OUTPUT_DIR,
                  cls.OUTPUT_DIR / "plots"]:
            d.mkdir(parents=True, exist_ok=True)
