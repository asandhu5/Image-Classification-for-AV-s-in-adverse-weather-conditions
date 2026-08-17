"""
config.py
─────────
Single source of truth for every hyperparameter, path, and constant
used across the project.  Import Config from here — never hard-code values
anywhere else.
"""
from pathlib import Path
import torch


class Config:
    # ══════════════════════════════════════════════════════════════════
    # Paths
    # ══════════════════════════════════════════════════════════════════
    DATA_ROOT      = Path("data/bdd100k")
    IMG_DIR        = DATA_ROOT / "images" / "100k"
    LABEL_DIR      = DATA_ROOT / "labels" / "det_20"
    TRAIN_JSON     = LABEL_DIR / "det_train.json"
    VAL_JSON       = LABEL_DIR / "det_val.json"

    CHECKPOINT_DIR = Path("checkpoints")
    LOG_DIR        = Path("logs")
    OUTPUT_DIR     = Path("outputs")

    # ══════════════════════════════════════════════════════════════════
    # Classes
    # BDD100K JSON uses: "motor" for motorcycle, "bicycle" for cycle
    # ══════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════
    # Dataset
    # ══════════════════════════════════════════════════════════════════
    IMG_SIZE     = 224          # ViT input (H = W)
    PATCH_SIZE   = 16           # → 196 patches per ROI
    TRAIN_SPLIT  = 0.70
    VAL_SPLIT    = 0.10
    TEST_SPLIT   = 0.20
    MIN_BOX_AREA = 400          # px² — discard tiny / noisy boxes
    MAX_ASPECT   = 10.0         # discard extreme aspect-ratio crops
    BOX_PADDING  = 0.05         # fractional padding added around each crop

    # ══════════════════════════════════════════════════════════════════
    # Model  (Table 5.2 from thesis)
    # ══════════════════════════════════════════════════════════════════
    EMBED_DIM  = 256    # D_model
    # Thesis specifies 6 heads, but 256 % 6 ≠ 0.
    # Adjusted to 8 (256 / 8 = 32 per head) — nearest valid choice.
    NUM_HEADS  = 8
    NUM_LAYERS = 8      # Transformer encoder layers
    MLP_DIM    = 512    # FFN hidden dim  (D_mlp) — also used in cls head
    DROPOUT    = 0.10
    IN_CHANNELS = 3

    # ══════════════════════════════════════════════════════════════════
    # Training  (Table 4.2 / 4.3 / 4.4 from thesis)
    # ══════════════════════════════════════════════════════════════════
    EPOCHS        = 200
    BATCH_SIZE    = 16
    LEARNING_RATE = 1e-3        # AdamW LR
    WEIGHT_DECAY  = 1e-6        # AdamW weight decay
    GRAD_CLIP     = 1.0         # gradient-norm clip
    WARMUP_EPOCHS = 10          # linear LR warmup before cosine decay
    LABEL_SMOOTH  = 0.1         # CrossEntropyLoss label smoothing

    # ══════════════════════════════════════════════════════════════════
    # YOLOv8m
    # ══════════════════════════════════════════════════════════════════
    YOLO_WEIGHTS  = "yolov8m.pt"
    YOLO_CONF     = 0.25
    YOLO_IOU      = 0.45
    YOLO_IMG_SIZE = 640

    # ══════════════════════════════════════════════════════════════════
    # Hardware  (Table 4.5 from thesis)
    # ══════════════════════════════════════════════════════════════════
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY  = True

    # ══════════════════════════════════════════════════════════════════
    # Misc
    # ══════════════════════════════════════════════════════════════════
    SEED       = 42
    SAVE_EVERY = 10     # checkpoint every N epochs

    @classmethod
    def make_dirs(cls) -> None:
        """Create all output directories."""
        for d in [cls.CHECKPOINT_DIR, cls.LOG_DIR,
                  cls.OUTPUT_DIR, cls.OUTPUT_DIR / "plots"]:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def summary(cls) -> str:
        lines = [
            "=" * 55,
            "  Visual Object Classification — Config",
            "=" * 55,
            f"  Device       : {cls.DEVICE}",
            f"  Classes      : {cls.NUM_CLASSES}",
            f"  IMG_SIZE     : {cls.IMG_SIZE}",
            f"  PATCH_SIZE   : {cls.PATCH_SIZE}",
            f"  EMBED_DIM    : {cls.EMBED_DIM}",
            f"  NUM_HEADS    : {cls.NUM_HEADS}",
            f"  NUM_LAYERS   : {cls.NUM_LAYERS}",
            f"  MLP_DIM      : {cls.MLP_DIM}",
            f"  DROPOUT      : {cls.DROPOUT}",
            f"  EPOCHS       : {cls.EPOCHS}",
            f"  BATCH_SIZE   : {cls.BATCH_SIZE}",
            f"  LR           : {cls.LEARNING_RATE}",
            f"  WEIGHT_DECAY : {cls.WEIGHT_DECAY}",
            "=" * 55,
        ]
        return "\n".join(lines)
