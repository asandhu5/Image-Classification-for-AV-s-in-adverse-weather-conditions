import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.config import SharedConfig

class ModelConfig(SharedConfig):
    MODEL_NAME     = "ViT-B/16"
    IMG_SIZE       = 224
    IN_CHANNELS    = 3
    CHECKPOINT_DIR = Path("checkpoints/vit_b16")
    OUTPUT_DIR     = Path("outputs/vit_b16")
    # Large ViT variants may need a smaller batch size on GPUs < 24 GB
    BATCH_SIZE     = 16
