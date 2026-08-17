import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.config import SharedConfig

class ModelConfig(SharedConfig):
    MODEL_NAME     = "ViT-H/14"
    IMG_SIZE       = 224
    IN_CHANNELS    = 3
    CHECKPOINT_DIR = Path("checkpoints/vit_h14")
    OUTPUT_DIR     = Path("outputs/vit_h14")
    # Large ViT variants may need a smaller batch size on GPUs < 24 GB
    BATCH_SIZE     = 8 if True else 16
