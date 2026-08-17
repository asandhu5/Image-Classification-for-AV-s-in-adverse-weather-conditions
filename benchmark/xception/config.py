import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.config import SharedConfig

class ModelConfig(SharedConfig):
    MODEL_NAME     = "Xception"
    IMG_SIZE       = 299   # Xception's native resolution
    CHECKPOINT_DIR = Path("checkpoints/xception")
    OUTPUT_DIR     = Path("outputs/xception")
