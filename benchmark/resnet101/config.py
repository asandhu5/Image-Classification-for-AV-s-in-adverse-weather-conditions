import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.config import SharedConfig

class ModelConfig(SharedConfig):
    MODEL_NAME     = "ResNet-101"
    IMG_SIZE       = 224
    CHECKPOINT_DIR = Path("checkpoints/resnet101")
    OUTPUT_DIR     = Path("outputs/resnet101")
