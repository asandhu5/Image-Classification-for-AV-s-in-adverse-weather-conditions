import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.config import SharedConfig

class ModelConfig(SharedConfig):
    MODEL_NAME     = "VGG16"
    IMG_SIZE       = 224
    CHECKPOINT_DIR = Path("checkpoints/vgg16")
    OUTPUT_DIR     = Path("outputs/vgg16")
