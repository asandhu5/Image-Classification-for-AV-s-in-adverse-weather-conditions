import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.config import SharedConfig

class ModelConfig(SharedConfig):
    MODEL_NAME     = "InceptionV3"
    IMG_SIZE       = 299   # Inception's native resolution
    CHECKPOINT_DIR = Path("checkpoints/inceptionv3")
    OUTPUT_DIR     = Path("outputs/inceptionv3")
