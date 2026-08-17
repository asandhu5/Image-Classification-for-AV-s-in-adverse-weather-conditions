"""
dataset/transforms.py
─────────────────────
Train  → augmented pipeline  (flip, color jitter, affine, normalize)
Val/Test → clean pipeline    (resize + normalize only)

ImageNet mean/std are used because the ViT patch projection acts as a
feature extractor whose weights can benefit from pre-training statistics.
"""
import torchvision.transforms as T
from config import Config

# ImageNet normalisation statistics
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_train_transforms() -> T.Compose:
    """
    Augmented transform for training ROI crops.

    Pipeline
    --------
    1. Resize to IMG_SIZE × IMG_SIZE
    2. Random horizontal flip (p=0.5)
    3. Color jitter  — brightness, contrast, saturation, hue
    4. Random affine — small rotation, translation, scale
    5. ToTensor + Normalize (ImageNet stats)
    """
    return T.Compose([
        T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1,
        ),
        T.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
        ),
        T.ToTensor(),
        T.Normalize(mean=_MEAN, std=_STD),
    ])


def get_val_transforms() -> T.Compose:
    """
    Clean transform for validation / test / inference ROI crops.

    Pipeline
    --------
    1. Resize to IMG_SIZE × IMG_SIZE
    2. ToTensor + Normalize (ImageNet stats)
    """
    return T.Compose([
        T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=_MEAN, std=_STD),
    ])
