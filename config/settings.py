"""
Configuration settings for the Agentic AI Medical Image Analysis project.

All configurable paths and parameters are defined here.
Every agent reads its paths and settings from this module.
Do not hardcode paths anywhere else in the project.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Project Root
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Dataset Directories
# ---------------------------------------------------------------------------
DATASETS_DIR: Path = PROJECT_ROOT / "datasets"
RAW_DATASET_DIR: Path = DATASETS_DIR / "raw"
CLEANED_DATASET_DIR: Path = DATASETS_DIR / "cleaned"

# ---------------------------------------------------------------------------
# Output Structure
# ---------------------------------------------------------------------------
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
EXPORTS_DIR: Path = OUTPUTS_DIR / "exports"

# ---------------------------------------------------------------------------
# Supported Image Formats (lowercase, without dot)
# ---------------------------------------------------------------------------
SUPPORTED_IMAGE_FORMATS: list[str] = ["png", "jpg", "jpeg", "tif"]

# ---------------------------------------------------------------------------
# All Known Image Extensions (for distinguishing unsupported image formats
# from non-image files)
# ---------------------------------------------------------------------------
ALL_IMAGE_EXTENSIONS: list[str] = [
    "png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif", "webp",
    "svg", "ico", "raw", "cr2", "nef", "dng", "psd", "heic", "heif",
]

# ---------------------------------------------------------------------------
# Expected Class Folder Names inside raw/ and cleaned/
# ---------------------------------------------------------------------------
EXPECTED_CLASS_FOLDERS: list[str] = ["with_cancer", "without_cancer"]

# ---------------------------------------------------------------------------
# Image Hashing – used by Agent 1 for duplicate detection
# ---------------------------------------------------------------------------
IMAGE_HASH_SIZE: int = 8  # produces a 64-bit perceptual hash

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
LOG_DIR: Path = PROJECT_ROOT / "logs"
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ---------------------------------------------------------------------------
# Agent 2 – Preprocessing Configuration
# ---------------------------------------------------------------------------
PROCESSED_DATASET_DIR: Path = DATASETS_DIR / "processed"

# Target image dimensions for Vision Transformer
IMAGE_SIZE: int = 224
IMAGE_CHANNELS: int = 3  # RGB

# Normalization parameters (ImageNet defaults, standard for pretrained ViT)
NORMALIZATION_MEAN: list[float] = [0.485, 0.456, 0.406]
NORMALIZATION_STD: list[float] = [0.229, 0.224, 0.225]

# Augmentation configuration (applied only during training mode)
AUGMENTATION_CONFIG: dict = {
    "horizontal_flip_prob": 0.5,
    "rotation_degrees": 15,
    "rotation_prob": 0.5,
    "brightness_range": 0.2,
    "contrast_range": 0.2,
    "brightness_contrast_prob": 0.5,
    "affine_translate": (0.1, 0.1),
    "affine_scale_range": (0.9, 1.1),
    "affine_shear_degrees": 10,
    "affine_prob": 0.5,
}
