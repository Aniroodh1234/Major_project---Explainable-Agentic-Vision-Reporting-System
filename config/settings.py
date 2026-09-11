from pathlib import Path



PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


DATASETS_DIR: Path = PROJECT_ROOT / "datasets"
RAW_DATASET_DIR: Path = DATASETS_DIR / "raw"
CLEANED_DATASET_DIR: Path = DATASETS_DIR / "cleaned"


OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
EXPORTS_DIR: Path = OUTPUTS_DIR / "exports"


SUPPORTED_IMAGE_FORMATS: list[str] = ["png", "jpg", "jpeg", "tif"]


ALL_IMAGE_EXTENSIONS: list[str] = [
    "png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif", "webp",
    "svg", "ico", "raw", "cr2", "nef", "dng", "psd", "heic", "heif",
]



EXPECTED_CLASS_FOLDERS: list[str] = ["with_cancer", "without_cancer"]


IMAGE_HASH_SIZE: int = 8  # produces a 64-bit perceptual hash


LOG_DIR: Path = PROJECT_ROOT / "logs"
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


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
