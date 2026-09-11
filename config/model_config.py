import torch
from pathlib import Path

from config.settings import DATASETS_DIR


FEATURES_DATASET_DIR: Path = DATASETS_DIR / "features"
SELECTED_FEATURES_DATASET_DIR: Path = DATASETS_DIR / "selected_features"


# We use the standard Vision Transformer Base (ViT-B/16) model.
MODEL_BACKBONE: str = "vit_b_16"
PRETRAINED_WEIGHTS: str = "DEFAULT"  # Use IMAGENET1K_V1 default weights

# The output dimension of the ViT-B/16 backbone before the classification head
FEATURE_DIM: int = 768

# Automatically select GPU if available, else fall back to CPU
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BATCH_SIZE: int = 32
LEARNING_RATE: float = 1e-4
NUM_EPOCHS: int = 20
PATIENCE: int = 5

OPTIMIZER: str = "AdamW"
SCHEDULER: str = "ReduceLROnPlateau"


from config.settings import OUTPUTS_DIR, PROJECT_ROOT

MODELS_DIR: Path = PROJECT_ROOT / "models"
CHECKPOINTS_DIR: Path = MODELS_DIR / "checkpoints"
TRAINED_MODEL_DIR: Path = MODELS_DIR / "trained"
FINAL_MODEL_PATH: Path = TRAINED_MODEL_DIR / "medical_classifier.pt"


CONFIDENCE_THRESHOLD: float = 0.80

HEATMAPS_DIR: Path = OUTPUTS_DIR / "heatmaps"
PREDICTIONS_DIR: Path = OUTPUTS_DIR / "predictions"
