"""
Preprocessing module – Preprocessing Pipeline.

Chains all preprocessing steps (resize → colour convert → normalize →
augment → tensor) into a single, reusable pipeline class.

This pipeline is used by:

* **Agent 2** – batch-process cleaned images and save to ``datasets/processed/``.
* **Agent 5** – apply on-the-fly augmentation during training.
* **Agent 6** – preprocess a single image during inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import (
    AUGMENTATION_CONFIG,
    IMAGE_SIZE,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
)
from preprocessing.augmentation import ImageAugmentor
from preprocessing.normalization import ImageNormalizer
from preprocessing.resize import ImageResizer
from preprocessing.tensor_conversion import TensorConverter
from utils.image_utils import convert_color_format, load_image
from utils.logger import setup_logger

logger = setup_logger(__name__)


class PreprocessingPipeline:
    """
    End-to-end image preprocessing pipeline for ViT-based training
    and inference.

    Pipeline steps (executed in order):

    1. **Load** image from disk (or accept a pre-loaded array).
    2. **Resize** to ``(IMAGE_SIZE, IMAGE_SIZE)``.
    3. **Convert colour** format to RGB.
    4. **Normalize** pixel values (scale + ImageNet mean/std).
    5. **Augment** *(training mode only)*.
    6. **Convert** to PyTorch tensor ``(C, H, W)``.

    Modes
    -----
    * ``"train"``     – augmentation enabled.
    * ``"val"``       – augmentation disabled.
    * ``"test"``      – augmentation disabled.
    * ``"inference"`` – augmentation disabled (default).

    Args:
        mode: One of ``"train"``, ``"val"``, ``"test"``, ``"inference"``.
    """

    VALID_MODES = {"train", "val", "test", "inference"}

    def __init__(self, mode: str = "inference") -> None:
        """
        Initialise the pipeline.

        Args:
            mode: Pipeline mode controlling augmentation behaviour.

        Raises:
            ValueError: If *mode* is not one of the valid modes.
        """
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of {self.VALID_MODES}."
            )

        self.mode: str = mode

        # Sub-modules
        self.resizer = ImageResizer(IMAGE_SIZE)
        self.normalizer = ImageNormalizer(NORMALIZATION_MEAN, NORMALIZATION_STD)
        self.augmentor = (
            ImageAugmentor(AUGMENTATION_CONFIG) if mode == "train" else None
        )
        self.converter = TensorConverter()

        logger.info(
            f"PreprocessingPipeline initialised: mode='{mode}', "
            f"image_size={IMAGE_SIZE}, "
            f"augmentation={'ON' if self.augmentor else 'OFF'}"
        )

    # ── Public API ──────────────────────────────────────────────────────

    def process(self, image: np.ndarray) -> torch.Tensor:
        """
        Run all preprocessing steps on a pre-loaded image array.

        Args:
            image: Raw image as a NumPy array (BGR uint8 from OpenCV).

        Returns:
            Preprocessed ``(C, H, W)`` float32 tensor.
        """
        # Step 2: Resize
        image = self.resizer.resize(image)

        # Step 3: Colour conversion (BGR/gray → RGB)
        image = convert_color_format(image)

        # Step 4: Normalize
        image = self.normalizer.normalize(image)

        # Step 5: Augment (training only)
        if self.augmentor is not None:
            image = self.augmentor.augment(image)

        # Step 6: Convert to tensor
        tensor = self.converter.to_tensor(image)

        return tensor

    def process_from_path(self, file_path: Path) -> torch.Tensor | None:
        """
        Load an image from *file_path* and run the full pipeline.

        This is the primary entry point for **Agent 6 inference** and
        convenient for any single-image processing.

        Args:
            file_path: Absolute or relative path to an image file.

        Returns:
            Preprocessed ``(C, H, W)`` tensor, or ``None`` if loading
            fails.
        """
        # Step 1: Load
        image = load_image(file_path)
        if image is None:
            logger.warning(f"Skipping (load failed): {file_path}")
            return None

        return self.process(image)
