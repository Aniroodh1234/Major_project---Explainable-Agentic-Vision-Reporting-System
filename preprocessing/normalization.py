"""
Preprocessing module – Pixel Normalization.

Scales pixel values from ``[0, 255]`` uint8 to ``[0, 1]`` float32 and
applies channel-wise mean/std normalization (ImageNet defaults for ViT).
"""

from __future__ import annotations

import numpy as np

from utils.logger import setup_logger

logger = setup_logger(__name__)


class ImageNormalizer:
    """
    Normalize image pixel values for Vision Transformer input.

    Two-stage normalization:

    1. **Scale** – convert ``uint8 [0, 255]`` → ``float32 [0, 1]``.
    2. **Standardize** – subtract channel mean and divide by channel std
       (default: ImageNet statistics).

    Args:
        mean: Per-channel mean values (length must match channel count).
        std:  Per-channel std values  (length must match channel count).
    """

    def __init__(
        self,
        mean: list[float],
        std: list[float],
    ) -> None:
        """
        Initialise the normalizer with mean and std.

        Args:
            mean: Per-channel means, e.g. ``[0.485, 0.456, 0.406]``.
            std:  Per-channel stds,  e.g. ``[0.229, 0.224, 0.225]``.
        """
        self.mean: np.ndarray = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
        self.std: np.ndarray = np.array(std, dtype=np.float32).reshape(1, 1, -1)
        logger.debug(f"ImageNormalizer initialised: mean={mean}, std={std}")

    def scale_to_float(self, image: np.ndarray) -> np.ndarray:
        """
        Convert ``uint8 [0, 255]`` to ``float32 [0, 1]``.

        Args:
            image: Input image (uint8).

        Returns:
            Float32 image with pixel values in ``[0, 1]``.
        """
        return image.astype(np.float32) / 255.0

    def apply_standardization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply channel-wise mean/std normalization.

        Assumes the input is already in ``[0, 1]`` float range.

        Args:
            image: Float32 image in ``[0, 1]`` range.

        Returns:
            Normalized image (approximately zero-centred, unit variance
            per channel).
        """
        return (image - self.mean) / self.std

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Full normalization: scale to ``[0, 1]`` then standardize.

        Convenience method that chains :meth:`scale_to_float` and
        :meth:`apply_standardization`.

        Args:
            image: Input image (uint8, ``[0, 255]``).

        Returns:
            Fully normalized float32 image.
        """
        image = self.scale_to_float(image)
        image = self.apply_standardization(image)
        return image
