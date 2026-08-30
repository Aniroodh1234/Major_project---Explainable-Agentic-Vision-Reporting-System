"""
Preprocessing module – Image Resizing.

Resizes images to a uniform target dimension required by the
Vision Transformer backbone.
"""

from __future__ import annotations

import cv2
import numpy as np

from utils.logger import setup_logger

logger = setup_logger(__name__)


class ImageResizer:
    """
    Resize images to a fixed square dimension.

    All images are resized to ``(target_size, target_size)`` using
    high-quality ``INTER_AREA`` interpolation (for down-scaling) or
    ``INTER_LINEAR`` (for up-scaling).

    Args:
        target_size: The target height and width in pixels.
    """

    def __init__(self, target_size: int) -> None:
        """
        Initialise the resizer.

        Args:
            target_size: Target height/width in pixels (e.g. 224).
        """
        self.target_size: int = target_size
        logger.debug(f"ImageResizer initialised: target_size={target_size}")

    def resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize *image* to ``(target_size, target_size)``.

        Automatically selects the best interpolation method based
        on whether the image is being enlarged or shrunk.

        Args:
            image: Input image as a NumPy array (H, W) or (H, W, C).

        Returns:
            Resized image with shape ``(target_size, target_size[, C])``.
        """
        h, w = image.shape[:2]

        # Choose interpolation: INTER_AREA for shrinking, INTER_LINEAR for enlarging
        if h > self.target_size or w > self.target_size:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR

        resized = cv2.resize(
            image,
            (self.target_size, self.target_size),
            interpolation=interpolation,
        )
        return resized
