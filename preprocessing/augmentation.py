"""
Preprocessing module – Training Augmentation.

Provides configurable data augmentation transforms applied **only**
during training to improve model generalization.

All transforms operate on float32 NumPy arrays (post-normalization)
using OpenCV and NumPy operations, so they work correctly regardless
of the pixel value range.
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np

from utils.logger import setup_logger

logger = setup_logger(__name__)


class ImageAugmentor:
    """
    Apply random augmentation transforms to images during training.

    Supported transforms (all configurable via *config* dict):

    * Random Horizontal Flip
    * Random Rotation
    * Random Brightness / Contrast adjustment
    * Random Affine Transform (translation, scale, shear)

    Each transform has an independent probability of being applied.

    Args:
        config: Dictionary of augmentation hyper-parameters.
                See :data:`config.settings.AUGMENTATION_CONFIG` for keys.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialise augmentor from a configuration dictionary.

        Args:
            config: Augmentation settings dictionary.
        """
        self.horizontal_flip_prob: float = config.get("horizontal_flip_prob", 0.5)
        self.rotation_degrees: int = config.get("rotation_degrees", 15)
        self.rotation_prob: float = config.get("rotation_prob", 0.5)
        self.brightness_range: float = config.get("brightness_range", 0.2)
        self.contrast_range: float = config.get("contrast_range", 0.2)
        self.brightness_contrast_prob: float = config.get("brightness_contrast_prob", 0.5)
        self.affine_translate: tuple = config.get("affine_translate", (0.1, 0.1))
        self.affine_scale_range: tuple = config.get("affine_scale_range", (0.9, 1.1))
        self.affine_shear_degrees: float = config.get("affine_shear_degrees", 10)
        self.affine_prob: float = config.get("affine_prob", 0.5)

        logger.debug("ImageAugmentor initialised with config.")

    # ── Public API ──────────────────────────────────────────────────────

    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply all configured random augmentations to *image*.

        Each transform is applied independently with its own probability.

        Args:
            image: Float32 image array of shape ``(H, W, C)``.

        Returns:
            Augmented image (same shape and dtype).
        """
        image = self._random_horizontal_flip(image)
        image = self._random_rotation(image)
        image = self._random_brightness_contrast(image)
        image = self._random_affine(image)
        return image

    # ── Individual transforms ───────────────────────────────────────────

    def _random_horizontal_flip(self, image: np.ndarray) -> np.ndarray:
        """Flip the image horizontally with probability *horizontal_flip_prob*."""
        if random.random() < self.horizontal_flip_prob:
            return cv2.flip(image, 1)
        return image

    def _random_rotation(self, image: np.ndarray) -> np.ndarray:
        """Rotate the image by a random angle within ±*rotation_degrees*."""
        if random.random() >= self.rotation_prob:
            return image

        h, w = image.shape[:2]
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

        return cv2.warpAffine(
            image, M, (w, h),
            borderMode=cv2.BORDER_REFLECT_101,
            flags=cv2.INTER_LINEAR,
        )

    def _random_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Adjust brightness and contrast by random factors.

        * **Brightness**: adds a uniform offset drawn from
          ``[-brightness_range, +brightness_range]``.
        * **Contrast**: multiplies pixel values by a factor drawn from
          ``[1 − contrast_range, 1 + contrast_range]``.
        """
        if random.random() >= self.brightness_contrast_prob:
            return image

        brightness = random.uniform(-self.brightness_range, self.brightness_range)
        contrast = random.uniform(
            1.0 - self.contrast_range,
            1.0 + self.contrast_range,
        )

        image = image * contrast + brightness
        return image

    def _random_affine(self, image: np.ndarray) -> np.ndarray:
        """
        Apply a random affine transformation (translate + scale + shear).

        The transformation matrix is built from three independent
        random components, composed around the image centre so the
        result stays centred.
        """
        if random.random() >= self.affine_prob:
            return image

        h, w = image.shape[:2]
        cx, cy = w / 2, h / 2

        # Random parameters
        scale = random.uniform(*self.affine_scale_range)
        tx = random.uniform(-self.affine_translate[0], self.affine_translate[0]) * w
        ty = random.uniform(-self.affine_translate[1], self.affine_translate[1]) * h
        shear_deg = random.uniform(-self.affine_shear_degrees, self.affine_shear_degrees)

        # Build 3×3 affine matrix: T_back @ Shear @ Scale @ T_origin
        M = self._build_affine_matrix(cx, cy, scale, shear_deg, tx, ty)

        return cv2.warpAffine(
            image, M, (w, h),
            borderMode=cv2.BORDER_REFLECT_101,
            flags=cv2.INTER_LINEAR,
        )

    # ── Helper ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_affine_matrix(
        cx: float,
        cy: float,
        scale: float,
        shear_deg: float,
        tx: float,
        ty: float,
    ) -> np.ndarray:
        """
        Compose a 2×3 affine matrix from scale, shear, and translation.

        The transform is centred on ``(cx, cy)`` so the image does not
        drift to a corner.

        Returns:
            A ``(2, 3)`` float64 matrix suitable for :func:`cv2.warpAffine`.
        """
        shear_rad = math.radians(shear_deg)

        # Translate to origin
        T1 = np.array(
            [[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64,
        )
        # Scale
        S = np.array(
            [[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64,
        )
        # Shear (horizontal)
        SH = np.array(
            [[1, math.tan(shear_rad), 0], [0, 1, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        # Translate back + random translate
        T2 = np.array(
            [[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]],
            dtype=np.float64,
        )

        # Combined: T2 @ SH @ S @ T1
        M_full = T2 @ SH @ S @ T1
        return M_full[:2, :]  # 2×3 for cv2.warpAffine
