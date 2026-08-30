"""
Image utility functions for the Agentic AI project.

Provides helpers for image format validation, corruption detection
(via OpenCV), and perceptual hashing for duplicate detection.
"""

import cv2
import numpy as np
from pathlib import Path

from config.settings import SUPPORTED_IMAGE_FORMATS, IMAGE_HASH_SIZE
from utils.logger import setup_logger

logger = setup_logger(__name__)


def is_supported_format(file_path: Path) -> bool:
    """
    Check whether a file has a supported image extension.

    Comparison is **case-insensitive** (e.g. ``.JPG`` is treated as
    ``.jpg``).

    Args:
        file_path: Path to the file.

    Returns:
        ``True`` if the extension (without the dot, lowercased) is in
        :data:`config.settings.SUPPORTED_IMAGE_FORMATS`.
    """
    extension = file_path.suffix.lstrip(".").lower()
    return extension in SUPPORTED_IMAGE_FORMATS


def is_valid_image(file_path: Path) -> bool:
    """
    Attempt to open an image with OpenCV to verify it is not corrupted.

    An image is considered **valid** if :func:`cv2.imread` returns a
    non-``None`` array with at least one pixel.

    Args:
        file_path: Path to the image file.

    Returns:
        ``True`` if the image can be successfully decoded by OpenCV.
    """
    try:
        image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return False
        if image.size == 0:
            return False
        return True
    except Exception as e:
        logger.debug(f"Image validation failed for {file_path}: {e}")
        return False


def compute_image_hash(
    file_path: Path,
    hash_size: int = IMAGE_HASH_SIZE,
) -> str | None:
    """
    Compute a **difference hash (dHash)** of the image for duplicate detection.

    Algorithm
    ---------
    1. Read the image in grayscale.
    2. Resize to ``(hash_size + 1) × hash_size`` pixels.
    3. For each row, compare each pixel with its right neighbour.
    4. Encode the resulting boolean matrix as a hexadecimal string.

    With the default *hash_size* of 8 the hash is 64 bits (16 hex chars).

    Args:
        file_path: Path to the image file.
        hash_size: Grid size for hashing (default ``8`` → 64-bit hash).

    Returns:
        Hexadecimal hash string, or ``None`` if hashing fails.
    """
    try:
        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None

        # Resize to (width=hash_size+1, height=hash_size)
        resized = cv2.resize(
            image,
            (hash_size + 1, hash_size),
            interpolation=cv2.INTER_AREA,
        )

        # Compare each pixel with its right neighbour → boolean matrix
        diff: np.ndarray = resized[:, 1:] > resized[:, :-1]

        # Pack booleans into a single integer, then format as hex
        hash_value: int = 0
        for bit in diff.flatten():
            hash_value = (hash_value << 1) | int(bit)

        hex_length = (hash_size * hash_size + 3) // 4  # bits → hex chars
        return format(hash_value, f"0{hex_length}x")

    except Exception as e:
        logger.debug(f"Failed to compute hash for {file_path}: {e}")
        return None


# ── Image loading & colour conversion (Agent 2+) ───────────────────────────

def load_image(file_path: Path) -> np.ndarray | None:
    """
    Load an image from disk using OpenCV.

    Returns the raw image as a NumPy array (BGR colour order, or
    grayscale) exactly as OpenCV reads it.

    Args:
        file_path: Path to the image file.

    Returns:
        NumPy array of the image, or ``None`` if the read fails.
    """
    try:
        image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            logger.warning(f"Failed to load image: {file_path}")
            return None
        return image
    except Exception as e:
        logger.error(f"Error loading image {file_path}: {e}")
        return None


def convert_color_format(
    image: np.ndarray,
    target_channels: int = 3,
) -> np.ndarray:
    """
    Convert an image to RGB format with the specified number of channels.

    Handles the following input scenarios:

    * **Grayscale** (2-D array) → replicated to 3-channel RGB.
    * **BGRA** (4 channels) → converted to 3-channel RGB.
    * **BGR**  (3 channels) → converted to RGB.

    This ensures every image follows the same colour format before
    it enters the preprocessing pipeline, supporting both X-ray
    (often grayscale) and ultrasound images.

    Args:
        image:           Input image as a NumPy array.
        target_channels: Desired number of output channels (default 3).

    Returns:
        Image in RGB format with *target_channels* channels.
    """
    if image.ndim == 2:
        # Grayscale → 3-channel RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        # BGRA → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        # BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image
