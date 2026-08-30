"""
Preprocessing module – Tensor Conversion.

Converts NumPy image arrays to PyTorch tensors in the ``(C, H, W)``
format expected by Vision Transformer models, and provides helpers
for saving / loading tensors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from utils.logger import setup_logger

logger = setup_logger(__name__)


class TensorConverter:
    """
    Convert preprocessed NumPy images to PyTorch tensors and
    manage tensor persistence (save / load).
    """

    @staticmethod
    def to_tensor(image: np.ndarray) -> torch.Tensor:
        """
        Convert an ``(H, W, C)`` float32 NumPy array to a
        ``(C, H, W)`` PyTorch float tensor.

        Args:
            image: Preprocessed image of shape ``(H, W, C)``.

        Returns:
            ``torch.float32`` tensor of shape ``(C, H, W)``.
        """
        # Ensure contiguous memory for efficient conversion
        image = np.ascontiguousarray(image)
        tensor = torch.from_numpy(image)

        if tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)  # HWC → CHW
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)  # HW → 1HW

        return tensor

    @staticmethod
    def save_tensor(tensor: torch.Tensor, path: Path) -> None:
        """
        Save a tensor to disk as a ``.pt`` file.

        Parent directories are created automatically.

        Args:
            tensor: The tensor to persist.
            path:   Destination file path (should end with ``.pt``).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, path)

    @staticmethod
    def load_tensor(path: Path) -> torch.Tensor:
        """
        Load a tensor from a ``.pt`` file.

        Args:
            path: Path to the ``.pt`` file.

        Returns:
            The loaded tensor.
        """
        return torch.load(path, weights_only=True)
