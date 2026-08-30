"""
Utility module for validating extracted feature embeddings.

Provides functions and classes to ensure feature integrity, correct
dimensions, and absence of corrupted data (NaN, Inf) before model fine-tuning.
"""

from typing import Tuple

import torch

from utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureValidator:
    """
    Validates a PyTorch feature embedding tensor.

    Ensures that:
    - The tensor has the correct shape.
    - The tensor dtype is float32.
    - The tensor contains no NaN values.
    - The tensor contains no Infinity values.
    """

    def __init__(self, expected_dim: int) -> None:
        """
        Initialize the validator with the expected feature dimension.

        Args:
            expected_dim (int): The expected size of the feature embedding.
        """
        self.expected_dim = expected_dim
        # Accept either [1, expected_dim] or [expected_dim]
        self.valid_shapes = [(1, expected_dim), (expected_dim,)]

    def validate(self, tensor: torch.Tensor) -> Tuple[bool, str]:
        """
        Validate the feature tensor.

        Args:
            tensor (torch.Tensor): The PyTorch tensor to validate.

        Returns:
            Tuple[bool, str]: A boolean indicating validity, and an error message
            if invalid (empty string if valid).
        """
        if not isinstance(tensor, torch.Tensor):
            return False, "Input is not a PyTorch tensor."

        if tensor.dtype != torch.float32:
            return False, f"Invalid dtype. Expected float32, got {tensor.dtype}."

        shape = tuple(tensor.shape)
        if shape not in self.valid_shapes:
            return False, f"Invalid shape. Expected one of {self.valid_shapes}, got {shape}."

        if torch.isnan(tensor).any().item():
            return False, "Tensor contains NaN values."

        if torch.isinf(tensor).any().item():
            return False, "Tensor contains Infinity values."

        return True, ""
