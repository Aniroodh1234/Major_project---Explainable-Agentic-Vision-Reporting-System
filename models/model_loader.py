"""
Model loader module for the Vision Transformer backbone.

Responsible for instantiating the pretrained ViT model, removing its
classification head, and exposing it purely as a feature extractor.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.model_config import DEVICE, FEATURE_DIM, MODEL_BACKBONE, PRETRAINED_WEIGHTS
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ViTFeatureExtractor(nn.Module):
    """
    Vision Transformer (ViT) wrapper for feature extraction.

    This class loads a pretrained ViT model (configurable via model_config.py),
    removes the final classification layer, and returns the deep feature
    embeddings.
    """

    def __init__(self) -> None:
        """
        Initialise the pretrained Vision Transformer without the classification head.
        """
        super().__init__()
        
        self.device = DEVICE
        logger.info(f"Loading pretrained backbone '{MODEL_BACKBONE}' on {self.device}...")

        # Load the base model dynamically from torchvision.models
        try:
            model_func = getattr(models, MODEL_BACKBONE)
            # Instantiate with specified weights
            self.backbone = model_func(weights=PRETRAINED_WEIGHTS)
        except AttributeError:
            msg = f"Model '{MODEL_BACKBONE}' is not available in torchvision.models."
            logger.error(msg)
            raise ValueError(msg)
        except Exception as e:
            logger.error(f"Failed to load model '{MODEL_BACKBONE}': {e}")
            raise

        # Remove the classification head.
        # In torchvision's ViT implementation, the classification head is named `heads`.
        if hasattr(self.backbone, "heads"):
            self.backbone.heads = nn.Identity()
            logger.debug("Replaced ViT 'heads' with nn.Identity()")
        elif hasattr(self.backbone, "fc"):
            self.backbone.fc = nn.Identity()
            logger.debug("Replaced ResNet-style 'fc' with nn.Identity()")
        else:
            logger.warning("Could not find standard classification head ('heads' or 'fc') to remove.")

        # Ensure the model is in evaluation mode (important for Dropout/BatchNorm layers if present)
        self.backbone.eval()
        
        # Move model to configured device
        self.backbone.to(self.device)
        
        logger.info(f"ViTFeatureExtractor initialised successfully. Expected output dim: {FEATURE_DIM}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from an input tensor.

        Args:
            x: Preprocessed image tensor of shape ``(B, C, H, W)`` or ``(C, H, W)``.
               Must already be normalized according to ImageNet standards.

        Returns:
            Extracted feature embeddings as a tensor of shape ``(B, FEATURE_DIM)``.
        """
        # Ensure input has batch dimension
        if x.ndim == 3:
            x = x.unsqueeze(0)
            
        x = x.to(self.device)
        
        # Forward pass through the modified backbone
        features = self.backbone(x)
        
        return features

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method to extract features (alias for forward).
        """
        return self.forward(x)


class MedicalClassifierViT(nn.Module):
    """
    Vision Transformer fine-tuned for medical image classification.

    Builds upon the ``ViTFeatureExtractor`` by adding a final classification
    linear layer matching the number of detected classes.
    """

    def __init__(self, num_classes: int) -> None:
        """
        Initialise the full classifier.

        Args:
            num_classes (int): Number of output classes (e.g., 2 for binary).
        """
        super().__init__()
        
        self.device = DEVICE
        self.num_classes = num_classes

        # 1. Base Feature Extractor (Backbone)
        self.feature_extractor = ViTFeatureExtractor()
        
        # 2. Classification Head
        # A simple linear layer mapping the 768-dim embeddings to the class logits.
        self.classifier = nn.Linear(FEATURE_DIM, num_classes)
        
        # Move the entire network to the configured device
        self.to(self.device)
        logger.info(f"MedicalClassifierViT initialised with {num_classes} classes on {self.device}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            x: Preprocessed image tensor of shape ``(B, C, H, W)``.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        x = x.to(self.device)
        
        # Extract deep features
        features = self.feature_extractor(x)
        
        # Pass features through the classification head
        logits = self.classifier(features)
        
        return logits
