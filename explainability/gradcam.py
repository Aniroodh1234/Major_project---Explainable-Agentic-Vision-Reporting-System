"""
Grad-CAM generation module.

Uses the pytorch-grad-cam library to generate a class activation map (CAM)
for a given Vision Transformer model and image tensor.
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from utils.logger import setup_logger

logger = setup_logger(__name__)


def reshape_transform(tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
    """
    Reshape the output from the ViT backbone to be compatible with Grad-CAM.
    
    Vision Transformers typically output a sequence of tokens [B, N, D].
    We strip the class token and reshape the spatial tokens into a 2D grid.
    
    Args:
        tensor (torch.Tensor): The feature tensor from ViT block.
        height (int): The spatial height of the grid (224 / 16 = 14).
        width (int): The spatial width of the grid (224 / 16 = 14).
        
    Returns:
        torch.Tensor: Reshaped tensor [B, C, H, W].
    """
    # ViT output tensor shape is usually [B, Sequence_Length, Hidden_Dim]
    # For ViT-B/16 with 224x224 image, Sequence_Length is 197 (1 class token + 14x14 spatial patches)
    
    # Strip the CLS token (first token)
    result = tensor[:, 1:, :]
    
    # Reshape from [B, H*W, C] to [B, C, H, W]
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))
    
    # Bring the channels to the first dimension -> [B, C, H, W]
    result = result.transpose(2, 3).transpose(1, 2)
    logger.debug(f"Reshaped ViT spatial tokens from {tensor.shape} to {result.shape} for Grad-CAM.")
    return result


class GradCAMGenerator:
    """
    Generates Grad-CAM activation maps for Vision Transformer.
    """

    def __init__(self, model: nn.Module) -> None:
        """
        Initialize the Grad-CAM generator with the trained model.
        
        Args:
            model (nn.Module): The trained MedicalClassifierViT model.
        """
        self.model = model
        
        # Ensure the model is in eval mode and gradients are enabled for the CAM
        self.model.eval()

        try:
            # For torchvision's ViT-B/16, the target layers are typically the layer normalizations
            # before the attention blocks, or the last attention block itself.
            # We target the last encoder block's layer normalization.
            target_layers = [self.model.feature_extractor.backbone.encoder.layers[-1].ln_1]
            
            self.cam = GradCAM(
                model=self.model,
                target_layers=target_layers,
                reshape_transform=reshape_transform
            )
            logger.info("GradCAMGenerator initialized successfully targeting ViT last encoder block.")
        except Exception as e:
            logger.error(f"Failed to initialize Grad-CAM: {e}")
            raise

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Generate the Grad-CAM heatmap for a given input tensor.
        
        Args:
            input_tensor (torch.Tensor): The preprocessed image tensor [1, 3, 224, 224].
            target_class (int, optional): The class index to generate the CAM for. 
                                          If None, uses the highest scoring class.
                                          
        Returns:
            np.ndarray: The generated grayscale CAM [H, W] normalized between 0 and 1.
        """
        # Ensure gradients can be computed on the input
        input_tensor.requires_grad_(True)
        
        targets = None
        if target_class is not None:
            targets = [ClassifierOutputTarget(target_class)]
            
        try:
            logger.info(f"Computing Grad-CAM for class {target_class if target_class is not None else 'predicted'}...")
            logger.info(f"Extracting gradients from ViT layer: {self.cam.target_layers[0].__class__.__name__}")
            # Generate the raw grayscale CAM
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
            
            # Extract the first image from the batch
            grayscale_cam = grayscale_cam[0, :]
            
            logger.info(f"Successfully generated grayscale heatmap of shape: {grayscale_cam.shape}. Normalized between [0.0, 1.0]")
            return grayscale_cam
        except Exception as e:
            logger.error(f"Error generating Grad-CAM: {e}")
            raise
        finally:
            # Clean up
            input_tensor.requires_grad_(False)
