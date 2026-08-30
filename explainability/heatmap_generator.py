"""
Module for generating and normalizing heatmaps.

Handles color mapping of the raw grayscale Grad-CAM output into an RGB heatmap
suitable for overlaying onto the original medical image.
"""

import cv2
import numpy as np

from utils.logger import setup_logger

logger = setup_logger(__name__)


class HeatmapGenerator:
    """
    Converts raw 2D grayscale CAM arrays into coloured heatmaps.
    """

    @staticmethod
    def apply_colormap(grayscale_cam: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Apply a colormap to a grayscale CAM.

        Args:
            grayscale_cam (np.ndarray): 2D array [H, W] normalized between 0 and 1.
            colormap (int): OpenCV colormap identifier.

        Returns:
            np.ndarray: RGB heatmap array [H, W, 3] of dtype float32 (values 0-1).
        """
        # OpenCV applyColorMap requires 8-bit image (0-255)
        cam_8bit = np.uint8(255 * grayscale_cam)
        
        # Apply colormap (returns BGR)
        heatmap = cv2.applyColorMap(cam_8bit, colormap)
        
        # Convert BGR to RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Normalize to float32 between 0 and 1 for easier overlaying
        heatmap = np.float32(heatmap) / 255.0
        
        return heatmap

    @staticmethod
    def overlay_heatmap(heatmap: np.ndarray, original_image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Overlay the colored heatmap onto the original image.

        Args:
            heatmap (np.ndarray): The RGB heatmap [H, W, 3] (values 0-1).
            original_image (np.ndarray): The original RGB image [H, W, 3] (values 0-1).
            alpha (float): Transparency factor for the heatmap (0 to 1).

        Returns:
            np.ndarray: The blended image [H, W, 3] normalized between 0 and 1.
        """
        # Ensure dimensions match
        if heatmap.shape != original_image.shape:
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
            
        # Blend images
        overlay = (1 - alpha) * original_image + alpha * heatmap
        
        # Clip to ensure valid range
        overlay = np.clip(overlay, 0, 1)
        
        return overlay
