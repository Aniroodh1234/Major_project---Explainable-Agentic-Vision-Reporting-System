"""
Visualization module for Explainable AI (XAI).

Creates comprehensive, side-by-side graphical structures to illustrate the
model's decision-making process based on the Grad-CAM outputs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.file_utils import ensure_directory_exists
from utils.logger import setup_logger

logger = setup_logger(__name__)


def export_explainability_visualization(
    original_image: np.ndarray,
    preprocessed_image: np.ndarray,
    grayscale_cam: np.ndarray,
    overlay_image: np.ndarray,
    predicted_class: str,
    confidence: float,
    output_path: Path
) -> None:
    """
    Generate and save a multi-panel figure displaying the XAI insights.

    Args:
        original_image (np.ndarray): Original RGB image (0-255).
        preprocessed_image (np.ndarray): Denormalized RGB tensor representation (0-1).
        grayscale_cam (np.ndarray): Raw CAM activation map (0-1).
        overlay_image (np.ndarray): Blended original image and heatmap (0-1).
        predicted_class (str): The class name predicted by the model.
        confidence (float): The confidence probability (0-1).
        output_path (Path): Path to save the final PNG figure.
    """
    ensure_directory_exists(output_path.parent)

    # Set up a 1x4 horizontal grid for the visualizations
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle(
        f"Model Explainability (Grad-CAM) | Prediction: {predicted_class} ({confidence:.1%})",
        fontsize=16, fontweight='bold', y=0.98
    )

    # 1. Original Image
    axes[0].imshow(original_image)
    axes[0].set_title("1. Original Image")
    axes[0].axis("off")

    # 2. Preprocessed Image (Model Input)
    axes[1].imshow(preprocessed_image)
    axes[1].set_title("2. Preprocessed Input")
    axes[1].axis("off")

    # 3. Raw Grad-CAM
    # Use the 'jet' colormap natively in matplotlib for the raw map
    im = axes[2].imshow(grayscale_cam, cmap='jet')
    axes[2].set_title("3. Raw Activation Map")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Final Overlay
    axes[3].imshow(overlay_image)
    axes[3].set_title("4. Grad-CAM Overlay")
    axes[3].axis("off")

    plt.tight_layout()
    
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Explainability visualization saved successfully: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save visualization at {output_path}: {e}")
    finally:
        plt.close(fig)
