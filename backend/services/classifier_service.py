"""
Classifier Service for Agent 6 (Inference).

Provides a high-level API to process a single medical image, predict its
class using the fine-tuned MedicalClassifierViT, and generate Grad-CAM
explainability visualizations.
"""

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch

from config.model_config import CONFIDENCE_THRESHOLD, DEVICE, FINAL_MODEL_PATH, HEATMAPS_DIR
from config.settings import EXPECTED_CLASS_FOLDERS
from explainability.gradcam import GradCAMGenerator
from explainability.heatmap_generator import HeatmapGenerator
from explainability.visualization import export_explainability_visualization
from models.model_loader import MedicalClassifierViT
from preprocessing.preprocessing_pipeline import PreprocessingPipeline
from utils.file_utils import ensure_directory_exists
from utils.image_utils import load_image
from utils.logger import setup_logger, print_step, print_substep

logger = setup_logger(__name__)


class ClassifierService:
    """
    Service class that encapsulates the inference and XAI workflow.
    Loads the trained model exactly once upon initialisation.
    """

    def __init__(self) -> None:
        """Initialise the service by loading the model and Grad-CAM generator."""
        self.device = DEVICE
        self.class_names = EXPECTED_CLASS_FOLDERS
        self.num_classes = len(self.class_names)
        
        # 1. Load Preprocessing Pipeline
        self.pipeline = PreprocessingPipeline(mode="inference")
        
        # 2. Load Model
        self._load_model()
        
        # 3. Initialise Explainability tools
        self.grad_cam_generator = GradCAMGenerator(self.model)

    def _load_model(self) -> None:
        """Load the fine-tuned weights into MedicalClassifierViT."""
        if not FINAL_MODEL_PATH.exists():
            msg = f"Trained model not found at {FINAL_MODEL_PATH}. Run Agent 5 first."
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info(f"Loading trained classifier from {FINAL_MODEL_PATH}")
        self.model = MedicalClassifierViT(num_classes=self.num_classes)
        
        # Load state dict
        state_dict = torch.load(FINAL_MODEL_PATH, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        
        # Strictly set to evaluation mode
        self.model.eval()
            
        logger.info("Classifier successfully loaded and set to evaluation mode.")

    def predict(self, image_path: Path) -> Dict[str, Any]:
        """
        Run the complete inference and explainability pipeline on a single image.

        Args:
            image_path (Path): Path to the uploaded medical image.

        Returns:
            Dict[str, Any]: Structured prediction dictionary containing class,
                            confidence, status, and heatmap path.
        """
        start_time = time.time()
        
        # 1. Validate and Load Image
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Ensure it is a supported format based on suffix (case-insensitive)
        supported_formats = {".png", ".jpg", ".jpeg"}
        if image_path.suffix.lower() not in supported_formats:
            raise ValueError(f"Unsupported image format: {image_path.suffix}. Expected {supported_formats}")

        logger.info(f"Processing image for inference: {image_path.name}")
        original_image = load_image(image_path)
        
        if original_image is None:
            raise ValueError("Failed to decode the image. The file may be corrupted.")
            
        # Keep a copy of original image in RGB (0-1 float) for visualization
        rgb_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        rgb_original_norm = np.float32(rgb_original) / 255.0

        # 2. Preprocess (Reuse Agent 2 pipeline)
        print_step(2, "AI Vision Analysis", "Agent 6 is scanning the image for medical abnormalities...", color="\033[1;35m")
        # Returns shape (3, 224, 224)
        input_tensor = self.pipeline.process(original_image)
        logger.info(f"Image preprocessed to tensor shape: {input_tensor.shape}")
        print_substep("Converted image to numeric tensor (224x224 pixels) for the Neural Network.")
        
        # Add batch dimension -> (1, 3, 224, 224)
        batch_tensor = input_tensor.unsqueeze(0).to(self.device)
        logger.debug(f"Pushed tensor {batch_tensor.shape} to device: {self.device}")

        # 3. Model Inference (Forward pass)
        logger.info("Executing ViT Model forward pass...")
        with torch.no_grad():
            forward_start = time.time()
            logits = self.model(batch_tensor)
            forward_end = time.time()
            logger.info(f"Forward pass completed in {round((forward_end - forward_start) * 1000, 2)}ms")
            logger.debug(f"Raw logits output: {logits.cpu().numpy().tolist()}")
            
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            
        confidence, predicted_idx = torch.max(probabilities, dim=0)
        confidence_val = confidence.item()
        predicted_class = self.class_names[predicted_idx.item()]
        
        logger.info(f"Predicted: {predicted_class} with confidence {confidence_val:.4f}")
        print_substep(f"AI Prediction: '{predicted_class}' with {round(confidence_val*100, 2)}% confidence!")

        # 4. Status & Threshold evaluation
        status = "VALID"
        warning = None
        if confidence_val < CONFIDENCE_THRESHOLD:
            status = "LOW_CONFIDENCE"
            warning = (
                "The uploaded medical image does not closely match the disease patterns on "
                "which this model was trained, or the model confidence is below the accepted "
                "threshold. The prediction should not be considered a definitive diagnosis. "
                "Please consult a qualified medical professional."
            )
            logger.warning(f"Low confidence prediction ({confidence_val:.2f} < {CONFIDENCE_THRESHOLD})")

        # 5. Explainability (Grad-CAM)
        print_step(3, "Visual Explanation (Grad-CAM)", "Extracting 'attention' maps to show what the AI saw...", color="\033[1;36m")
        print_substep("Generating heatmap of active neural pathways...")
        heatmap_path_str = self._generate_explainability(
            batch_tensor, rgb_original_norm, predicted_class, confidence_val, image_path.stem
        )
        print_substep("Saved X-Ray visualization with colored heatmap overlay.")
        
        elapsed = time.time() - start_time

        # 6. Prepare structured response
        response = {
            "image_name": image_path.name,
            "predicted_class": predicted_class,
            "confidence_score": round(confidence_val, 4),
            "prediction_status": status,
            "warning_message": warning,
            "heatmap_path": heatmap_path_str,
            "model_name": "MedicalClassifierViT",
            "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
            "inference_time_seconds": round(elapsed, 3)
        }
        
        return response

    def _generate_explainability(
        self, batch_tensor: torch.Tensor, original_image: np.ndarray, 
        predicted_class: str, confidence: float, stem: str
    ) -> str:
        """Helper to generate Grad-CAM, overlay it, and save the full visualization."""
        try:
            logger.info("Initiating Explainability Pipeline (Heatmap Overlay)...")
            # Generate Raw Grad-CAM grayscale map
            grayscale_cam = self.grad_cam_generator.generate(batch_tensor)
            
            # Colourise and Overlay
            heatmap_rgb = HeatmapGenerator.apply_colormap(grayscale_cam)
            overlay = HeatmapGenerator.overlay_heatmap(heatmap_rgb, original_image)
            logger.info("Applied Jet colormap and alpha overlay to original image.")
            
            # Generate a "denormalized" preprocessed image view for debugging/visualization
            # Extract the actual model input back into an image
            input_array = batch_tensor[0].cpu().numpy()  # (3, 224, 224)
            input_array = np.transpose(input_array, (1, 2, 0)) # (224, 224, 3)
            # Revert ImageNet normalization just for display purposes
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            denorm_input = std * input_array + mean
            denorm_input = np.clip(denorm_input, 0, 1)

            # Unique filename
            unique_id = uuid.uuid4().hex[:8]
            output_filename = f"gradcam_{stem}_{unique_id}.png"
            output_path = HEATMAPS_DIR / output_filename
            
            # Generate the 4-panel graphical structure
            export_explainability_visualization(
                original_image=original_image,
                preprocessed_image=denorm_input,
                grayscale_cam=grayscale_cam,
                overlay_image=overlay,
                predicted_class=predicted_class,
                confidence=confidence,
                output_path=output_path
            )
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to generate explainability visualization: {e}")
            return ""
