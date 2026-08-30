"""
Agent 6 – Medical Image Classification & Grad-CAM (Pipeline B).

Acts as the primary Inference Agent. Accepts a raw user-uploaded medical image,
uses the ClassifierService to predict the class, generates Explainable AI (Grad-CAM)
visualizations, and outputs a structured prediction report for Agent 7.

Usage (from the project root)::

    python -m agents.agent6_classifier [path_to_image]

If no image is provided, it will attempt to pick a random test image from
the datasets directory to demonstrate the pipeline.
"""

import argparse
import json
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.services.classifier_service import ClassifierService
from config.model_config import PREDICTIONS_DIR
from config.settings import CLEANED_DATASET_DIR
from utils.file_utils import ensure_directory_exists, get_all_files_recursive
from utils.logger import setup_logger

logger = setup_logger(__name__, log_file="agent6_classifier.log")


class MedicalClassificationAgent:
    """
    Agent 6: Runs inference on user-uploaded medical images.
    """

    def __init__(self) -> None:
        """Initialise the Agent and load the backend Classifier Service."""
        logger.info("Initializing Agent 6: Loading Medical Classifier Service...")
        self.service = ClassifierService()

    def execute(self, image_path: Path) -> dict:
        """
        Execute the classification pipeline on a given image.

        Args:
            image_path (Path): Path to the medical image.

        Returns:
            dict: The structured prediction JSON object.
        """
        logger.info("=" * 60)
        logger.info("  MEDICAL IMAGE CLASSIFICATION AGENT (Agent 6) – Starting")
        logger.info("=" * 60)
        
        try:
            # The service handles validation, preprocessing, inference, and Grad-CAM
            response = self.service.predict(image_path)
            
            # Save the prediction report
            self._save_prediction_report(response)
            
            # Log summary
            self._log_summary(response)
            
            logger.info("=" * 60)
            logger.info("  MEDICAL IMAGE CLASSIFICATION AGENT – Completed Successfully")
            logger.info("=" * 60)
            
            return response
            
        except Exception as e:
            logger.error(f"Inference pipeline failed: {e}")
            raise

    def _save_prediction_report(self, response: dict) -> None:
        """Save the structured response to a JSON file."""
        ensure_directory_exists(PREDICTIONS_DIR)
        
        report_path = PREDICTIONS_DIR / "prediction_report.json"
        
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(response, f, indent=4, ensure_ascii=False)
            logger.info(f"Structured prediction report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Failed to save prediction report: {e}")
            raise

    def _log_summary(self, response: dict) -> None:
        """Print a human-readable summary of the prediction."""
        logger.info("-" * 50)
        logger.info("  INFERENCE SUMMARY")
        logger.info("-" * 50)
        logger.info(f"  Image           : {response['image_name']}")
        logger.info(f"  Predicted Class : {response['predicted_class']}")
        logger.info(f"  Confidence      : {response['confidence_score']:.2%}")
        logger.info(f"  Status          : {response['prediction_status']}")
        if response['prediction_status'] == "LOW_CONFIDENCE":
            logger.warning(f"  Warning         : {response['warning_message']}")
        logger.info(f"  Grad-CAM Heatmap: {response['heatmap_path']}")
        logger.info(f"  Inference Time  : {response['inference_time_seconds']}s")
        logger.info("-" * 50)


# =========================================================================
#  Entry point
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent 6 - Medical Image Inference")
    parser.add_argument("image", nargs="?", type=str, help="Path to the medical image to classify.")
    args = parser.parse_args()
    
    image_path = None
    if args.image:
        image_path = Path(args.image)
    else:
        # If no image is provided, pick a random image from the cleaned dataset for demonstration.
        logger.info("No image provided. Selecting a random image from the cleaned dataset for demonstration...")
        if CLEANED_DATASET_DIR.exists():
            all_images = get_all_files_recursive(CLEANED_DATASET_DIR)
            if all_images:
                image_path = random.choice(all_images)
            
    if not image_path:
        logger.error("No valid image found to process.")
        sys.exit(1)
        
    logger.info(f"Target Image: {image_path}")
    
    # Run the Agent
    agent = MedicalClassificationAgent()
    result = agent.execute(image_path)
