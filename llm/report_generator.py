"""
Medical Report Generation Module.

Contains the System Prompt, Report Template, and logic to construct LangChain
messages (including multi-modal image inputs) for the LLM.
"""

import base64
from pathlib import Path
from typing import Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage

from config.llm_config import MODEL_NAME
from utils.logger import setup_logger

logger = setup_logger(__name__)

SYSTEM_PROMPT = """You are a highly skilled and professional Medical AI Assistant.
Your task is to generate a structured medical report based on the output of an AI Vision Transformer classifier and the provided medical image.

CRITICAL RULES:
1. You MUST NOT contradict the classifier. The classifier output (Predicted Class, Confidence, Status) is the source of truth. Your job is to EXPLAIN the prediction, not replace it.
2. You MUST NOT invent or hallucinate any patient history, age, gender, clinical symptoms, treatment plans, or laboratory results. If information is not provided, explicitly state "Information not available."
3. If the Prediction Status is "LOW_CONFIDENCE", you MUST include the provided warning message exactly as given in your report. Do NOT invent additional diagnoses.
4. Your tone must be professional, clear, concise, and medical.
5. You MUST return your final response ONLY as a valid JSON object matching the exact schema requested by the user, without any markdown formatting wrappers or conversational text outside the JSON.
"""

USER_TEMPLATE = """
Below is the structured output from the Medical Vision Classifier (Agent 6):

Predicted Class: {predicted_class}
Confidence Score: {confidence_score}
Prediction Status: {prediction_status}
Warning Message: {warning_message}
Model Used: {model_name}

Attached is the medical image and/or Grad-CAM heatmap visualization.

Please analyze the provided data and generate a JSON report with the following exact keys:
{{
    "patient_information": "(Use 'Information not available' if unknown)",
    "image_summary": "(Detailed pointwise description of what the image shows)",
    "classification": "(predicted_class)",
    "confidence": "(confidence_score)",
    "gradcam_observation": "(Detailed pointwise description of the highlighted regions from the Grad-CAM heatmap)",
    "clinical_interpretation": "(Detailed pointwise interpretation of what the AI prediction and heatmap signify clinically)",
    "recommendation": "(Clear, pointwise actionable next steps based on the findings)",
    "disclaimer": "(Standard AI disclaimer)",
    "generated_report": "(A fully assembled, human-readable, detailed pointwise clinical report combining all above fields)"
}}
"""


class ReportGenerator:
    """
    Constructs multi-modal LangChain messages for Medical Report Generation.
    """

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Encodes an image to a base64 string for LLM consumption."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image at {image_path}: {e}")
            return ""

    @staticmethod
    def construct_messages(prediction_data: Dict[str, Any], custom_prompt: str = None) -> List:
        """
        Builds the LangChain message payload including the system prompt, textual data,
        and multi-modal image inputs (if vision is supported).
        """
        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        # Prepare formatting data
        warning_msg = prediction_data.get("warning_message")
        if not warning_msg:
            warning_msg = "None. Confidence is high."

        user_text = custom_prompt if custom_prompt else USER_TEMPLATE.format(
            predicted_class=prediction_data.get("predicted_class", "Unknown"),
            confidence_score=prediction_data.get("confidence_score", "Unknown"),
            prediction_status=prediction_data.get("prediction_status", "Unknown"),
            warning_message=warning_msg,
            model_name=prediction_data.get("model_name", "Unknown")
        )

        heatmap_path = prediction_data.get("heatmap_path")
        if heatmap_path and Path(heatmap_path).exists() and "vision" in MODEL_NAME.lower():
            base64_image = ReportGenerator._encode_image(heatmap_path)
            if base64_image:
                human_content = [{"type": "text", "text": user_text}]
                human_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
                logger.debug("Attached Grad-CAM heatmap to LLM prompt.")
                messages.append(HumanMessage(content=human_content))
                return messages

        # If no vision is supported, just pass string content directly
        messages.append(HumanMessage(content=user_text))

        return messages
