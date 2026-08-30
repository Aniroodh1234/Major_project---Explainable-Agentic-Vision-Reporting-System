"""
Parser for the LLM Judge JSON output.
"""

import json
from typing import Dict, Any

from utils.logger import setup_logger

logger = setup_logger(__name__)


class OutputParser:
    """Parses and validates JSON responses from the judge."""
    
    @staticmethod
    def parse_judge_response(raw_response: str) -> Dict[str, Any]:
        """
        Extracts JSON from the LLM output.
        
        Args:
            raw_response (str): The raw string from the LLM.
            
        Returns:
            Dict containing criterion_scores, feedback, and refined_prompt.
        """
        try:
            # Attempt to strip any markdown formatting
            cleaned = raw_response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
                
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
                
            parsed = json.loads(cleaned.strip())
            
            # Validation
            if "criterion_scores" not in parsed:
                raise ValueError("Missing 'criterion_scores' in judge output.")
                
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Judge JSON: {e}\nRaw output: {raw_response}")
            # Fallback safe payload
            return {
                "criterion_scores": [],
                "feedback": "Failed to parse judge output.",
                "refined_prompt": None
            }
        except Exception as e:
            logger.error(f"Unexpected error parsing Judge output: {e}")
            return {
                "criterion_scores": [],
                "feedback": "Unexpected error parsing output.",
                "refined_prompt": None
            }
