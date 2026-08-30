"""
Prompt Refiner module.
"""

from typing import Dict, Any, Optional

class PromptRefiner:
    """Extracts the refined prompt from the unified judge payload."""
    
    @staticmethod
    def extract_refined_prompt(parsed_judge_output: Dict[str, Any]) -> Optional[str]:
        """
        Retrieves the refined prompt string.
        """
        return parsed_judge_output.get("refined_prompt")
