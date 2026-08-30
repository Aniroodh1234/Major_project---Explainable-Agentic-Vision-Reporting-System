"""
Feedback extraction module.
"""

from typing import Dict, Any, Optional

class FeedbackGenerator:
    """Extracts feedback from the unified judge payload."""
    
    @staticmethod
    def extract_feedback(parsed_judge_output: Dict[str, Any]) -> Optional[str]:
        """
        Retrieves the feedback string.
        """
        return parsed_judge_output.get("feedback")
