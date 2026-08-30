"""
Threshold checking logic for Report Evaluation.
"""

from config.llm_config import QUALITY_THRESHOLD_PERCENT

class Threshold:
    """Validates if the percentage score meets the passing threshold."""
    
    @staticmethod
    def is_passed(percentage_score: float) -> bool:
        """
        Check if the score is greater than or equal to the configured threshold.
        
        Args:
            percentage_score (float): The calculated percentage score (0-100).
            
        Returns:
            bool: True if passing, False otherwise.
        """
        return percentage_score >= QUALITY_THRESHOLD_PERCENT
