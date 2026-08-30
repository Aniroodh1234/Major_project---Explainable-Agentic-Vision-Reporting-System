"""
High-level Validator orchestrating Scoring and Threshold checks.
"""

from typing import List, Dict, Any, Tuple

from evaluation.scoring_engine import ScoringEngine
from evaluation.threshold import Threshold

class ReportValidator:
    """Combines scoring and threshold validation."""
    
    @staticmethod
    def validate(criterion_scores: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        """
        Calculates scores and determines if the threshold is met.
        
        Args:
            criterion_scores: Raw scores from the LLM Judge.
            
        Returns:
            Tuple[bool, Dict]: Boolean indicating if passed, and the score dictionary.
        """
        score_data = ScoringEngine.compute_scores(criterion_scores)
        percentage = score_data["percentage_score"]
        
        is_valid = Threshold.is_passed(percentage)
        
        return is_valid, score_data
