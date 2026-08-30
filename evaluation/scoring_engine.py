"""
Scoring Engine for evaluating medical reports.
"""

from typing import Dict, Any, List

class ScoringEngine:
    """Computes total and percentage scores based on LLM judge output."""
    
    # 0 = Incorrect, 1 = Partially Correct, 2 = Fully Correct
    MAX_SCORE_PER_QUESTION = 2
    
    @classmethod
    def compute_scores(cls, criterion_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate total and percentage score.
        
        Args:
            criterion_scores: List of dicts, each with an integer 'score' key.
            
        Returns:
            Dict containing total_score and percentage_score.
        """
        total_possible = len(criterion_scores) * cls.MAX_SCORE_PER_QUESTION
        
        total_score = sum(item.get("score", 0) for item in criterion_scores)
        
        if total_possible == 0:
            percentage = 0.0
        else:
            percentage = round((total_score / total_possible) * 100.0, 2)
            
        return {
            "total_score": total_score,
            "total_possible": total_possible,
            "percentage_score": percentage
        }
