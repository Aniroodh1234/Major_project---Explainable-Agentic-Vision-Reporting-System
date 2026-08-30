"""
Data schemas for evaluation metrics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class EvaluationMetrics:
    """Holds all metadata and scoring for a single evaluation iteration."""
    iteration_number: int
    criterion_scores: List[Dict[str, Any]]
    total_score: int
    percentage_score: float
    feedback: Optional[str]
    validation_status: str
    execution_time_seconds: float
    refined_prompt: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialisation."""
        return {
            "iteration_number": self.iteration_number,
            "criterion_scores": self.criterion_scores,
            "total_score": self.total_score,
            "percentage_score": self.percentage_score,
            "feedback": self.feedback,
            "validation_status": self.validation_status,
            "execution_time_seconds": self.execution_time_seconds
        }
