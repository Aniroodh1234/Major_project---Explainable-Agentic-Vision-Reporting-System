"""
Single-pass LLM Judge, Feedback, and Prompt Refiner.
"""

import json
from typing import Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage

from llm.llm_client import LLMClient
from evaluation.evaluation_questions import EVALUATION_CRITERIA
from utils.logger import setup_logger

logger = setup_logger(__name__)


JUDGE_SYSTEM_PROMPT = """You are an expert Medical AI Auditor and Prompt Engineer.
Your job is to evaluate a generated medical report against a strict set of criteria, score it, provide actionable feedback, and instantly refine the generation prompt to fix any errors.

You will receive:
1. The structured prediction object (Classifier output).
2. The generated Medical Report.
3. The evaluation criteria checklist.

CRITICAL RULES:
1. You MUST output ONLY valid JSON matching the exact schema below. No markdown wrappers.
2. Evaluate each criterion honestly. 0 = Incorrect/Missing, 1 = Partially Correct, 2 = Fully Correct.
3. If ANY score is 0 or 1, you MUST provide 'feedback' and a 'refined_prompt'.
4. If ALL scores are 2, 'feedback' should be "All criteria met." and 'refined_prompt' should be null.

EXPECTED JSON SCHEMA:
{
    "criterion_scores": [
        {"id": "q1", "score": 2},
        {"id": "q2", "score": 0},
        ...
    ],
    "feedback": "...",
    "refined_prompt": "..."
}
"""

class ReportJudge:
    """Executes the single-pass evaluation and refinement."""
    
    def __init__(self):
        self.llm_client = LLMClient()
        
    def evaluate(self, prediction_data: Dict[str, Any], generated_report: Dict[str, Any], original_prompt: str) -> str:
        """
        Sends the payload to the LLM to get scores, feedback, and refined prompt.
        """
        criteria_str = json.dumps(EVALUATION_CRITERIA, indent=2)
        pred_str = json.dumps(prediction_data, indent=2)
        report_str = json.dumps(generated_report, indent=2)
        
        user_content = (
            f"--- EVALUATION CRITERIA ---\n{criteria_str}\n\n"
            f"--- CLASSIFIER PREDICTION ---\n{pred_str}\n\n"
            f"--- GENERATED REPORT ---\n{report_str}\n\n"
            f"--- ORIGINAL PROMPT (to refine if needed) ---\n{original_prompt}\n"
        )
        
        messages = [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=user_content)
        ]
        
        logger.info("Sending report to LLM Judge for evaluation...")
        raw_response = self.llm_client.invoke(messages)
        return raw_response
