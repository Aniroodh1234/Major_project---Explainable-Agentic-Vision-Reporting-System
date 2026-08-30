"""
Evaluation Engine (Agent 8 Core).
Orchestrates the LLM Judge, Scoring, Feedback, and Prompt Refinement loops.
"""

import time
from typing import Dict, Any, Tuple

from config.llm_config import MAX_REFINEMENT_ITERATIONS
from evaluation.report_validator import ReportValidator
from evaluation.metrics import EvaluationMetrics
from llm.judge import ReportJudge
from llm.output_parser import OutputParser
from llm.feedback import FeedbackGenerator
from llm.prompt_refiner import PromptRefiner
from llm.report_generator import ReportGenerator, USER_TEMPLATE
from llm.llm_client import LLMClient
from llm.response_formatter import ResponseFormatter
from utils.logger import setup_logger, print_step, print_substep

logger = setup_logger(__name__)


class EvaluationEngine:
    """Manages the evaluation and refinement loop for medical reports."""
    
    def __init__(self):
        self.judge = ReportJudge()
        self.llm_client = LLMClient()
        
    def evaluate_and_refine(self, prediction_data: Dict[str, Any], initial_report: Dict[str, Any]) -> Tuple[Dict[str, Any], EvaluationMetrics]:
        """
        Executes the LLM-as-a-judge loop.
        
        Args:
            prediction_data: Output from Agent 6.
            initial_report: The first pass report from Agent 7.
            
        Returns:
            Tuple of (Final Validated Report Dict, Evaluation Metrics)
        """
        current_report = initial_report
        
        # We start with the base prompt template used by Agent 7
        current_prompt = USER_TEMPLATE.format(
            predicted_class=prediction_data.get("predicted_class", "Unknown"),
            confidence_score=prediction_data.get("confidence_score", "Unknown"),
            prediction_status=prediction_data.get("prediction_status", "Unknown"),
            warning_message=prediction_data.get("warning_message", ""),
            model_name=prediction_data.get("model_name", "Unknown")
        )
        
        iteration = 1
        max_iters = max(1, MAX_REFINEMENT_ITERATIONS)
        
        metrics = None
        start_time = time.time()
        
        while iteration <= max_iters:
            logger.info(f"Evaluation Iteration {iteration}/{max_iters}")
            
            # 1. Judge the current report
            raw_judge_response = self.judge.evaluate(prediction_data, current_report, current_prompt)
            parsed_judge = OutputParser.parse_judge_response(raw_judge_response)
            
            # 2. Score and Validate
            criterion_scores = parsed_judge.get("criterion_scores", [])
            is_valid, score_data = ReportValidator.validate(criterion_scores)
            
            # Log detailed criterion breakdown
            print_substep(f"Audit Complete! Criteria Passed: {score_data['total_score']}/{score_data['total_possible']} ({score_data['percentage_score']}%)")
            for item in criterion_scores:
                status_icon = "✅" if item.get('score') == 2 else ("⚠️" if item.get('score') == 1 else "❌")
                logger.info(f"Criterion {item.get('id')}: Score {item.get('score')}/2")
                print_substep(f"{status_icon} {item.get('id')}: Score {item.get('score')}/2")
            
            # 3. Extract feedback and refined prompt
            feedback = FeedbackGenerator.extract_feedback(parsed_judge)
            refined_prompt = PromptRefiner.extract_refined_prompt(parsed_judge)
            
            status = "VALIDATED" if is_valid else "PARTIALLY_VALIDATED"
            
            # Save metrics for this iteration
            metrics = EvaluationMetrics(
                iteration_number=iteration,
                criterion_scores=criterion_scores,
                total_score=score_data["total_score"],
                percentage_score=score_data["percentage_score"],
                feedback=feedback,
                validation_status=status,
                execution_time_seconds=round(time.time() - start_time, 2),
                refined_prompt=refined_prompt
            )
            
            logger.info(f"Iteration {iteration} Total Score: {score_data['total_score']}/{score_data['total_possible']} ({score_data['percentage_score']}%)")
            logger.info(f"Iteration {iteration} Status: {status}")
            if feedback and not is_valid:
                logger.warning(f"Judge Feedback: {feedback}")
                print_substep(f"Issue Found: {feedback}")
            
            if is_valid:
                logger.info("Report passed quality threshold. Stopping evaluation.")
                print_substep(f"Report Validation Status: {status} (Passed Quality Threshold!)")
                break
                
            if iteration == max_iters:
                logger.warning("Max iterations reached. Returning highest scoring report as PARTIALLY_VALIDATED.")
                print_substep(f"Max attempts reached. Validated as much as possible.")
                break
                
            # 4. Refine & Regenerate (if not valid and not max iter)
            logger.info("Report failed threshold. Regenerating with refined prompt...")
            print_step(6, "Self-Correction & Refinement", "Agent 8 is rewriting the report to fix the identified issues...", color="\033[1;33m")
            if not refined_prompt:
                logger.warning("No refined prompt provided by judge. Using base prompt.")
                refined_prompt = current_prompt
                
            current_prompt = refined_prompt
            
            # Construct new messages and invoke generator LLM
            messages = ReportGenerator.construct_messages(prediction_data, custom_prompt=current_prompt)
            raw_gen_response = self.llm_client.invoke(messages)
            current_report = ResponseFormatter.parse_json_report(raw_gen_response)
            
            iteration += 1
            
        return current_report, metrics
