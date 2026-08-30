"""
Medical Report Service.

Orchestrates the LLMClient, ReportGenerator, and ResponseFormatter to generate
a structured medical report from the Agent 6 prediction output.
"""

import time
from typing import Dict, Any

from llm.llm_client import LLMClient
from llm.report_generator import ReportGenerator
from llm.response_formatter import ResponseFormatter
from utils.logger import setup_logger, print_step, print_substep

logger = setup_logger(__name__)


class ReportService:
    """
    Service layer for generating AI-assisted medical reports.
    """

    def __init__(self) -> None:
        """Initialise the ReportService with required LLM components."""
        self.llm_client = LLMClient()

    def generate_report(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinates the multi-step process of generating a report.

        Args:
            prediction_data (Dict[str, Any]): The structured output from Agent 6.

        Returns:
            Dict[str, Any]: A merged dictionary containing the original prediction
                            data and the newly generated medical report.
        """
        start_time = time.time()
        logger.info(f"Generating medical report for image: {prediction_data.get('image_name')}")

        # 1. Construct LangChain Messages (Text + Vision)
        messages = ReportGenerator.construct_messages(prediction_data)

        # 2. Invoke LLM
        print_step(4, "Medical Report Generation", "Agent 7 is analyzing the vision output to draft a clinical report...", color="\033[1;35m")
        raw_response = self.llm_client.invoke(messages)
        print_substep("Report drafted successfully using OpenAI/GPT architecture.")

        # 3. Format and Validate the JSON response
        parsed_report = ResponseFormatter.parse_json_report(raw_response)

        # 4. Agent 8: Evaluate and Refine
        from llm.evaluation_engine import EvaluationEngine
        print_step(5, "AI Quality Audit (Agent 8)", "Double-checking the report for medical accuracy, hallucinations, and formatting...", color="\033[1;36m")
        logger.info("Handing off to Evaluation Engine (Agent 8) for quality scoring and refinement...")
        eval_engine = EvaluationEngine()
        validated_report, metrics = eval_engine.evaluate_and_refine(prediction_data, parsed_report)

        # 5. Compile final output object as per Prompt 8 requirements
        generation_time = round(time.time() - start_time, 3)
        logger.info(f"Report generation and evaluation completed in {generation_time}s.")

        # Save evaluation report metadata to disk
        from config.settings import PROJECT_ROOT
        import json
        eval_report_path = PROJECT_ROOT / "outputs" / "reports" / f"evaluation_report_{prediction_data.get('image_name', 'unknown')}.json"
        with open(eval_report_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=4)

        # The Prompt dictates we return:
        # Structured Prediction Object + Validated Medical Report + Evaluation Score + Validation Status
        final_payload = prediction_data.copy()
        
        # Merge validated report fields
        for key, value in validated_report.items():
            final_payload[key] = value
            
        # Add evaluation scoring data for transparency
        final_payload["evaluation_score"] = metrics.percentage_score
        final_payload["validation_status"] = metrics.validation_status
        final_payload["evaluation_feedback"] = metrics.feedback
        final_payload["total_refinement_iterations"] = metrics.iteration_number

        return final_payload
