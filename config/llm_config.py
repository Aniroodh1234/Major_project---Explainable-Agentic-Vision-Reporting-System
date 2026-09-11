import os
from pathlib import Path

from dotenv import load_dotenv

from config.settings import PROJECT_ROOT

# Load environment variables from .env file
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


# We are using Groq API as per user instruction.
LLM_PROVIDER: str = "groq"
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# The model to use on Groq
# We use a Vision-capable model because Agent 7 must analyze the medical image and Grad-CAM heatmap.
MODEL_NAME: str = "openai/gpt-oss-120b"


TEMPERATURE: float = 0.0  # Zero temperature for deterministic medical reports
MAX_TOKENS: int = 2048
TIMEOUT: int = 60

# Evaluation/Judge thresholds
QUALITY_THRESHOLD_PERCENT: float = 85.0
MAX_REFINEMENT_ITERATIONS: int = 2  # Set low to guarantee fast latency (4-5s max)


LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "Agentic AI Medical Imaging")

if not GROQ_API_KEY:
    import logging
    logging.getLogger(__name__).warning("GROQ_API_KEY is not set in the environment.")
