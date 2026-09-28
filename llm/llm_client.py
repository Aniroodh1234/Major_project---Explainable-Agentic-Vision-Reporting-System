from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential

import time
from config.llm_config import GROQ_API_KEY, MAX_TOKENS, MODEL_NAME, TEMPERATURE, TIMEOUT
from utils.logger import setup_logger
from observability.metrics_registry import (
    LLM_INVOCATIONS_TOTAL,
    LLM_DURATION_SECONDS,
    LLM_TOKENS_TOTAL,
    LLM_ERRORS_TOTAL,
)

logger = setup_logger(__name__)


class LLMClient:
    """
    Client for interacting with the LLM via LangChain.
    """

    def __init__(self) -> None:
        """Initialise the ChatGroq model with configurations."""
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing. Check your .env file.")
            
        try:
            self.llm: BaseChatModel = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name=MODEL_NAME,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                timeout=TIMEOUT,
                max_retries=2
            )
            logger.info(f"LLMClient initialised successfully with model: {MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize ChatGroq: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def invoke(self, messages: list) -> str:
        """
        Send a list of messages to the LLM and return the response content.
        Features automatic retries with exponential backoff for network issues/rate limits.

        Args:
            messages (list): A list of LangChain message objects (e.g., SystemMessage, HumanMessage).

        Returns:
            str: The textual response from the LLM.
        """
        try:
            logger.debug(f"Invoking LLM ({MODEL_NAME}) with {len(messages)} messages.")
            _llm_start = time.time()
            response = self.llm.invoke(messages)

            _llm_elapsed = time.time() - _llm_start
            LLM_INVOCATIONS_TOTAL.labels(purpose="llm_call").inc()
            LLM_DURATION_SECONDS.labels(purpose="llm_call").observe(_llm_elapsed)

            # Optionally log token usage if available in response_metadata
            token_usage = response.response_metadata.get("token_usage", {})
            if token_usage:
                logger.info(f"LLM Token Usage: {token_usage}")
                if "prompt_tokens" in token_usage:
                    LLM_TOKENS_TOTAL.labels(type="prompt").inc(token_usage["prompt_tokens"])
                if "completion_tokens" in token_usage:
                    LLM_TOKENS_TOTAL.labels(type="completion").inc(token_usage["completion_tokens"])
                if "total_tokens" in token_usage:
                    LLM_TOKENS_TOTAL.labels(type="total").inc(token_usage["total_tokens"])

            return response.content
        except Exception as e:
            LLM_ERRORS_TOTAL.inc()
            logger.warning(f"LLM invocation failed: {e}. Retrying...")
            raise
