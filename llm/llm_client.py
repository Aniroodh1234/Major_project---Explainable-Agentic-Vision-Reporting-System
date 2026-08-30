"""
LangChain Groq Client Module.

Initializes the LLM connection using langchain-groq and handles API errors,
timeouts, and retries.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential

from config.llm_config import GROQ_API_KEY, MAX_TOKENS, MODEL_NAME, TEMPERATURE, TIMEOUT
from utils.logger import setup_logger

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
            response = self.llm.invoke(messages)
            
            # Optionally log token usage if available in response_metadata
            token_usage = response.response_metadata.get("token_usage", {})
            if token_usage:
                logger.info(f"LLM Token Usage: {token_usage}")
                
            return response.content
        except Exception as e:
            logger.warning(f"LLM invocation failed: {e}. Retrying...")
            raise
