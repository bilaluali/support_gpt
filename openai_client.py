import logging
import os
import time
from typing import List, Optional

from fastapi import HTTPException
from openai import OpenAI
from openai import RateLimitError as OpenAIRateLimitError
from openai.types.chat import ChatCompletionMessageParam as OpenAIMessage

logger = logging.getLogger(__name__)


DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2  # Lower temperature for more deterministic responses
DEFAULT_MAX_TOKENS = 150


class OpenAIClient:
    """
    OpenAI client that handles rate limiting, retries, and error handling
    for both moderation and chat completion API calls.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay: float = DEFAULT_BASE_DELAY,
    ):
        """Initialize the OpenAI client with retry configuration."""
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.max_retries = max_retries
        self.base_delay = base_delay


    def _handle_rate_limit_error(self, attempt: int) -> None:
        """Handle rate limit errors with exponential backoff"""
        if attempt >= self.max_retries:
            logger.error(f"Rate limit exceeded after {self.max_retries} retries")
            raise HTTPException(429, "Rate limit exceeded. Please try again later.")

        # Exponential backoff: delay = base_delay * (2 ^ attempt)
        delay = self.base_delay * (2**attempt)
        logger.warning(
            f"Rate limit hit, retrying in {delay:.2f} seconds (attempt {attempt + 1}/{self.max_retries})"
        )
        time.sleep(delay)

    def is_offensive_content(self, text: str) -> bool:
        """Check if the text contains offensive content using OpenAI's moderation API"""
        for attempt in range(self.max_retries + 1):
            try:
                moderation = self.client.moderations.create(input=text)
                return moderation.results[0].flagged

            except OpenAIRateLimitError:
                self._handle_rate_limit_error(attempt)

            except Exception as e:
                logger.error(f"Failed to moderate content: {str(e)}")
                raise HTTPException(500, f"Failed to moderate conversation: {str(e)}")

        # Defensive programming: This should never be reached due to the exception in _handle_rate_limit_error
        raise HTTPException(429, "Rate limit exceeded. Please try again later.")

    def create_chat_completion(
        self,
        messages: List[OpenAIMessage],
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Create a chat completion using OpenAI's chat completions API"""
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content

            except OpenAIRateLimitError:
                self._handle_rate_limit_error(attempt)

            except Exception as e:
                logger.error(f"Failed to create chat completion: {str(e)}")
                raise HTTPException(500, f"Failed to generate response: {str(e)}")

        # Defensive programming: This should never be reached due to the exception in _handle_rate_limit_error
        raise HTTPException(429, "Rate limit exceeded. Please try again later.")
