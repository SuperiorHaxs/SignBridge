#!/usr/bin/env python3
"""
Groq Provider - Fast LLM inference using Groq API

Groq provides extremely fast inference for various open-source models.
Uses OpenAI-compatible API endpoint.
"""

from typing import Optional, List, Dict, Any
import time
from openai import OpenAI

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_interface import (
    LLMInterface,
    LLMAPIError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMInvalidResponseError
)


class GroqProvider(LLMInterface):
    """
    Groq LLM provider using OpenAI-compatible API.

    Groq provides fast inference for models like:
    - llama-3.3-70b-versatile
    - llama-3.1-70b-versatile
    - mixtral-8x7b-32768
    """

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize Groq provider.

        Args:
            model_name: Groq model identifier
            api_key: Groq API key
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for transient failures
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries
        )

        if not api_key:
            raise LLMAPIError(
                "Groq API key is required. "
                "Get your free API key at: https://console.groq.com/keys"
            )

        # Initialize OpenAI client with Groq endpoint
        self.client = OpenAI(
            base_url=self.BASE_URL,
            api_key=self.api_key,
            timeout=self.timeout
        )

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text using Groq API.

        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional arguments passed to API

        Returns:
            Generated text string

        Raises:
            LLMAPIError: API request failed
            LLMTimeoutError: Request timed out
            LLMRateLimitError: Rate limit exceeded
            LLMInvalidResponseError: Invalid response from API
        """
        use_temperature = temperature if temperature is not None else self.temperature
        use_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=use_temperature,
                    max_tokens=use_max_tokens,
                    **kwargs
                )

                # Extract response
                if not completion.choices:
                    raise LLMInvalidResponseError("No choices in response")

                response = completion.choices[0].message.content

                if response is None:
                    raise LLMInvalidResponseError("Empty response from API")

                return response.strip()

            except Exception as e:
                error_msg = str(e).lower()

                # Handle rate limiting
                if "rate" in error_msg or "429" in error_msg:
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        print(f"  Rate limit hit. Waiting {wait_time}s... (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(wait_time)
                        continue
                    raise LLMRateLimitError(f"Rate limit exceeded: {e}")

                # Handle timeout
                if "timeout" in error_msg or "timed out" in error_msg:
                    if attempt < self.max_retries - 1:
                        print(f"  Request timeout. Retrying... (attempt {attempt + 1}/{self.max_retries})")
                        continue
                    raise LLMTimeoutError(f"Request timed out after {self.timeout}s: {e}")

                # Handle model loading (503)
                if "503" in error_msg or "loading" in error_msg:
                    if attempt < self.max_retries - 1:
                        wait_time = 5
                        print(f"  Model loading. Waiting {wait_time}s... (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(wait_time)
                        continue
                    raise LLMAPIError(f"Model loading timeout: {e}")

                # Other API errors
                raise LLMAPIError(f"API error: {e}")

        raise LLMAPIError(f"Max retries ({self.max_retries}) exceeded")

    def generate_batch(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for multiple prompts.

        Groq API doesn't have native batch support, so we call sequentially.

        Args:
            prompts: List of input prompts
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional arguments passed to API

        Returns:
            List of generated text strings
        """
        return [
            self.generate(prompt, temperature, max_tokens, **kwargs)
            for prompt in prompts
        ]

    def is_available(self) -> bool:
        """
        Check if Groq API is available.

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Try a minimal request
            self.generate("test", max_tokens=5)
            return True
        except Exception:
            return False
