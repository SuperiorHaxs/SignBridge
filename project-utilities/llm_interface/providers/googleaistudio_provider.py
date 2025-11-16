#!/usr/bin/env python3
"""
Google AI Studio Provider - Access to Gemini models via Google AI Studio API

Provides access to Google's Gemini models through AI Studio API.
"""

from typing import Optional, List, Dict, Any
import time

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

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


class GoogleAIStudioProvider(LLMInterface):
    """
    Google AI Studio LLM provider for Gemini models.

    Supports models like:
    - gemini-2.0-flash-exp (latest experimental)
    - gemini-2.0-flash (stable)
    - gemini-2.5-flash (preview)
    - gemini-2.5-pro (preview)

    Note: The provider automatically adds the 'models/' prefix.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-exp",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize Google AI Studio provider.

        Args:
            model_name: Gemini model identifier
            api_key: Google AI Studio API key
            temperature: Sampling temperature (0.0-2.0 for Gemini)
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

        if not GEMINI_AVAILABLE:
            raise LLMAPIError(
                "google-generativeai package not installed.\n"
                "Install with: pip install google-generativeai"
            )

        if not api_key:
            raise LLMAPIError(
                "Google AI Studio API key is required. "
                "Get your free API key at: https://aistudio.google.com/app/apikey"
            )

        # Configure Gemini
        genai.configure(api_key=self.api_key)

        # Add 'models/' prefix if not present
        if not self.model_name.startswith('models/'):
            full_model_name = f'models/{self.model_name}'
        else:
            full_model_name = self.model_name

        # Initialize model with safety settings
        self.model = genai.GenerativeModel(
            model_name=full_model_name,
            safety_settings={
                'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
            }
        )

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text using Gemini API.

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
                # Configure generation parameters
                generation_config = genai.GenerationConfig(
                    temperature=use_temperature,
                    max_output_tokens=use_max_tokens,
                    **kwargs
                )

                # Generate response
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    request_options={'timeout': self.timeout}
                )

                # Extract text from response
                if not response.text:
                    # Check if blocked by safety filters
                    if hasattr(response, 'prompt_feedback'):
                        raise LLMInvalidResponseError(
                            f"Response blocked by safety filters: {response.prompt_feedback}"
                        )
                    raise LLMInvalidResponseError("Empty response from API")

                return response.text.strip()

            except Exception as e:
                error_msg = str(e).lower()

                # Handle rate limiting
                if "quota" in error_msg or "rate" in error_msg or "429" in error_msg:
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        print(f"  Rate limit hit. Waiting {wait_time}s... (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(wait_time)
                        continue
                    raise LLMRateLimitError(f"Rate limit exceeded: {e}")

                # Handle timeout
                if "timeout" in error_msg or "timed out" in error_msg or "deadline" in error_msg:
                    if attempt < self.max_retries - 1:
                        print(f"  Request timeout. Retrying... (attempt {attempt + 1}/{self.max_retries})")
                        continue
                    raise LLMTimeoutError(f"Request timed out after {self.timeout}s: {e}")

                # Handle service unavailable
                if "503" in error_msg or "unavailable" in error_msg:
                    if attempt < self.max_retries - 1:
                        wait_time = 5
                        print(f"  Service unavailable. Waiting {wait_time}s... (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(wait_time)
                        continue
                    raise LLMAPIError(f"Service unavailable: {e}")

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

        Gemini API doesn't have native batch support, so we call sequentially.

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
        Check if Google AI Studio API is available.

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Try a minimal request
            self.generate("test", max_tokens=5)
            return True
        except Exception:
            return False
