#!/usr/bin/env python3
"""
HuggingFace Provider (OpenAI-compatible API)

Uses HuggingFace's OpenAI-compatible API for text generation.
Documentation: https://huggingface.co/docs/api-inference/
"""

import time
from typing import Optional, List, Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_interface import (
    LLMInterface,
    LLMError,
    LLMAPIError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMInvalidResponseError
)

try:
    from openai import OpenAI
    from openai import APIError, APITimeoutError, RateLimitError, AuthenticationError
except ImportError:
    raise ImportError(
        "OpenAI package is required for HuggingFace provider.\n"
        "Install with: pip install openai"
    )


class HuggingFaceProvider(LLMInterface):
    """
    HuggingFace provider using OpenAI-compatible API.

    Supports any text-generation model available on HuggingFace Hub.

    Recommended models:
    - meta-llama/Llama-3.3-70B-Instruct (best quality)
    - meta-llama/Llama-3.1-8B-Instruct (fast, good quality)
    - mistralai/Mistral-7B-Instruct-v0.3 (fast)

    Note: Provider suffix (e.g., :novita) is optional. Leave it off for default provider.
    """

    BASE_URL = "https://router.huggingface.co/v1"

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: int = 30,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize HuggingFace provider.

        Args:
            model_name: HuggingFace model identifier
                       (e.g., "meta-llama/Llama-3.1-8B-Instruct")
                       Provider suffix (e.g., :novita) is optional
            api_key: HuggingFace API token (environment: HF_TOKEN or HUGGINGFACE_API_KEY)
                    Get from https://huggingface.co/settings/tokens
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Additional parameters (top_p, etc.)
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs
        )

        if not self.api_key:
            raise LLMAPIError(
                "HuggingFace API key is required.\n"
                "Get your token at: https://huggingface.co/settings/tokens\n"
                "Set it in .env as: HUGGINGFACE_API_KEY=your_token_here"
            )

        # Initialize OpenAI client with HuggingFace endpoint
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
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Additional parameters (top_p, etc.)

        Returns:
            Generated text as string

        Raises:
            LLMError: If generation fails
        """
        use_temperature = temperature if temperature is not None else self.temperature
        use_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        # Retry loop
        for attempt in range(1, self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=use_temperature,
                    max_tokens=use_max_tokens,
                    **kwargs
                )

                # Extract and return generated text
                if completion.choices and len(completion.choices) > 0:
                    message = completion.choices[0].message
                    if message and message.content:
                        return message.content.strip()
                    else:
                        raise LLMInvalidResponseError("Response message is empty")
                else:
                    raise LLMInvalidResponseError("No choices in response")

            except AuthenticationError as e:
                raise LLMAPIError(
                    f"Authentication failed: {e}\n"
                    "Please check your HuggingFace API token.\n"
                    "Get your token at: https://huggingface.co/settings/tokens"
                )

            except RateLimitError as e:
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                    print(f"  Rate limited. Retrying in {wait_time}s... (attempt {attempt}/{self.max_retries})")
                    time.sleep(wait_time)
                else:
                    raise LLMRateLimitError(
                        f"Rate limit exceeded after {self.max_retries} retries: {e}\n"
                        "Consider using a different model or waiting before retrying."
                    )

            except APITimeoutError as e:
                if attempt < self.max_retries:
                    print(f"  Request timeout. Retrying... (attempt {attempt}/{self.max_retries})")
                    time.sleep(1)
                else:
                    raise LLMTimeoutError(
                        f"Request timed out after {self.max_retries} retries ({self.timeout}s each): {e}"
                    )

            except APIError as e:
                # Check if it's a model loading error (503)
                error_str = str(e)
                if "503" in error_str or "loading" in error_str.lower():
                    if attempt < self.max_retries:
                        wait_time = 5
                        print(f"  Model loading. Waiting {wait_time}s... (attempt {attempt}/{self.max_retries})")
                        time.sleep(wait_time)
                    else:
                        raise LLMAPIError(
                            f"Model '{self.model_name}' is still loading after {self.max_retries} retries.\n"
                            "Try again in a few minutes or use a different model."
                        )
                else:
                    raise LLMAPIError(f"API error: {e}")

            except Exception as e:
                raise LLMError(f"Unexpected error: {e}")

        # Should not reach here, but just in case
        raise LLMError("Generation failed after all retries")

    def generate_batch(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts.

        Note: Processes each prompt sequentially.

        Args:
            prompts: List of input text prompts
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Additional generation parameters

        Returns:
            List of generated texts
        """
        results = []
        for i, prompt in enumerate(prompts):
            print(f"Processing prompt {i+1}/{len(prompts)}...")
            result = self.generate(prompt, temperature, max_tokens, **kwargs)
            results.append(result)
        return results

    def is_available(self) -> bool:
        """
        Check if the provider is available.

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Simple test request
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                timeout=5
            )
            return True
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration info."""
        info = super().get_model_info()
        info['base_url'] = self.BASE_URL
        info['has_api_key'] = bool(self.api_key)
        return info
