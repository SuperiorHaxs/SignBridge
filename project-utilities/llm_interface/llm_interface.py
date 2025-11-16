#!/usr/bin/env python3
"""
Model-agnostic LLM Interface

Provides a unified interface for different LLM providers (HuggingFace, OpenAI, Anthropic, etc.)
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class LLMInterface(ABC):
    """
    Abstract base class for LLM providers.

    All provider implementations must inherit from this class and implement
    the required methods.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        timeout: int = 30,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize LLM interface.

        Args:
            model_name: Name/identifier of the model to use
            api_key: API key for authentication (if required)
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts on failure
            **kwargs: Additional provider-specific parameters
        """
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_params = kwargs

    @abstractmethod
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
            **kwargs: Additional generation parameters

        Returns:
            Generated text as string

        Raises:
            LLMError: If generation fails after all retries
        """
        pass

    @abstractmethod
    def generate_batch(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts (batch processing).

        Args:
            prompts: List of input text prompts
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Additional generation parameters

        Returns:
            List of generated texts

        Raises:
            LLMError: If generation fails after all retries
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the LLM provider is available and properly configured.

        Returns:
            True if available, False otherwise
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.

        Returns:
            Dictionary with model information
        """
        return {
            'provider': self.__class__.__name__,
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'timeout': self.timeout,
            'max_retries': self.max_retries
        }


class LLMError(Exception):
    """Base exception for LLM-related errors"""
    pass


class LLMAPIError(LLMError):
    """Exception for API-related errors (network, authentication, etc.)"""
    pass


class LLMTimeoutError(LLMError):
    """Exception for timeout errors"""
    pass


class LLMRateLimitError(LLMError):
    """Exception for rate limit errors"""
    pass


class LLMInvalidResponseError(LLMError):
    """Exception for invalid/unexpected API responses"""
    pass
