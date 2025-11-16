"""
LLM Interface - Model-agnostic language model interface

Provides a unified interface for different LLM providers.

Recommended usage (simple, model-agnostic):
    from llm_interface import create_llm_provider

    # All configuration from .env file
    llm = create_llm_provider()
    response = llm.generate("Translate this ASL gloss to English: DOG WALK NOW")
    print(response)

Advanced usage (manual configuration):
    from llm_interface import HuggingFaceProvider

    llm = HuggingFaceProvider(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        api_key="your_hf_token_here",
        temperature=0.7,
        max_tokens=500
    )
    response = llm.generate("Your prompt here")
"""

from .llm_interface import (
    LLMInterface,
    LLMError,
    LLMAPIError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMInvalidResponseError
)

from .providers import HuggingFaceProvider
from .llm_factory import create_llm_provider, get_current_model_info

__all__ = [
    'LLMInterface',
    'HuggingFaceProvider',
    'create_llm_provider',
    'get_current_model_info',
    'LLMError',
    'LLMAPIError',
    'LLMTimeoutError',
    'LLMRateLimitError',
    'LLMInvalidResponseError'
]

__version__ = '1.0.0'
