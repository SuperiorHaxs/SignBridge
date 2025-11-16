#!/usr/bin/env python3
"""
LLM Factory - Creates LLM providers from environment configuration

This module provides a simple factory function to create LLM providers
without needing to specify model details in application code.
"""

import os
from pathlib import Path
from typing import Optional

# Load .env file
try:
    from dotenv import load_dotenv

    # Try to load from llm_interface directory first
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try project root
        project_root = Path(__file__).parent.parent.parent
        env_path_root = project_root / '.env'
        if env_path_root.exists():
            load_dotenv(env_path_root)
except ImportError:
    pass  # dotenv not available, will use system environment

from providers.huggingface_provider import HuggingFaceProvider
from providers.groq_provider import GroqProvider
from providers.googleaistudio_provider import GoogleAIStudioProvider
from llm_interface import LLMError, LLMAPIError


def create_llm_provider(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None
):
    """
    Create an LLM provider using environment configuration.

    This is the recommended way to create LLM providers - all configuration
    is read from the .env file, so you only need to update the .env file
    to test different models and providers.

    Args:
        provider: Override provider (default: from LLM_PROVIDER env var, "huggingface" or "groq")
        api_key: Override API key (default: from provider-specific env var)
        model_name: Override model (default: from LLM_MODEL env var)
        temperature: Override temperature (default: from LLM_TEMPERATURE env var)
        max_tokens: Override max tokens (default: from LLM_MAX_TOKENS env var)
        timeout: Override timeout (default: from LLM_TIMEOUT env var)

    Returns:
        LLM provider instance configured from environment

    Raises:
        LLMAPIError: If API key is not provided or found in environment

    Example:
        # Simple usage - all config from .env
        llm = create_llm_provider()
        response = llm.generate("Your prompt here")

        # Override specific parameters if needed
        llm = create_llm_provider(temperature=0.9)

        # Use specific provider
        llm = create_llm_provider(provider="groq")
    """
    # Get provider selection
    use_provider = provider or os.getenv('LLM_PROVIDER', 'huggingface').lower()

    # Get model configuration from environment
    use_model = model_name or os.getenv('LLM_MODEL')
    use_temperature = temperature if temperature is not None else float(os.getenv('LLM_TEMPERATURE', '0.7'))
    use_max_tokens = max_tokens if max_tokens is not None else int(os.getenv('LLM_MAX_TOKENS', '1000'))
    use_timeout = timeout if timeout is not None else int(os.getenv('LLM_TIMEOUT', '30'))

    # Create provider based on selection
    if use_provider == 'groq':
        # Get Groq API key
        use_api_key = api_key or os.getenv('GROQ_API_KEY')
        if not use_api_key:
            raise LLMAPIError(
                "GROQ_API_KEY not found.\n"
                "Please set it in your .env file or pass it directly.\n"
                "Get your free API key at: https://console.groq.com/keys"
            )

        # Default model for Groq
        if not use_model:
            use_model = 'llama-3.3-70b-versatile'

        return GroqProvider(
            model_name=use_model,
            api_key=use_api_key,
            temperature=use_temperature,
            max_tokens=use_max_tokens,
            timeout=use_timeout
        )

    elif use_provider == 'googleaistudio':
        # Get Google AI Studio API key
        use_api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not use_api_key:
            raise LLMAPIError(
                "GOOGLE_API_KEY not found.\n"
                "Please set it in your .env file or pass it directly.\n"
                "Get your free API key at: https://aistudio.google.com/app/apikey"
            )

        # Default model for Google AI Studio
        if not use_model:
            use_model = 'gemini-2.0-flash-exp'  # Note: provider adds 'models/' prefix automatically

        return GoogleAIStudioProvider(
            model_name=use_model,
            api_key=use_api_key,
            temperature=use_temperature,
            max_tokens=use_max_tokens,
            timeout=use_timeout
        )

    elif use_provider == 'huggingface':
        # Get HuggingFace API key
        use_api_key = api_key or os.getenv('HUGGINGFACE_API_KEY')
        if not use_api_key:
            raise LLMAPIError(
                "HUGGINGFACE_API_KEY not found.\n"
                "Please set it in your .env file or pass it directly.\n"
                "Get your token at: https://huggingface.co/settings/tokens"
            )

        # Default model for HuggingFace
        if not use_model:
            use_model = 'meta-llama/Llama-3.1-8B-Instruct'

        return HuggingFaceProvider(
            model_name=use_model,
            api_key=use_api_key,
            temperature=use_temperature,
            max_tokens=use_max_tokens,
            timeout=use_timeout
        )

    else:
        raise LLMAPIError(
            f"Unknown provider: {use_provider}\n"
            f"Supported providers: huggingface, groq, googleaistudio"
        )


def get_current_model_info():
    """
    Get information about the currently configured model and provider.

    Returns:
        Dictionary with current model configuration
    """
    provider = os.getenv('LLM_PROVIDER', 'huggingface').lower()

    # Get default model based on provider
    if provider == 'groq':
        default_model = 'llama-3.3-70b-versatile'
        api_key_set = bool(os.getenv('GROQ_API_KEY'))
    else:
        default_model = 'meta-llama/Llama-3.1-8B-Instruct'
        api_key_set = bool(os.getenv('HUGGINGFACE_API_KEY'))

    return {
        'provider': provider,
        'model': os.getenv('LLM_MODEL', default_model),
        'temperature': float(os.getenv('LLM_TEMPERATURE', '0.7')),
        'max_tokens': int(os.getenv('LLM_MAX_TOKENS', '1000')),
        'timeout': int(os.getenv('LLM_TIMEOUT', '30')),
        'api_key_set': api_key_set
    }
