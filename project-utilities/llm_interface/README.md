# LLM Interface

Model-agnostic language model interface for ASL-v1 project.

## Overview

This module provides a unified interface for different LLM providers (HuggingFace, OpenAI, Anthropic, etc.), making it easy to switch between models without changing your code.

## Features

- ✓ Model-agnostic base interface
- ✓ HuggingFace Inference API support
- ✓ Built-in retry logic and error handling
- ✓ Rate limit handling
- ✓ Batch generation support
- ✓ Easy to extend with new providers

## Installation

### Required Packages

```bash
pip install openai python-dotenv
```

Or install from project requirements:

```bash
cd C:\Users\ashwi\Projects\WLASL-proj\asl-v1
pip install -r requirements.txt
```

### HuggingFace API Token Setup

**Step 1: Get API Token**

Get your free API token from: https://huggingface.co/settings/tokens

**Step 2: Configure .env File (Recommended)**

```bash
# Copy the template
cd project-utilities/llm_interface
cp .env.template .env

# Edit .env and add your token
# HUGGINGFACE_API_KEY=your_actual_token_here
```

**Alternative: Environment Variable**

If you prefer not to use .env:

```bash
# Linux/Mac
export HUGGINGFACE_API_KEY=your_token_here

# Windows (Command Prompt)
set HUGGINGFACE_API_KEY=your_token_here

# Windows (PowerShell)
$env:HUGGINGFACE_API_KEY="your_token_here"
```

## Quick Start

### Basic Usage

```python
import os
from llm_interface import HuggingFaceProvider

# Initialize provider
llm = HuggingFaceProvider(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    api_key=os.getenv('HUGGINGFACE_API_KEY'),
    temperature=0.7,
    max_tokens=500
)

# Generate text
prompt = "Translate this ASL gloss to English: DOG WALK NOW"
response = llm.generate(prompt)
print(response)
```

### With Custom Parameters

```python
# Generate with custom temperature and token limit
response = llm.generate(
    prompt="Hello, how are you?",
    temperature=0.3,  # More deterministic
    max_tokens=100
)
```

### Batch Generation

```python
prompts = [
    "Translate: HELLO",
    "Translate: GOODBYE",
    "Translate: THANK YOU"
]

responses = llm.generate_batch(prompts)
for prompt, response in zip(prompts, responses):
    print(f"{prompt} -> {response}")
```

## Testing

Run the test suite:

```bash
cd project-utilities/llm_interface

# With API key
python test_llm_interface.py --api-key your_token_here

# Or set environment variable
export HUGGINGFACE_API_KEY=your_token_here
python test_llm_interface.py

# Test specific model
python test_llm_interface.py --model meta-llama/Llama-3.1-8B-Instruct

# Skip batch tests (faster)
python test_llm_interface.py --skip-batch
```

## Recommended Models

### HuggingFace Models

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `meta-llama/Llama-3.3-70B-Instruct` | 70B | Slow | Best | Production, high quality |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | Fast | Good | Development, testing ⭐ |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Fast | Good | Lightweight tasks |
| `google/gemma-2-9b-it` | 9B | Medium | Good | General purpose |

**Note:** Provider suffix (e.g., `:novita`) is optional. The router will automatically select an available provider.

## Error Handling

The interface provides specific error types:

```python
from llm_interface import (
    LLMError,
    LLMAPIError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMInvalidResponseError
)

try:
    response = llm.generate(prompt)
except LLMRateLimitError:
    print("Rate limited! Wait and retry.")
except LLMTimeoutError:
    print("Request timed out.")
except LLMAPIError as e:
    print(f"API error: {e}")
except LLMError as e:
    print(f"General LLM error: {e}")
```

## Architecture

```
llm_interface/
├── __init__.py                          # Package exports
├── llm_interface.py                     # Base abstract class
├── providers/
│   ├── __init__.py
│   └── huggingface_provider.py         # HuggingFace implementation
├── test_llm_interface.py               # Test suite
└── README.md                            # This file
```

## Adding New Providers

To add a new provider (e.g., OpenAI):

1. Create `providers/openai_provider.py`
2. Inherit from `LLMInterface`
3. Implement required methods: `generate()`, `generate_batch()`, `is_available()`
4. Add to `providers/__init__.py`

Example:

```python
from llm_interface import LLMInterface

class OpenAIProvider(LLMInterface):
    def generate(self, prompt, **kwargs):
        # Your OpenAI implementation
        pass

    def generate_batch(self, prompts, **kwargs):
        # Your batch implementation
        pass

    def is_available(self):
        # Check OpenAI API availability
        pass
```

## API Reference

### `LLMInterface` (Base Class)

**Constructor Parameters:**
- `model_name` (str): Model identifier
- `api_key` (str, optional): API authentication key
- `temperature` (float): Sampling temperature (0.0-1.0)
- `max_tokens` (int): Maximum tokens to generate
- `timeout` (int): Request timeout in seconds
- `max_retries` (int): Maximum retry attempts

**Methods:**
- `generate(prompt, temperature=None, max_tokens=None, **kwargs)`: Generate text from prompt
- `generate_batch(prompts, temperature=None, max_tokens=None, **kwargs)`: Batch generation
- `is_available()`: Check provider availability
- `get_model_info()`: Get model configuration

### `HuggingFaceProvider`

Inherits all `LLMInterface` methods plus:

**Additional Features:**
- Automatic retry with exponential backoff
- Model loading detection (503 handling)
- Rate limit handling with retry
- Free tier support (no API key required, but rate limited)

## Troubleshooting

### "Authentication failed"
- Check your API token is correct
- Get a new token from: https://huggingface.co/settings/tokens
- Ensure token has correct permissions

### "Rate limit exceeded"
- Free tier has limited requests per hour
- Use an API key for higher limits
- Wait a few minutes and retry
- Consider using a smaller/faster model

### "Model is loading"
- HuggingFace loads models on-demand
- Wait 5-10 seconds and retry
- The interface automatically retries up to 3 times

### "Request timed out"
- Increase timeout parameter
- Use a smaller model
- Check internet connection

## License

Part of the ASL-v1 project.
