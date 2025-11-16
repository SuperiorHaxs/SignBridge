#!/usr/bin/env python3
"""
Test script for LLM Interface

Tests the LLM interface with configured model.

Usage:
    # 1. Copy .env.template to .env
    # 2. Configure your API key and model settings
    # 3. Run: python test_llm_interface.py

    # Skip batch tests (faster)
    python test_llm_interface.py --skip-batch

All configuration (model, API key, temperature, etc.) is read from the .env file.
To test different models, edit the settings in your .env file.
"""

import os
import sys
import argparse
from pathlib import Path

# Add llm_interface to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from llm_factory import create_llm_provider, get_current_model_info
    from llm_interface import (
        LLMError,
        LLMAPIError,
        LLMTimeoutError,
        LLMRateLimitError,
        LLMInvalidResponseError
    )
except ImportError as e:
    print(f"Error importing LLM interface: {e}")
    print("Make sure you're running from the llm_interface directory")
    sys.exit(1)


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_initialization():
    """Test provider initialization from .env configuration."""
    print_section("Test 1: Provider Initialization")

    try:
        # Create provider using factory (reads all config from .env)
        provider = create_llm_provider(
            max_tokens=200  # Override for testing (shorter responses)
        )

        print(f"✓ Provider initialized successfully")
        print(f"\nModel Information:")
        info = provider.get_model_info()
        for key, value in info.items():
            if key == 'has_api_key':
                value = "Yes" if value else "No (using free tier)"
            print(f"  {key}: {value}")

        return provider

    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        print(f"\nPlease check your .env configuration:")
        print(f"  1. Copy .env.template to .env")
        print(f"  2. Set your API key")
        print(f"  3. Configure LLM_MODEL if desired")
        return None


def test_availability(provider):
    """Test if LLM is available and responding."""
    print_section("Test 2: LLM Availability")

    print(f"Model: {provider.model_name}")
    print(f"\nMaking test request...")

    try:
        test_response = provider.generate("Say hello", max_tokens=10)
        print(f"✓ LLM is available and responding")
        print(f"  Test response: {test_response}")
        return True
    except Exception as e:
        print(f"✗ LLM test failed")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error message: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False


def test_simple_generation(provider):
    """Test simple text generation."""
    print_section("Test 3: Simple Text Generation")

    test_prompt = "Say 'Hello, the LLM interface is working!' and nothing else."

    print(f"Prompt: {test_prompt}")
    print(f"\nGenerating response...")

    try:
        response = provider.generate(test_prompt, temperature=0.3, max_tokens=50)
        print(f"\n✓ Generation successful!")
        print(f"\nResponse:")
        print(f"  {response}")
        return True

    except LLMRateLimitError as e:
        print(f"\n✗ Rate limit error: {e}")
        print(f"  Try again in a few minutes or use an API key")
        return False

    except LLMTimeoutError as e:
        print(f"\n✗ Timeout error: {e}")
        return False

    except LLMAPIError as e:
        print(f"\n✗ API error: {e}")
        return False

    except LLMError as e:
        print(f"\n✗ LLM error: {e}")
        return False

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_asl_translation(provider):
    """Test ASL gloss-to-sentence translation (real use case)."""
    print_section("Test 4: ASL Gloss Translation")

    test_prompt = """You are an expert in American Sign Language (ASL) and English grammar.
Given the following ASL glosses in order, construct a natural, grammatically correct English sentence.

ASL Glosses: DOG, WALK, NOW

Instructions:
1. Consider that ASL has different grammar rules than English
2. Add appropriate articles (a, an, the) where needed
3. Add appropriate prepositions and conjunctions
4. Ensure proper verb tenses and subject-verb agreement
5. Make the sentence sound natural and professional

Return only the constructed English sentence, nothing else."""

    print(f"Testing ASL translation: DOG, WALK, NOW")
    print(f"\nGenerating response...")

    try:
        response = provider.generate(test_prompt, temperature=0.7, max_tokens=100)
        print(f"\n✓ Generation successful!")
        print(f"\nTranslated sentence:")
        print(f"  {response}")

        # Check if response looks reasonable
        if len(response) > 5 and any(word in response.lower() for word in ['dog', 'walk', 'now']):
            print(f"\n✓ Response looks valid!")
            return True
        else:
            print(f"\n⚠ Response might not be correct")
            return False

    except Exception as e:
        print(f"\n✗ Translation failed: {e}")
        return False


def test_batch_generation(provider):
    """Test batch generation."""
    print_section("Test 5: Batch Generation")

    test_prompts = [
        "Translate to English: HELLO",
        "Translate to English: THANK YOU",
        "Translate to English: GOODBYE"
    ]

    print(f"Testing batch generation with {len(test_prompts)} prompts...")

    try:
        responses = provider.generate_batch(test_prompts, temperature=0.5, max_tokens=50)

        print(f"\n✓ Batch generation successful!")
        print(f"\nResults:")
        for i, (prompt, response) in enumerate(zip(test_prompts, responses), 1):
            print(f"\n{i}. Prompt: {prompt}")
            print(f"   Response: {response}")

        return True

    except Exception as e:
        print(f"\n✗ Batch generation failed: {e}")
        print(f"  Note: This might fail without an API key due to rate limiting")
        return False


def test_error_handling(provider):
    """Test error handling with invalid inputs."""
    print_section("Test 6: Error Handling")

    # Test 1: Empty prompt
    print("Test 6a: Empty prompt")
    try:
        response = provider.generate("", max_tokens=10)
        print(f"  Response: {response}")
    except Exception as e:
        print(f"  ✓ Handled gracefully: {type(e).__name__}")

    # Test 2: Very long prompt (might hit token limit)
    print("\nTest 6b: Very long prompt")
    long_prompt = "test " * 1000
    try:
        response = provider.generate(long_prompt, max_tokens=10)
        print(f"  ✓ Generated response despite long prompt")
    except Exception as e:
        print(f"  ✓ Handled gracefully: {type(e).__name__}")

    return True


def run_all_tests(skip_batch=False):
    """Run all tests using configuration from .env file."""
    print("\n" + "=" * 70)
    print("  LLM INTERFACE TEST SUITE")
    print("=" * 70)

    # Display current configuration
    print("\nCurrent LLM Configuration (from .env):")
    model_info = get_current_model_info()
    print(f"  Model: {model_info['model']}")
    print(f"  Temperature: {model_info['temperature']}")
    print(f"  Max Tokens: {model_info['max_tokens']}")
    print(f"  Timeout: {model_info['timeout']}s")
    print(f"  API Key: {'Set' if model_info['api_key_set'] else 'NOT SET'}")

    results = {}

    # Test 1: Initialization
    provider = test_initialization()
    results['initialization'] = provider is not None
    if not provider:
        print("\n✗ Cannot continue without successful initialization")
        return results

    # Test 2: Availability
    results['availability'] = test_availability(provider)
    if not results['availability']:
        print("\n⚠ LLM not available, skipping generation tests")
        return results

    # Test 3: Simple generation
    results['simple_generation'] = test_simple_generation(provider)

    # Test 4: ASL translation
    results['asl_translation'] = test_asl_translation(provider)

    # Test 5: Batch generation (optional, can be slow)
    if not skip_batch:
        results['batch_generation'] = test_batch_generation(provider)

    # Test 6: Error handling
    results['error_handling'] = test_error_handling(provider)

    # Print summary
    print_section("Test Summary")
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\nTests Passed: {passed}/{total}")
    print(f"\nDetailed Results:")
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name:20} {status}")

    if passed == total:
        print(f"\n🎉 All tests passed!")
    elif passed > 0:
        print(f"\n⚠ Some tests passed, some failed")
    else:
        print(f"\n✗ All tests failed")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Test LLM Interface with configured model'
    )
    parser.add_argument(
        '--skip-batch',
        action='store_true',
        help='Skip batch generation test (to save time/quota)'
    )

    args = parser.parse_args()

    # Run tests (all configuration from .env)
    results = run_all_tests(args.skip_batch)

    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
