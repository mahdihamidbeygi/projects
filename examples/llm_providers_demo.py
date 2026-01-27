#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script showing how to use the new LLM providers.

This script demonstrates:
1. Setting up Google Gemini provider (requires API key)
2. Setting up Ollama provider (requires local Ollama installation)
3. Using the service manager with fallback
4. Generating completions and structured output
"""

import os
import json
from news_market_predictor.llm.service_manager import LLMServiceManager
from news_market_predictor.llm.models import LLMConfiguration
from news_market_predictor.llm.providers.gemini_provider import (
    GeminiProvider,
    GeminiConfig,
)
from news_market_predictor.llm.providers.ollama_provider import (
    OllamaProvider,
    OllamaConfig,
)


def demo_gemini_provider():
    """Demo Google Gemini provider (requires API key)."""
    print("=== Google Gemini Provider Demo ===")

    # Check if API key is available
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  GEMINI_API_KEY environment variable not set. Skipping Gemini demo.")
        print(
            "   To use Gemini, get an API key from https://makersuite.google.com/app/apikey"
        )
        return None

    try:
        # Create Gemini provider
        gemini_config = GeminiConfig(api_key=api_key)
        provider = GeminiProvider(gemini_config)

        print(f"Gemini provider initialized")
        print(f"   Available: {provider.is_available()}")
        print(
            f"   Rate limits: {gemini_config.max_requests_per_minute}/min, {gemini_config.max_requests_per_day}/day"
        )

        # Test completion (commented out to avoid API usage in demo)
        # response = provider.generate_completion("What is artificial intelligence?")
        # print(f"   Response: {response.content[:100]}...")

        return provider

    except Exception as e:
        print(f"❌ Error setting up Gemini provider: {e}")
        return None


def demo_ollama_provider():
    """Demo Ollama provider (requires local Ollama installation)."""
    print("\n=== Ollama Provider Demo ===")

    try:
        # Create Ollama provider
        ollama_config = OllamaConfig()
        provider = OllamaProvider(ollama_config)

        print(f"Ollama provider initialized")
        print(f"   Available: {provider.is_available()}")
        print(f"   Base URL: {ollama_config.base_url}")

        # List available models
        models = provider.list_available_models()
        if models:
            print(
                f"   Available models: {', '.join(models[:3])}{'...' if len(models) > 3 else ''}"
            )
        else:
            print(
                "   No models found. Install Ollama and pull a model (e.g., 'ollama pull llama3.2')"
            )

        # Test completion (commented out to avoid long inference time in demo)
        # if models:
        #     response = provider.generate_completion("What is machine learning?")
        #     print(f"   Response: {response.content[:100]}...")

        return provider

    except Exception as e:
        print(f"❌ Error setting up Ollama provider: {e}")
        return None


def demo_service_manager(gemini_provider, ollama_provider):
    """Demo service manager with multiple providers."""
    print("\n=== Service Manager Demo ===")

    # Create service manager
    config = LLMConfiguration(
        default_provider="gemini" if gemini_provider else "ollama",
        fallback_enabled=True,
        cost_limit_per_hour=1.0,  # $1 per hour limit
    )
    manager = LLMServiceManager(config)

    # Add available providers
    if gemini_provider:
        manager.add_provider(gemini_provider)
        print("Added Gemini provider")

    if ollama_provider:
        manager.add_provider(ollama_provider)
        print("Added Ollama provider")

    if not gemini_provider and not ollama_provider:
        print("❌ No providers available")
        return

    # Show available providers
    available = manager.get_available_providers()
    print(f"   Available providers: {', '.join(available)}")

    # Demo structured output schema
    sentiment_schema = {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"],
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reasoning": {"type": "string"},
        },
        "required": ["sentiment", "confidence", "reasoning"],
    }

    print(f"\n   Example structured output schema:")
    print(f"   {json.dumps(sentiment_schema, indent=2)}")

    # Show usage stats
    stats = manager.get_usage_stats()
    for provider_name, provider_stats in stats.items():
        print(f"\n   {provider_name.title()} usage stats:")
        print(f"     Total requests: {provider_stats.total_requests}")
        print(
            f"     Success rate: {provider_stats.successful_requests}/{provider_stats.total_requests}"
        )
        print(f"     Total cost: ${provider_stats.total_cost:.4f}")


def demo_rate_limiting():
    """Demo rate limiting features."""
    print("\n=== Rate Limiting Demo ===")

    # Show Gemini rate limits
    gemini_config = GeminiConfig(api_key="demo-key")
    print(f"Gemini free tier limits:")
    print(f"  • {gemini_config.max_requests_per_minute} requests per minute")
    print(f"  • {gemini_config.max_requests_per_day} requests per day")
    print(
        f"  • ${gemini_config.cost_per_1k_input_tokens:.3f} per 1K input tokens (free tier)"
    )

    # Show Ollama (no limits)
    print(f"\nOllama local inference:")
    print(f"  • No rate limits")
    print(f"  • No costs (local inference)")
    print(f"  • Depends on local hardware performance")


def main():
    """Main demo function."""
    print("🚀 LLM Providers Demo")
    print("=" * 50)

    # Demo individual providers
    gemini_provider = demo_gemini_provider()
    ollama_provider = demo_ollama_provider()

    # Demo service manager
    demo_service_manager(gemini_provider, ollama_provider)

    # Demo rate limiting
    demo_rate_limiting()

    print("\n" + "=" * 50)
    print("Demo completed!")
    print("\nNext steps:")
    print("1. Set GEMINI_API_KEY environment variable to test Gemini")
    print("2. Install and run Ollama locally to test local inference")
    print("3. Use the service manager in your applications for robust LLM integration")


if __name__ == "__main__":
    main()
