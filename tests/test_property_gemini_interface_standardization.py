"""
Property-based tests for Gemini provider interface standardization.

**Feature: ai-enhanced-pipeline, Property 9: LLM Service Interface Standardization**
**Validates: Requirements 6.1, 6.5**
"""

import json
from unittest.mock import Mock, patch
from hypothesis import given, strategies as st, settings, assume

from news_market_predictor.llm.providers.gemini_provider import (
    GeminiProvider,
    GeminiConfig,
)
from news_market_predictor.llm.interfaces import LLMProvider
from news_market_predictor.llm.models import LLMResponse, LLMUsageStats
from news_market_predictor.llm.exceptions import (
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMInvalidResponseError,
)


# Strategy for generating valid API keys
def api_key_strategy():
    """Generate valid-looking API keys for testing."""
    return st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Pc")),
        min_size=20,
        max_size=50,
    )


# Strategy for generating prompts
def prompt_strategy():
    """Generate various prompts for testing."""
    return st.text(min_size=1, max_size=1000).filter(lambda x: x.strip())


# Strategy for generating model names
def model_name_strategy():
    """Generate valid model names."""
    return st.sampled_from(
        ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro", "gemini-pro-vision"]
    )


# Strategy for generating temperature values
def temperature_strategy():
    """Generate valid temperature values."""
    return st.floats(
        min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False
    )


# Strategy for generating max_tokens values
def max_tokens_strategy():
    """Generate valid max_tokens values."""
    return st.integers(min_value=1, max_value=8192)


# Strategy for generating JSON schemas
def json_schema_strategy():
    """Generate valid JSON schemas for structured output testing."""
    return st.fixed_dictionaries(
        {
            "type": st.just("object"),
            "properties": st.dictionaries(
                keys=st.text(
                    min_size=1,
                    max_size=20,
                    alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
                ),
                values=st.fixed_dictionaries(
                    {"type": st.sampled_from(["string", "number", "boolean", "array"])}
                ),
                min_size=1,
                max_size=5,
            ),
        }
    ).flatmap(
        lambda schema: st.fixed_dictionaries(
            {
                "type": st.just(schema["type"]),
                "properties": st.just(schema["properties"]),
                "required": st.lists(
                    st.sampled_from(list(schema["properties"].keys())),
                    min_size=0,
                    max_size=min(3, len(schema["properties"])),
                    unique=True,
                ),
            }
        )
    )


@given(api_key=api_key_strategy())
@settings(max_examples=50, deadline=5000)
def test_gemini_provider_implements_llm_provider_interface(api_key):
    """
    **Property 9: LLM Service Interface Standardization**

    For any valid API key, the Gemini provider should implement all required
    LLMProvider interface methods with consistent behavior.

    **Validates: Requirements 6.1, 6.5**
    """
    config = GeminiConfig(api_key=api_key)
    provider = GeminiProvider(config)

    # Verify provider implements LLMProvider interface
    assert isinstance(provider, LLMProvider)

    # Verify all required methods exist and are callable
    assert hasattr(provider, "get_provider_name")
    assert callable(provider.get_provider_name)

    assert hasattr(provider, "is_available")
    assert callable(provider.is_available)

    assert hasattr(provider, "generate_completion")
    assert callable(provider.generate_completion)

    assert hasattr(provider, "generate_structured_output")
    assert callable(provider.generate_structured_output)

    assert hasattr(provider, "estimate_cost")
    assert callable(provider.estimate_cost)

    assert hasattr(provider, "get_usage_stats")
    assert callable(provider.get_usage_stats)

    assert hasattr(provider, "reset_usage_stats")
    assert callable(provider.reset_usage_stats)


@given(api_key=api_key_strategy())
@settings(max_examples=50, deadline=5000)
def test_gemini_provider_name_consistency(api_key):
    """
    **Property 9: LLM Service Interface Standardization**

    For any valid configuration, the Gemini provider should consistently
    return "gemini" as its provider name.

    **Validates: Requirements 6.1, 6.5**
    """
    config = GeminiConfig(api_key=api_key)
    provider = GeminiProvider(config)

    # Provider name should always be "gemini"
    assert provider.get_provider_name() == "gemini"

    # Should be consistent across multiple calls
    assert provider.get_provider_name() == provider.get_provider_name()


@given(
    api_key=api_key_strategy(),
    prompt=prompt_strategy(),
    model=st.one_of(st.none(), model_name_strategy()),
    max_tokens=st.one_of(st.none(), max_tokens_strategy()),
    temperature=st.one_of(st.none(), temperature_strategy()),
)
@settings(max_examples=30, deadline=10000)
@patch("requests.post")
def test_gemini_generate_completion_interface_compliance(
    mock_post, api_key, prompt, model, max_tokens, temperature
):
    """
    **Property 9: LLM Service Interface Standardization**

    For any valid inputs, generate_completion should return an LLMResponse
    with all required fields populated according to the interface contract.

    **Validates: Requirements 6.1, 6.5**
    """
    # Mock successful API response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Test response content"}]}}]
    }
    mock_post.return_value = mock_response

    config = GeminiConfig(api_key=api_key)
    provider = GeminiProvider(config)

    # Generate completion
    response = provider.generate_completion(
        prompt=prompt, model=model, max_tokens=max_tokens, temperature=temperature
    )

    # Verify response type and required fields
    assert isinstance(response, LLMResponse)
    assert hasattr(response, "content")
    assert hasattr(response, "model_used")
    assert hasattr(response, "provider")
    assert hasattr(response, "tokens_used")
    assert hasattr(response, "cost_estimate")
    assert hasattr(response, "confidence_score")

    # Verify field types and values
    assert isinstance(response.content, str)
    assert len(response.content) > 0

    assert isinstance(response.model_used, str)
    assert len(response.model_used) > 0

    assert response.provider == "gemini"

    assert isinstance(response.tokens_used, int)
    assert response.tokens_used > 0

    assert isinstance(response.cost_estimate, (int, float))
    assert response.cost_estimate >= 0.0

    assert isinstance(response.confidence_score, (int, float))
    assert 0.0 <= response.confidence_score <= 1.0


@given(
    api_key=api_key_strategy(),
    prompt=prompt_strategy(),
    schema=json_schema_strategy(),
    model=st.one_of(st.none(), model_name_strategy()),
)
@settings(max_examples=20, deadline=10000)
@patch("requests.post")
def test_gemini_generate_structured_output_interface_compliance(
    mock_post, api_key, prompt, schema, model
):
    """
    **Property 9: LLM Service Interface Standardization**

    For any valid inputs, generate_structured_output should return a dictionary
    that conforms to the interface contract for structured output.

    **Validates: Requirements 6.1, 6.5**
    """
    # Create a valid JSON response that matches the schema
    sample_response = {}

    # First, add all required fields
    required_fields = schema.get("required", [])
    properties = schema.get("properties", {})

    for field_name in required_fields:
        if field_name in properties:
            prop_def = properties[field_name]
            if prop_def["type"] == "string":
                sample_response[field_name] = "test_value"
            elif prop_def["type"] == "number":
                sample_response[field_name] = 42.0
            elif prop_def["type"] == "boolean":
                sample_response[field_name] = True
            elif prop_def["type"] == "array":
                sample_response[field_name] = ["item1", "item2"]

    # Then add other properties
    for prop_name, prop_def in properties.items():
        if prop_name not in sample_response:  # Don't overwrite required fields
            if prop_def["type"] == "string":
                sample_response[prop_name] = "test_value"
            elif prop_def["type"] == "number":
                sample_response[prop_name] = 42.0
            elif prop_def["type"] == "boolean":
                sample_response[prop_name] = True
            elif prop_def["type"] == "array":
                sample_response[prop_name] = ["item1", "item2"]

    # Mock successful API response with valid JSON
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": json.dumps(sample_response)}]}}]
    }
    mock_post.return_value = mock_response

    config = GeminiConfig(api_key=api_key)
    provider = GeminiProvider(config)

    # Generate structured output
    result = provider.generate_structured_output(
        prompt=prompt, schema=schema, model=model
    )

    # Verify result type and structure
    assert isinstance(result, dict)

    # Verify required fields are present if specified in schema
    required_fields = schema.get("required", [])
    for field in required_fields:
        if field in schema.get("properties", {}):
            assert field in result, f"Required field '{field}' missing from result"


@given(
    api_key=api_key_strategy(),
    prompt=prompt_strategy(),
    model=st.one_of(st.none(), model_name_strategy()),
    max_tokens=st.one_of(st.none(), max_tokens_strategy()),
)
@settings(max_examples=50, deadline=5000)
def test_gemini_estimate_cost_interface_compliance(api_key, prompt, model, max_tokens):
    """
    **Property 9: LLM Service Interface Standardization**

    For any valid inputs, estimate_cost should return a non-negative float
    representing the estimated cost according to the interface contract.

    **Validates: Requirements 6.1, 6.5**
    """
    config = GeminiConfig(api_key=api_key)
    provider = GeminiProvider(config)

    # Estimate cost
    cost = provider.estimate_cost(prompt=prompt, max_tokens=max_tokens, model=model)

    # Verify cost type and constraints
    assert isinstance(cost, (int, float))
    assert cost >= 0.0
    assert not (cost != cost)  # Check for NaN
    assert cost != float("inf")  # Check for infinity


@given(api_key=api_key_strategy())
@settings(max_examples=50, deadline=5000)
def test_gemini_usage_stats_interface_compliance(api_key):
    """
    **Property 9: LLM Service Interface Standardization**

    For any valid configuration, get_usage_stats should return an LLMUsageStats
    object with all required fields according to the interface contract.

    **Validates: Requirements 6.1, 6.5**
    """
    config = GeminiConfig(api_key=api_key)
    provider = GeminiProvider(config)

    # Get usage stats
    stats = provider.get_usage_stats()

    # Verify stats type and required fields
    assert isinstance(stats, LLMUsageStats)
    assert hasattr(stats, "provider")
    assert hasattr(stats, "total_requests")
    assert hasattr(stats, "successful_requests")
    assert hasattr(stats, "failed_requests")
    assert hasattr(stats, "total_tokens_used")
    assert hasattr(stats, "total_cost")

    # Verify field types and constraints
    assert stats.provider == "gemini"
    assert isinstance(stats.total_requests, int)
    assert isinstance(stats.successful_requests, int)
    assert isinstance(stats.failed_requests, int)
    assert isinstance(stats.total_tokens_used, int)
    assert isinstance(stats.total_cost, (int, float))

    # Verify logical constraints
    assert stats.total_requests >= 0
    assert stats.successful_requests >= 0
    assert stats.failed_requests >= 0
    assert stats.total_tokens_used >= 0
    assert stats.total_cost >= 0.0
    assert stats.successful_requests + stats.failed_requests <= stats.total_requests


@given(api_key=api_key_strategy())
@settings(max_examples=30, deadline=5000)
def test_gemini_reset_usage_stats_interface_compliance(api_key):
    """
    **Property 9: LLM Service Interface Standardization**

    For any valid configuration, reset_usage_stats should reset all usage
    statistics to their initial state according to the interface contract.

    **Validates: Requirements 6.1, 6.5**
    """
    config = GeminiConfig(api_key=api_key)
    provider = GeminiProvider(config)

    # Get initial stats
    initial_stats = provider.get_usage_stats()

    # Simulate some usage by modifying internal state directly
    # Use absolute values to ensure we have a clear change
    provider.usage_stats.total_requests = 10
    provider.usage_stats.successful_requests = 8
    provider.usage_stats.failed_requests = 2
    provider.usage_stats.total_tokens_used = 1000
    provider.usage_stats.total_cost = 5.0

    # Verify stats were modified
    modified_stats = provider.get_usage_stats()
    assert modified_stats.total_requests == 10
    assert modified_stats.successful_requests == 8
    assert modified_stats.failed_requests == 2
    assert modified_stats.total_tokens_used == 1000
    assert modified_stats.total_cost == 5.0

    # Reset stats
    provider.reset_usage_stats()

    # Verify stats were reset
    reset_stats = provider.get_usage_stats()
    assert reset_stats.provider == "gemini"
    assert reset_stats.total_requests == 0
    assert reset_stats.successful_requests == 0
    assert reset_stats.failed_requests == 0
    assert reset_stats.total_tokens_used == 0
    assert reset_stats.total_cost == 0.0


@given(api_key=api_key_strategy())
@settings(max_examples=50, deadline=5000)
def test_gemini_is_available_interface_compliance(api_key):
    """
    **Property 9: LLM Service Interface Standardization**

    For any valid configuration, is_available should return a boolean
    indicating availability according to the interface contract.

    **Validates: Requirements 6.1, 6.5**
    """
    config = GeminiConfig(api_key=api_key)
    provider = GeminiProvider(config)

    # Check availability
    availability = provider.is_available()

    # Verify return type
    assert isinstance(availability, bool)

    # Should be consistent across multiple calls (within rate limits)
    availability2 = provider.is_available()
    assert isinstance(availability2, bool)
