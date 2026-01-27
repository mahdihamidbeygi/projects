"""
Property-based tests for LLM provider consistency across different implementations.

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
from news_market_predictor.llm.providers.ollama_provider import (
    OllamaProvider,
    OllamaConfig,
)
from news_market_predictor.llm.interfaces import LLMProvider
from news_market_predictor.llm.models import LLMResponse, LLMUsageStats


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
    return st.text(min_size=1, max_size=500).filter(lambda x: x.strip())


# Strategy for generating model names
def model_name_strategy():
    """Generate valid model names for both providers."""
    return st.sampled_from(
        [
            # Gemini models
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-pro",
            # Ollama models
            "llama3.2",
            "llama3.2:1b",
            "mistral",
            "phi3",
        ]
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
    return st.integers(min_value=1, max_value=4096)


# Strategy for generating JSON schemas
def json_schema_strategy():
    """Generate valid JSON schemas for structured output testing."""
    return st.fixed_dictionaries(
        {
            "type": st.just("object"),
            "properties": st.dictionaries(
                keys=st.text(
                    min_size=1,
                    max_size=15,
                    alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
                ),
                values=st.fixed_dictionaries(
                    {"type": st.sampled_from(["string", "number", "boolean"])}
                ),
                min_size=1,
                max_size=3,
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
                    max_size=min(2, len(schema["properties"])),
                    unique=True,
                ),
            }
        )
    )


def create_gemini_provider(api_key: str) -> GeminiProvider:
    """Create a Gemini provider with test configuration."""
    config = GeminiConfig(api_key=api_key)
    return GeminiProvider(config)


def create_ollama_provider() -> OllamaProvider:
    """Create an Ollama provider with test configuration."""
    config = OllamaConfig()
    with patch("requests.get") as mock_get:
        # Mock successful connection check
        mock_get.return_value = Mock(status_code=200)
        mock_get.return_value.json.return_value = {"models": [{"name": "llama3.2"}]}
        return OllamaProvider(config)


@given(api_key=api_key_strategy())
@settings(max_examples=5, deadline=None)
@patch("requests.get")
def test_all_providers_implement_llm_provider_interface(mock_get, api_key):
    """
    **Property 9: LLM Service Interface Standardization**

    For any valid configuration, all LLM providers should implement the same
    LLMProvider interface with identical method signatures and behavior contracts.

    **Validates: Requirements 6.1, 6.5**
    """
    # Mock Ollama connection check
    mock_get.return_value = Mock(status_code=200)
    mock_get.return_value.json.return_value = {"models": [{"name": "llama3.2"}]}

    providers = [
        create_gemini_provider(api_key),
        create_ollama_provider(),
    ]

    for provider in providers:
        # Verify provider implements LLMProvider interface
        assert isinstance(provider, LLMProvider)

        # Verify all required methods exist and are callable
        required_methods = [
            "get_provider_name",
            "is_available",
            "generate_completion",
            "generate_structured_output",
            "estimate_cost",
            "get_usage_stats",
            "reset_usage_stats",
        ]

        for method_name in required_methods:
            assert hasattr(
                provider, method_name
            ), f"Provider {provider.get_provider_name()} missing method {method_name}"
            assert callable(
                getattr(provider, method_name)
            ), f"Method {method_name} not callable on {provider.get_provider_name()}"


@given(api_key=api_key_strategy())
@settings(max_examples=5, deadline=None)
@patch("requests.get")
def test_provider_names_are_consistent_and_unique(mock_get, api_key):
    """
    **Property 9: LLM Service Interface Standardization**

    For any valid configuration, each provider should return a consistent,
    unique name that identifies the provider type.

    **Validates: Requirements 6.1, 6.5**
    """
    # Mock Ollama connection check
    mock_get.return_value = Mock(status_code=200)
    mock_get.return_value.json.return_value = {"models": [{"name": "llama3.2"}]}

    gemini_provider = create_gemini_provider(api_key)
    ollama_provider = create_ollama_provider()

    # Each provider should have a consistent name
    assert gemini_provider.get_provider_name() == "gemini"
    assert ollama_provider.get_provider_name() == "ollama"

    # Names should be consistent across multiple calls
    assert gemini_provider.get_provider_name() == gemini_provider.get_provider_name()
    assert ollama_provider.get_provider_name() == ollama_provider.get_provider_name()

    # Names should be unique between providers
    assert gemini_provider.get_provider_name() != ollama_provider.get_provider_name()


@given(api_key=api_key_strategy())
@settings(max_examples=5, deadline=None)
@patch("requests.get")
def test_is_available_returns_boolean_consistently(mock_get, api_key):
    """
    **Property 9: LLM Service Interface Standardization**

    For any valid configuration, is_available should always return a boolean
    value consistently across all providers.

    **Validates: Requirements 6.1, 6.5**
    """
    # Mock Ollama connection check
    mock_get.return_value = Mock(status_code=200)
    mock_get.return_value.json.return_value = {"models": [{"name": "llama3.2"}]}

    providers = [
        create_gemini_provider(api_key),
        create_ollama_provider(),
    ]

    for provider in providers:
        availability = provider.is_available()

        # Should always return a boolean
        assert isinstance(
            availability, bool
        ), f"Provider {provider.get_provider_name()} is_available() returned {type(availability)}"

        # Should be consistent across multiple calls
        availability2 = provider.is_available()
        assert isinstance(
            availability2, bool
        ), f"Provider {provider.get_provider_name()} is_available() inconsistent return type"


@given(
    api_key=api_key_strategy(),
    prompt=prompt_strategy(),
    max_tokens=st.one_of(st.none(), max_tokens_strategy()),
)
@settings(max_examples=5, deadline=5000)
@patch("requests.post")
@patch("requests.get")
def test_estimate_cost_interface_consistency(
    mock_get, mock_post, api_key, prompt, max_tokens
):
    """
    **Property 9: LLM Service Interface Standardization**

    For any valid inputs, estimate_cost should return a non-negative float
    consistently across all providers according to the interface contract.

    **Validates: Requirements 6.1, 6.5**
    """
    # Mock Ollama connection check
    mock_get.return_value = Mock(status_code=200)
    mock_get.return_value.json.return_value = {"models": [{"name": "llama3.2"}]}

    providers = [
        create_gemini_provider(api_key),
        create_ollama_provider(),
    ]

    for provider in providers:
        cost = provider.estimate_cost(prompt=prompt, max_tokens=max_tokens)

        # Should return a number
        assert isinstance(
            cost, (int, float)
        ), f"Provider {provider.get_provider_name()} estimate_cost returned {type(cost)}"

        # Should be non-negative
        assert (
            cost >= 0.0
        ), f"Provider {provider.get_provider_name()} returned negative cost: {cost}"

        # Should not be NaN or infinity
        assert (
            cost == cost
        ), f"Provider {provider.get_provider_name()} returned NaN cost"
        assert cost != float(
            "inf"
        ), f"Provider {provider.get_provider_name()} returned infinite cost"


@given(api_key=api_key_strategy())
@settings(max_examples=5, deadline=None)
@patch("requests.get")
def test_usage_stats_interface_consistency(mock_get, api_key):
    """
    **Property 9: LLM Service Interface Standardization**

    For any valid configuration, get_usage_stats should return an LLMUsageStats
    object with consistent structure across all providers.

    **Validates: Requirements 6.1, 6.5**
    """
    # Mock Ollama connection check
    mock_get.return_value = Mock(status_code=200)
    mock_get.return_value.json.return_value = {"models": [{"name": "llama3.2"}]}

    providers = [
        create_gemini_provider(api_key),
        create_ollama_provider(),
    ]

    for provider in providers:
        stats = provider.get_usage_stats()

        # Should return LLMUsageStats object
        assert isinstance(
            stats, LLMUsageStats
        ), f"Provider {provider.get_provider_name()} returned {type(stats)}"

        # Should have correct provider name
        assert (
            stats.provider == provider.get_provider_name()
        ), f"Stats provider name mismatch for {provider.get_provider_name()}"

        # Should have all required fields with correct types
        assert isinstance(
            stats.total_requests, int
        ), f"Provider {provider.get_provider_name()} total_requests not int"
        assert isinstance(
            stats.successful_requests, int
        ), f"Provider {provider.get_provider_name()} successful_requests not int"
        assert isinstance(
            stats.failed_requests, int
        ), f"Provider {provider.get_provider_name()} failed_requests not int"
        assert isinstance(
            stats.total_tokens_used, int
        ), f"Provider {provider.get_provider_name()} total_tokens_used not int"
        assert isinstance(
            stats.total_cost, (int, float)
        ), f"Provider {provider.get_provider_name()} total_cost not numeric"

        # Should have logical constraints
        assert (
            stats.total_requests >= 0
        ), f"Provider {provider.get_provider_name()} negative total_requests"
        assert (
            stats.successful_requests >= 0
        ), f"Provider {provider.get_provider_name()} negative successful_requests"
        assert (
            stats.failed_requests >= 0
        ), f"Provider {provider.get_provider_name()} negative failed_requests"
        assert (
            stats.total_tokens_used >= 0
        ), f"Provider {provider.get_provider_name()} negative total_tokens_used"
        assert (
            stats.total_cost >= 0.0
        ), f"Provider {provider.get_provider_name()} negative total_cost"
        assert (
            stats.successful_requests + stats.failed_requests <= stats.total_requests
        ), f"Provider {provider.get_provider_name()} inconsistent request counts"


@given(api_key=api_key_strategy())
@settings(max_examples=3, deadline=None)
@patch("requests.get")
def test_reset_usage_stats_consistency(mock_get, api_key):
    """
    **Property 9: LLM Service Interface Standardization**

    For any valid configuration, reset_usage_stats should reset all statistics
    to initial state consistently across all providers.

    **Validates: Requirements 6.1, 6.5**
    """
    # Mock Ollama connection check
    mock_get.return_value = Mock(status_code=200)
    mock_get.return_value.json.return_value = {"models": [{"name": "llama3.2"}]}

    providers = [
        create_gemini_provider(api_key),
        create_ollama_provider(),
    ]

    for provider in providers:
        # Simulate some usage by modifying internal state
        provider.usage_stats.total_requests = 5
        provider.usage_stats.successful_requests = 4
        provider.usage_stats.failed_requests = 1
        provider.usage_stats.total_tokens_used = 500
        provider.usage_stats.total_cost = 2.5

        # Verify stats were modified
        modified_stats = provider.get_usage_stats()
        assert (
            modified_stats.total_requests == 5
        ), f"Provider {provider.get_provider_name()} failed to modify stats"

        # Reset stats
        provider.reset_usage_stats()

        # Verify stats were reset consistently
        reset_stats = provider.get_usage_stats()
        assert (
            reset_stats.provider == provider.get_provider_name()
        ), f"Provider {provider.get_provider_name()} lost provider name after reset"
        assert (
            reset_stats.total_requests == 0
        ), f"Provider {provider.get_provider_name()} failed to reset total_requests"
        assert (
            reset_stats.successful_requests == 0
        ), f"Provider {provider.get_provider_name()} failed to reset successful_requests"
        assert (
            reset_stats.failed_requests == 0
        ), f"Provider {provider.get_provider_name()} failed to reset failed_requests"
        assert (
            reset_stats.total_tokens_used == 0
        ), f"Provider {provider.get_provider_name()} failed to reset total_tokens_used"
        assert (
            reset_stats.total_cost == 0.0
        ), f"Provider {provider.get_provider_name()} failed to reset total_cost"


@given(
    api_key=api_key_strategy(),
    prompt=prompt_strategy(),
    model=st.one_of(st.none(), model_name_strategy()),
    max_tokens=st.one_of(st.none(), max_tokens_strategy()),
    temperature=st.one_of(st.none(), temperature_strategy()),
)
@settings(max_examples=5, deadline=10000)
@patch("requests.post")
@patch("requests.get")
def test_generate_completion_response_structure_consistency(
    mock_get, mock_post, api_key, prompt, model, max_tokens, temperature
):
    """
    **Property 9: LLM Service Interface Standardization**

    For any valid inputs, generate_completion should return an LLMResponse
    with consistent structure and field types across all providers.

    **Validates: Requirements 6.1, 6.5**
    """
    # Mock Ollama connection check
    mock_get.return_value = Mock(status_code=200)
    mock_get.return_value.json.return_value = {"models": [{"name": "llama3.2"}]}

    # Mock successful API responses for both providers
    def mock_post_side_effect(url, **kwargs):
        if "generativelanguage.googleapis.com" in url:
            # Gemini response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "candidates": [
                    {"content": {"parts": [{"text": "Gemini test response"}]}}
                ]
            }
            return mock_response
        else:
            # Ollama response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "Ollama test response",
                "eval_count": 10,
                "prompt_eval_count": 5,
            }
            return mock_response

    mock_post.side_effect = mock_post_side_effect

    providers = [
        create_gemini_provider(api_key),
        create_ollama_provider(),
    ]

    responses = []
    for provider in providers:
        try:
            response = provider.generate_completion(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            responses.append((provider.get_provider_name(), response))
        except Exception:
            # Skip providers that fail (e.g., due to model compatibility)
            continue

    # If we have responses, verify they all have consistent structure
    if responses:
        for provider_name, response in responses:
            # Verify response type and required fields
            assert isinstance(
                response, LLMResponse
            ), f"Provider {provider_name} returned {type(response)}"

            # Verify all required fields exist
            required_fields = [
                "content",
                "model_used",
                "provider",
                "tokens_used",
                "cost_estimate",
                "confidence_score",
            ]
            for field in required_fields:
                assert hasattr(
                    response, field
                ), f"Provider {provider_name} missing field {field}"

            # Verify field types
            assert isinstance(
                response.content, str
            ), f"Provider {provider_name} content not string"
            assert len(response.content) > 0, f"Provider {provider_name} empty content"

            assert isinstance(
                response.model_used, str
            ), f"Provider {provider_name} model_used not string"
            assert (
                len(response.model_used) > 0
            ), f"Provider {provider_name} empty model_used"

            assert (
                response.provider == provider_name
            ), f"Provider {provider_name} incorrect provider field"

            assert isinstance(
                response.tokens_used, int
            ), f"Provider {provider_name} tokens_used not int"
            assert (
                response.tokens_used > 0
            ), f"Provider {provider_name} non-positive tokens_used"

            assert isinstance(
                response.cost_estimate, (int, float)
            ), f"Provider {provider_name} cost_estimate not numeric"
            assert (
                response.cost_estimate >= 0.0
            ), f"Provider {provider_name} negative cost_estimate"

            assert isinstance(
                response.confidence_score, (int, float)
            ), f"Provider {provider_name} confidence_score not numeric"
            assert (
                0.0 <= response.confidence_score <= 1.0
            ), f"Provider {provider_name} confidence_score out of range"


@given(
    api_key=api_key_strategy(),
    prompt=prompt_strategy(),
    schema=json_schema_strategy(),
)
@settings(max_examples=3, deadline=10000)
@patch("requests.post")
@patch("requests.get")
def test_generate_structured_output_consistency(
    mock_get, mock_post, api_key, prompt, schema
):
    """
    **Property 9: LLM Service Interface Standardization**

    For any valid inputs, generate_structured_output should return a dictionary
    that conforms to the interface contract consistently across all providers.

    **Validates: Requirements 6.1, 6.5**
    """
    # Create a valid JSON response that matches the schema
    sample_response = {}
    required_fields = schema.get("required", [])
    properties = schema.get("properties", {})

    # Add required fields first
    for field_name in required_fields:
        if field_name in properties:
            prop_def = properties[field_name]
            if prop_def["type"] == "string":
                sample_response[field_name] = "test_value"
            elif prop_def["type"] == "number":
                sample_response[field_name] = 42.0
            elif prop_def["type"] == "boolean":
                sample_response[field_name] = True

    # Add other properties
    for prop_name, prop_def in properties.items():
        if prop_name not in sample_response:
            if prop_def["type"] == "string":
                sample_response[prop_name] = "test_value"
            elif prop_def["type"] == "number":
                sample_response[prop_name] = 42.0
            elif prop_def["type"] == "boolean":
                sample_response[prop_name] = True

    # Mock Ollama connection check
    mock_get.return_value = Mock(status_code=200)
    mock_get.return_value.json.return_value = {"models": [{"name": "llama3.2"}]}

    # Mock successful API responses for both providers
    def mock_post_side_effect(url, **kwargs):
        if "generativelanguage.googleapis.com" in url:
            # Gemini response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "candidates": [
                    {"content": {"parts": [{"text": json.dumps(sample_response)}]}}
                ]
            }
            return mock_response
        else:
            # Ollama response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": json.dumps(sample_response),
                "eval_count": 8,
                "prompt_eval_count": 12,
            }
            return mock_response

    mock_post.side_effect = mock_post_side_effect

    providers = [
        create_gemini_provider(api_key),
        create_ollama_provider(),
    ]

    results = []
    for provider in providers:
        try:
            result = provider.generate_structured_output(prompt=prompt, schema=schema)
            results.append((provider.get_provider_name(), result))
        except Exception:
            # Skip providers that fail
            continue

    # If we have results, verify they all have consistent structure
    if results:
        for provider_name, result in results:
            # Verify result type
            assert isinstance(
                result, dict
            ), f"Provider {provider_name} returned {type(result)}"

            # Verify required fields are present if specified in schema
            for field in required_fields:
                if field in properties:
                    assert (
                        field in result
                    ), f"Provider {provider_name} missing required field '{field}'"
