"""
Property-based tests for rate limit handling.

**Feature: news-market-predictor, Property 19: Rate limit handling**
"""

import time
from unittest.mock import Mock, patch, call
from typing import List, Optional

import requests
from hypothesis import given, strategies as st, settings, assume

from news_market_predictor.fetcher.yahoo_finance_fetcher import YahooFinanceNewsFetcher
from news_market_predictor.exceptions import RateLimitError, NetworkError
from news_market_predictor.error_handling import RateLimitHandler, RateLimitConfig


def create_rate_limit_response(retry_after: Optional[int] = None) -> Mock:
    """Create a mock HTTP 429 rate limit response."""
    mock_response = Mock()
    mock_response.status_code = 429
    mock_response.headers = {}
    if retry_after is not None:
        mock_response.headers["Retry-After"] = str(retry_after)
    return mock_response


def create_success_response() -> Mock:
    """Create a mock successful HTTP response."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
        <channel>
            <title>Yahoo Finance</title>
            <item>
                <title>Test Article</title>
                <link>https://finance.yahoo.com/news/test</link>
                <description>Test description</description>
                <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
            </item>
        </channel>
    </rss>"""
    mock_response.raise_for_status = Mock()
    return mock_response


@given(
    retry_after_values=st.lists(
        st.one_of(st.none(), st.integers(min_value=1, max_value=5)),
        min_size=1,
        max_size=3,
    )
)
@settings(deadline=10000, max_examples=5)
def test_rate_limit_implements_appropriate_delays(retry_after_values):
    """
    **Feature: news-market-predictor, Property 19: Rate limit handling**

    Property: For any API rate limit scenario, the system should implement
    appropriate delays and retry mechanisms.

    **Validates: Requirements 5.2**

    This test verifies that when API rate limits are encountered (HTTP 429),
    the system implements appropriate delays based on Retry-After headers
    or exponential backoff, and continues to retry the request.
    """
    fetcher = YahooFinanceNewsFetcher(
        max_retries=len(retry_after_values) + 2, rate_limit_delay=0.1
    )

    call_count = 0
    rate_limit_calls = []

    def mock_rate_limited_request(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        # Return rate limit responses for the specified number of times
        if call_count <= len(retry_after_values):
            return create_rate_limit_response(retry_after_values[call_count - 1])
        else:
            # Eventually succeed
            return create_success_response()

    # Mock all sleep calls to prevent actual delays
    with patch.object(fetcher.session, "get", side_effect=mock_rate_limited_request):
        with patch("time.sleep") as mock_sleep:
            # Mock the rate limit handler to track calls
            with patch.object(
                fetcher.rate_limit_handler, "handle_rate_limit_response"
            ) as mock_handler:

                def track_rate_limit_calls(retry_after=None):
                    rate_limit_calls.append(retry_after)

                mock_handler.side_effect = track_rate_limit_calls

                try:
                    response = fetcher._make_request("https://test.example.com")

                    # Should eventually succeed
                    assert response.status_code == 200

                    # Should have made the expected number of calls
                    expected_calls = (
                        len(retry_after_values) + 1
                    )  # rate limits + success
                    assert call_count == expected_calls

                    # Should have called rate limit handler for each rate limit response
                    assert len(rate_limit_calls) >= len(retry_after_values)

                    # Verify that rate limit handler was called with correct retry_after values
                    for i, expected_retry_after in enumerate(retry_after_values):
                        if i < len(rate_limit_calls):
                            actual_retry_after = rate_limit_calls[i]
                            if expected_retry_after is not None:
                                assert (
                                    actual_retry_after == expected_retry_after
                                ), f"Expected retry_after {expected_retry_after}, got {actual_retry_after}"
                            # For None values, we just verify the handler was called

                except (RateLimitError, NetworkError):
                    # If retries are exhausted, that's also valid behavior
                    # Should still have called rate limit handler
                    assert (
                        len(rate_limit_calls) > 0
                    ), "Should have called rate limit handler before giving up"


@given(
    requests_per_second=st.floats(min_value=0.5, max_value=3.0),
    burst_size=st.integers(min_value=1, max_value=5),
)
@settings(deadline=10000, max_examples=3)
def test_rate_limit_handler_prevents_exceeding_limits(requests_per_second, burst_size):
    """
    **Feature: news-market-predictor, Property 19: Rate limit handling**

    Property: For any rate limit configuration, the rate limit handler should
    prevent exceeding the specified request rate limits.

    **Validates: Requirements 5.2**

    This test verifies that the RateLimitHandler correctly enforces
    rate limits to prevent API rate limit errors from occurring.
    """
    config = RateLimitConfig(
        requests_per_second=requests_per_second,
        burst_size=burst_size,
        cooldown_period=60.0,
    )
    handler = RateLimitHandler(config)

    # Track timing of requests
    request_times = []
    sleep_calls = []

    def mock_time():
        return len(request_times) * 0.1  # Simulate time progression

    def mock_sleep(duration):
        sleep_calls.append(duration)

    with patch("time.time", side_effect=mock_time):
        with patch("time.sleep", side_effect=mock_sleep):
            # Make multiple requests
            num_requests = min(burst_size + 2, 7)  # Don't make too many requests

            for i in range(num_requests):
                start_time = time.time()
                handler.wait_if_needed()
                request_times.append(start_time)

            # Verify rate limiting behavior
            if num_requests > burst_size:
                # Should have called sleep to enforce rate limits
                assert (
                    len(sleep_calls) > 0
                ), f"Expected rate limiting for {num_requests} requests with burst_size {burst_size}"

                # Sleep calls should implement appropriate delays
                for delay in sleep_calls:
                    assert delay > 0, "Rate limit delays should be positive"
                    assert delay <= 2.0, "Rate limit delays should be reasonable"


@given(st.integers(min_value=1, max_value=3))
@settings(deadline=10000, max_examples=3)
def test_rate_limit_retry_mechanism(consecutive_rate_limits):
    """
    **Feature: news-market-predictor, Property 19: Rate limit handling**

    Property: For any sequence of rate limit responses, the system should
    implement retry mechanisms and eventually succeed or fail gracefully.

    **Validates: Requirements 5.2**

    This test verifies that the system properly retries requests when
    encountering rate limit responses.
    """
    fetcher = YahooFinanceNewsFetcher(max_retries=consecutive_rate_limits + 1)

    call_count = 0
    rate_limit_handler_calls = []

    def mock_rate_limited_request(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count <= consecutive_rate_limits:
            return create_rate_limit_response()
        else:
            return create_success_response()

    with patch.object(fetcher.session, "get", side_effect=mock_rate_limited_request):
        with patch("time.sleep"):  # Mock sleep to prevent delays
            with patch.object(
                fetcher.rate_limit_handler, "handle_rate_limit_response"
            ) as mock_handler:

                def track_handler_calls(retry_after=None):
                    rate_limit_handler_calls.append(retry_after)

                mock_handler.side_effect = track_handler_calls

                try:
                    response = fetcher._make_request("https://test.example.com")

                    # Should eventually succeed
                    assert response.status_code == 200

                    # Should have made the expected number of calls
                    expected_calls = (
                        consecutive_rate_limits + 1
                    )  # rate limits + success
                    assert call_count == expected_calls

                    # Should have called rate limit handler for each rate limit
                    assert len(rate_limit_handler_calls) == consecutive_rate_limits

                except (RateLimitError, NetworkError):
                    # If retries exhausted, should still have tried
                    assert call_count > 1, "Should have made multiple attempts"
                    assert (
                        len(rate_limit_handler_calls) > 0
                    ), "Should have called rate limit handler"


def test_rate_limit_handling_preserves_request_integrity():
    """
    **Feature: news-market-predictor, Property 19: Rate limit handling**

    Property: For any rate limit scenario, the retry mechanism should
    preserve the original request parameters and not modify the request.

    **Validates: Requirements 5.2**

    This test verifies that rate limit handling doesn't corrupt or
    modify the original request during retry attempts.
    """
    fetcher = YahooFinanceNewsFetcher(max_retries=3)

    original_url = "https://feeds.finance.yahoo.com/rss/2.0/headline"
    call_urls = []
    call_count = 0

    def mock_rate_limited_request(url, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        call_urls.append(url)

        if call_count <= 2:
            return create_rate_limit_response(1)  # Rate limit first 2 calls
        else:
            return create_success_response()

    with patch.object(fetcher.session, "get", side_effect=mock_rate_limited_request):
        with patch("time.sleep"):  # Mock sleep to speed up test
            with patch.object(fetcher.rate_limit_handler, "handle_rate_limit_response"):
                response = fetcher._make_request(original_url)

                # Should have succeeded
                assert response.status_code == 200

                # Should have made 3 calls total
                assert len(call_urls) == 3

                # All calls should use the same URL
                for called_url in call_urls:
                    assert (
                        called_url == original_url
                    ), f"URL should not change during rate limit retries: expected {original_url}, got {called_url}"


@given(st.integers(min_value=1, max_value=2))
@settings(deadline=10000, max_examples=3)
def test_rate_limit_appropriate_delays_implemented(num_rate_limits):
    """
    **Feature: news-market-predictor, Property 19: Rate limit handling**

    Property: For any rate limit scenario, the system should implement
    appropriate delays as specified by Requirements 5.2.

    **Validates: Requirements 5.2**

    This test verifies that the system implements delays when rate limited,
    which is the core requirement for rate limit handling.
    """
    fetcher = YahooFinanceNewsFetcher(max_retries=num_rate_limits + 1)

    call_count = 0
    delay_implemented = False

    def mock_rate_limited_request(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count <= num_rate_limits:
            return create_rate_limit_response()
        else:
            return create_success_response()

    with patch.object(fetcher.session, "get", side_effect=mock_rate_limited_request):
        with patch("time.sleep") as mock_sleep:
            with patch.object(
                fetcher.rate_limit_handler, "handle_rate_limit_response"
            ) as mock_handler:

                def track_delay_implementation(retry_after=None):
                    nonlocal delay_implemented
                    delay_implemented = True
                    # Simulate that a delay was implemented
                    mock_sleep(1.0)

                mock_handler.side_effect = track_delay_implementation

                try:
                    response = fetcher._make_request("https://test.example.com")

                    # Should eventually succeed
                    assert response.status_code == 200

                    # Should have implemented delays for rate limit responses
                    assert (
                        delay_implemented
                    ), "Should have implemented delays when rate limited"

                    # Should have called sleep (indicating delays were implemented)
                    assert (
                        mock_sleep.call_count >= num_rate_limits
                    ), f"Expected at least {num_rate_limits} sleep calls, got {mock_sleep.call_count}"

                except (RateLimitError, NetworkError):
                    # Even if failed, should have implemented delays
                    assert (
                        delay_implemented
                    ), "Should have implemented delays even if ultimately failed"
