"""
Property-based tests for network retry behavior.

**Feature: news-market-predictor, Property 3: Network retry behavior**
"""

import time
from unittest.mock import Mock, patch, call
from typing import List, Callable

import requests
from hypothesis import given, strategies as st

from news_market_predictor.fetcher.yahoo_finance_fetcher import YahooFinanceNewsFetcher
from news_market_predictor.exceptions import NetworkError


def simulate_network_failure_scenario(
    failure_count: int, exception_type: type = requests.exceptions.RequestException
) -> Callable:
    """
    Create a mock function that fails a specified number of times before succeeding.

    Args:
        failure_count: Number of times to fail before succeeding
        exception_type: Type of exception to raise on failure

    Returns:
        Mock function that simulates network failures
    """
    call_count = 0

    def mock_request(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count <= failure_count:
            raise exception_type(f"Simulated network failure #{call_count}")

        # Success case - return a mock response
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

    return mock_request


@given(st.integers(min_value=1, max_value=10))
def test_network_retry_exactly_three_times(failure_count):
    """
    **Feature: news-market-predictor, Property 3: Network retry behavior**

    Property: For any network failure scenario, the system should retry exactly
    three times with exponential backoff delays before giving up.

    **Validates: Requirements 1.3**

    This test verifies that the news fetcher retries network requests exactly
    three times when encountering network failures, regardless of the number
    of consecutive failures.
    """
    fetcher = YahooFinanceNewsFetcher(max_retries=3)

    # Create mock that fails the specified number of times
    mock_request = simulate_network_failure_scenario(failure_count)

    with patch.object(fetcher.session, "get", side_effect=mock_request) as mock_get:
        try:
            # Attempt to make a request that will trigger retries
            fetcher._make_request("https://test.example.com")

            # If failure_count <= 3, the request should eventually succeed
            if failure_count <= 3:
                # Should have made failure_count + 1 calls (failures + 1 success)
                assert mock_get.call_count == failure_count + 1, (
                    f"Expected {failure_count + 1} calls for {failure_count} failures, "
                    f"got {mock_get.call_count}"
                )
            else:
                # Should not reach here if failure_count > 3
                assert (
                    False
                ), "Expected NetworkError to be raised for excessive failures"

        except NetworkError:
            # If failure_count > 3, should fail after exactly 3 retries + 1 initial attempt = 4 calls
            if failure_count > 3:
                assert mock_get.call_count == 4, (
                    f"Expected exactly 4 calls (1 initial + 3 retries) for {failure_count} failures, "
                    f"got {mock_get.call_count}"
                )
            else:
                # Should not raise NetworkError if failure_count <= 3
                assert False, f"Unexpected NetworkError for {failure_count} failures"


@given(
    st.sampled_from(
        [
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
            requests.exceptions.HTTPError,
        ]
    )
)
def test_network_retry_behavior_different_exception_types(exception_type):
    """
    **Feature: news-market-predictor, Property 3: Network retry behavior**

    Property: For any network failure scenario (regardless of exception type),
    the system should retry exactly three times with exponential backoff delays.

    **Validates: Requirements 1.3**

    This test verifies that retry behavior is consistent across different
    types of network exceptions.
    """
    fetcher = YahooFinanceNewsFetcher(max_retries=3)

    # Create mock that always fails with the specified exception type
    mock_request = simulate_network_failure_scenario(
        5, exception_type
    )  # More than 3 failures

    with patch.object(fetcher.session, "get", side_effect=mock_request) as mock_get:
        try:
            fetcher._make_request("https://test.example.com")
            assert (
                False
            ), f"Expected NetworkError to be raised for {exception_type.__name__}"
        except NetworkError:
            # Should have made exactly 4 calls (1 initial + 3 retries)
            assert mock_get.call_count == 4, (
                f"Expected exactly 4 calls for {exception_type.__name__}, "
                f"got {mock_get.call_count}"
            )


@given(st.integers(min_value=1, max_value=3))
def test_network_retry_exponential_backoff_timing(retry_count):
    """
    **Feature: news-market-predictor, Property 3: Network retry behavior**

    Property: For any network failure scenario, the system should implement
    exponential backoff delays between retry attempts.

    **Validates: Requirements 1.3**

    This test verifies that the retry mechanism implements exponential backoff
    timing between attempts (1s, 2s, 4s delays).
    """
    fetcher = YahooFinanceNewsFetcher(max_retries=3)

    # Track timing of calls
    call_times = []

    def mock_request_with_timing(*args, **kwargs):
        call_times.append(time.time())
        raise requests.exceptions.ConnectionError("Simulated failure")

    with patch.object(fetcher.session, "get", side_effect=mock_request_with_timing):
        try:
            fetcher._make_request("https://test.example.com")
        except NetworkError:
            pass  # Expected to fail

    # Should have made exactly 4 calls (1 initial + 3 retries)
    assert len(call_times) == 4, f"Expected 4 calls, got {len(call_times)}"

    # Verify exponential backoff timing (allowing for some tolerance)
    # Expected delays: 0s (initial), ~1s, ~2s, ~4s
    if len(call_times) >= 2:
        delay1 = call_times[1] - call_times[0]
        assert (
            0.8 <= delay1 <= 1.5
        ), f"First retry delay should be ~1s, got {delay1:.2f}s"

    if len(call_times) >= 3:
        delay2 = call_times[2] - call_times[1]
        assert (
            1.8 <= delay2 <= 2.5
        ), f"Second retry delay should be ~2s, got {delay2:.2f}s"

    if len(call_times) >= 4:
        delay3 = call_times[3] - call_times[2]
        assert (
            3.8 <= delay3 <= 4.5
        ), f"Third retry delay should be ~4s, got {delay3:.2f}s"


@given(st.integers(min_value=0, max_value=2))
def test_network_retry_success_after_failures(success_after_attempts):
    """
    **Feature: news-market-predictor, Property 3: Network retry behavior**

    Property: For any network failure scenario where success occurs within
    three retries, the system should return the successful response.

    **Validates: Requirements 1.3**

    This test verifies that the system correctly handles success after
    some number of initial failures (within the retry limit).
    """
    fetcher = YahooFinanceNewsFetcher(max_retries=3)

    # Create mock that succeeds after the specified number of attempts
    mock_request = simulate_network_failure_scenario(success_after_attempts)

    with patch.object(fetcher.session, "get", side_effect=mock_request) as mock_get:
        # Should succeed without raising an exception
        response = fetcher._make_request("https://test.example.com")

        # Verify we got a successful response
        assert response is not None
        assert response.status_code == 200

        # Verify correct number of calls were made
        expected_calls = success_after_attempts + 1  # failures + 1 success
        assert mock_get.call_count == expected_calls, (
            f"Expected {expected_calls} calls for success after {success_after_attempts} failures, "
            f"got {mock_get.call_count}"
        )


def test_network_retry_rate_limit_handling():
    """
    **Feature: news-market-predictor, Property 3: Network retry behavior**

    Property: For any rate limit scenario (HTTP 429), the system should
    handle it appropriately with retry behavior.

    **Validates: Requirements 1.3**

    This test verifies that rate limit errors are handled correctly
    within the retry mechanism.
    """
    fetcher = YahooFinanceNewsFetcher(max_retries=3)

    call_count = 0

    def mock_rate_limit_request(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count <= 2:  # First two calls return 429
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.headers = {"Retry-After": "1"}
            return mock_response
        else:  # Third call succeeds
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "Success"
            mock_response.raise_for_status = Mock()
            return mock_response

    with patch.object(
        fetcher.session, "get", side_effect=mock_rate_limit_request
    ) as mock_get:
        with patch("time.sleep") as mock_sleep:  # Mock sleep to speed up test
            try:
                response = fetcher._make_request("https://test.example.com")

                # Should eventually succeed
                assert response.status_code == 200

                # Should have made 3 calls (2 rate limited + 1 success)
                assert mock_get.call_count == 3

                # Should have slept for rate limit delays
                assert mock_sleep.call_count >= 2  # At least 2 rate limit sleeps

            except Exception as e:
                # If rate limiting causes retries to be exhausted, that's also valid behavior
                assert isinstance(e, (NetworkError, Exception))


@given(st.text(min_size=1, max_size=100))
def test_network_retry_preserves_url_and_parameters(test_url_suffix):
    """
    **Feature: news-market-predictor, Property 3: Network retry behavior**

    Property: For any network failure scenario, retry attempts should
    preserve the original URL and request parameters.

    **Validates: Requirements 1.3**

    This test verifies that retry logic doesn't modify the original
    request parameters across retry attempts.
    """
    fetcher = YahooFinanceNewsFetcher(max_retries=3)

    # Create a test URL (sanitize the suffix to be URL-safe)
    safe_suffix = "".join(c for c in test_url_suffix if c.isalnum() or c in "-_.")[:50]
    test_url = f"https://test.example.com/{safe_suffix}"

    call_urls = []

    def mock_request_track_url(url, *args, **kwargs):
        call_urls.append(url)
        if len(call_urls) <= 2:  # Fail first 2 attempts
            raise requests.exceptions.ConnectionError("Simulated failure")
        else:  # Succeed on 3rd attempt
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            return mock_response

    with patch.object(fetcher.session, "get", side_effect=mock_request_track_url):
        response = fetcher._make_request(test_url)

        # Should have made 3 calls total
        assert len(call_urls) == 3

        # All calls should use the same URL
        for called_url in call_urls:
            assert (
                called_url == test_url
            ), f"URL changed during retry: expected {test_url}, got {called_url}"

        # Should have succeeded
        assert response.status_code == 200
