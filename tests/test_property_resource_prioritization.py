"""
Property-based tests for resource prioritization.

**Feature: news-market-predictor, Property 22: Resource prioritization**
"""

from datetime import datetime
from unittest.mock import Mock, patch
from typing import List

from hypothesis import given, strategies as st, settings, assume

from news_market_predictor.models import NewsArticle
from news_market_predictor.error_handling import (
    ResourcePrioritizer,
    ResourceConstraints,
    Priority,
    ErrorHandlingManager,
)


# Strategy for generating NewsArticle objects with different impact levels
@st.composite
def article_with_priority_strategy(draw, priority_level=None):
    """Generate NewsArticle objects with specific priority indicators."""
    if priority_level is None:
        priority_level = draw(st.sampled_from(["high", "medium", "low"]))

    # Keywords that determine priority
    high_impact_keywords = [
        "earnings",
        "acquisition",
        "merger",
        "bankruptcy",
        "lawsuit",
        "fda approval",
        "clinical trial",
        "breakthrough",
        "partnership",
    ]

    medium_impact_keywords = [
        "revenue",
        "profit",
        "guidance",
        "upgrade",
        "downgrade",
        "analyst",
        "rating",
        "target price",
    ]

    low_impact_keywords = [
        "update",
        "news",
        "report",
        "statement",
        "comment",
        "discussion",
    ]

    # Select keywords based on priority level
    if priority_level == "high":
        keyword = draw(st.sampled_from(high_impact_keywords))
    elif priority_level == "medium":
        keyword = draw(st.sampled_from(medium_impact_keywords))
    else:
        keyword = draw(st.sampled_from(low_impact_keywords))

    # Generate article with keyword in title or content
    base_title = draw(st.text(min_size=5, max_size=50))
    base_content = draw(st.text(min_size=20, max_size=200))

    # Inject keyword to ensure priority classification
    title = f"{base_title} {keyword} news"
    content = f"{base_content} This article discusses {keyword} in detail."

    return NewsArticle(
        id=draw(st.text(min_size=1, max_size=50)),
        title=title,
        content=content,
        url=draw(
            st.text(min_size=10, max_size=100).map(
                lambda x: f"https://finance.yahoo.com/{x.replace(' ', '-')}"
            )
        ),
        published_at=draw(
            st.datetimes(
                min_value=datetime(2020, 1, 1), max_value=datetime(2024, 12, 31)
            )
        ),
        source="Yahoo Finance",
        category="markets",
        raw_metadata={},
    )


@st.composite
def mixed_priority_articles_strategy(draw):
    """Generate a list of articles with mixed priorities."""
    num_high = draw(st.integers(min_value=1, max_value=3))
    num_medium = draw(st.integers(min_value=1, max_value=3))
    num_low = draw(st.integers(min_value=1, max_value=3))

    articles = []

    # Generate high priority articles
    for _ in range(num_high):
        articles.append(draw(article_with_priority_strategy(priority_level="high")))

    # Generate medium priority articles
    for _ in range(num_medium):
        articles.append(draw(article_with_priority_strategy(priority_level="medium")))

    # Generate low priority articles
    for _ in range(num_low):
        articles.append(draw(article_with_priority_strategy(priority_level="low")))

    # Shuffle using hypothesis's permutation to avoid random module
    return draw(st.permutations(articles))


@given(articles=mixed_priority_articles_strategy())
@settings(max_examples=100, deadline=5000)
def test_high_impact_articles_processed_first(articles):
    """
    **Feature: news-market-predictor, Property 22: Resource prioritization**

    Property: For any low-resource scenario, high-impact news articles should
    be processed before lower-impact articles.

    **Validates: Requirements 5.5**

    This test verifies that when system resources are constrained, the
    resource prioritizer orders articles so that high-impact articles
    are processed first.
    """
    # Create resource prioritizer with constraints
    constraints = ResourceConstraints(
        max_memory_mb=512, max_cpu_percent=80.0, max_concurrent_tasks=5
    )
    prioritizer = ResourcePrioritizer(constraints)

    # Get processing order
    ordered_articles = prioritizer.get_processing_order(articles)

    # Verify that articles are ordered by priority
    # Extract priorities for each article
    def get_article_priority(article) -> Priority:
        content = article.content.lower()
        title = article.title.lower()
        text = f"{title} {content}"

        high_impact_keywords = [
            "earnings",
            "acquisition",
            "merger",
            "bankruptcy",
            "lawsuit",
            "fda approval",
            "clinical trial",
            "breakthrough",
            "partnership",
        ]

        medium_impact_keywords = [
            "revenue",
            "profit",
            "guidance",
            "upgrade",
            "downgrade",
            "analyst",
            "rating",
            "target price",
        ]

        if any(keyword in text for keyword in high_impact_keywords):
            return Priority.HIGH
        elif any(keyword in text for keyword in medium_impact_keywords):
            return Priority.MEDIUM
        else:
            return Priority.LOW

    # Get priorities for ordered articles
    ordered_priorities = [get_article_priority(article) for article in ordered_articles]

    # Verify that high priority articles come before medium and low
    # Find indices of first occurrence of each priority
    first_high = next(
        (i for i, p in enumerate(ordered_priorities) if p == Priority.HIGH), None
    )
    first_medium = next(
        (i for i, p in enumerate(ordered_priorities) if p == Priority.MEDIUM), None
    )
    first_low = next(
        (i for i, p in enumerate(ordered_priorities) if p == Priority.LOW), None
    )

    # If high priority articles exist, they should come first
    if first_high is not None:
        if first_medium is not None:
            assert (
                first_high < first_medium
            ), "High priority articles should come before medium priority"
        if first_low is not None:
            assert (
                first_high < first_low
            ), "High priority articles should come before low priority"

    # If medium priority articles exist, they should come before low
    if first_medium is not None and first_low is not None:
        assert (
            first_medium < first_low
        ), "Medium priority articles should come before low priority"

    # Verify that the list is sorted by priority value (1=HIGH, 2=MEDIUM, 3=LOW)
    priority_values = [p.value for p in ordered_priorities]
    assert priority_values == sorted(
        priority_values
    ), "Articles should be sorted by priority (HIGH=1, MEDIUM=2, LOW=3)"


@given(
    num_high=st.integers(min_value=1, max_value=5),
    num_low=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=50, deadline=5000)
def test_resource_prioritization_consistency(num_high, num_low):
    """
    **Feature: news-market-predictor, Property 22: Resource prioritization**

    Property: For any set of articles, the prioritization should be consistent
    across multiple calls with the same input.

    **Validates: Requirements 5.5**

    This test verifies that resource prioritization is deterministic.
    """
    # Create articles with known priorities
    high_articles = [
        NewsArticle(
            id=f"high_{i}",
            title=f"Breaking: Major earnings announcement {i}",
            content=f"Company reports earnings for quarter {i}",
            url=f"https://finance.yahoo.com/high_{i}",
            published_at=datetime(2024, 1, 1),
            source="Yahoo Finance",
            category="markets",
            raw_metadata={},
        )
        for i in range(num_high)
    ]

    low_articles = [
        NewsArticle(
            id=f"low_{i}",
            title=f"Market update {i}",
            content=f"General market news and updates {i}",
            url=f"https://finance.yahoo.com/low_{i}",
            published_at=datetime(2024, 1, 1),
            source="Yahoo Finance",
            category="markets",
            raw_metadata={},
        )
        for i in range(num_low)
    ]

    articles = high_articles + low_articles

    # Create resource prioritizer
    constraints = ResourceConstraints(
        max_memory_mb=512, max_cpu_percent=80.0, max_concurrent_tasks=5
    )
    prioritizer = ResourcePrioritizer(constraints)

    # Get processing order multiple times
    order1 = prioritizer.get_processing_order(articles.copy())
    order2 = prioritizer.get_processing_order(articles.copy())
    order3 = prioritizer.get_processing_order(articles.copy())

    # Extract IDs for comparison
    ids1 = [article.id for article in order1]
    ids2 = [article.id for article in order2]
    ids3 = [article.id for article in order3]

    # All orderings should be identical
    assert ids1 == ids2 == ids3, "Prioritization should be consistent across calls"

    # Verify high priority articles come first
    high_ids = {f"high_{i}" for i in range(num_high)}
    low_ids = {f"low_{i}" for i in range(num_low)}

    # Find last high priority article index
    last_high_index = max(
        (i for i, article_id in enumerate(ids1) if article_id in high_ids), default=-1
    )

    # Find first low priority article index
    first_low_index = next(
        (i for i, article_id in enumerate(ids1) if article_id in low_ids), len(ids1)
    )

    # All high priority should come before all low priority
    assert (
        last_high_index < first_low_index
    ), "All high priority articles should come before low priority articles"


@given(
    max_concurrent_tasks=st.integers(min_value=2, max_value=10),
    num_articles=st.integers(min_value=5, max_value=15),
)
@settings(max_examples=50, deadline=5000)
def test_resource_constraints_respected(max_concurrent_tasks, num_articles):
    """
    **Feature: news-market-predictor, Property 22: Resource prioritization**

    Property: For any resource constraint configuration, the prioritizer should
    respect the maximum concurrent task limits for each priority level.

    **Validates: Requirements 5.5**

    This test verifies that resource constraints are properly enforced per priority.
    """
    # Ensure we have more articles than concurrent task limit
    assume(num_articles > max_concurrent_tasks)

    # Create resource prioritizer with specific constraints
    constraints = ResourceConstraints(
        max_memory_mb=512,
        max_cpu_percent=80.0,
        max_concurrent_tasks=max_concurrent_tasks,
    )
    prioritizer = ResourcePrioritizer(constraints)

    # Test each priority level separately
    for priority in [Priority.HIGH, Priority.MEDIUM, Priority.LOW]:
        # Reset for each priority
        prioritizer.active_tasks = {
            Priority.HIGH: 0,
            Priority.MEDIUM: 0,
            Priority.LOW: 0,
        }

        tasks_started = 0
        # Try to start multiple tasks of this priority
        for _ in range(max_concurrent_tasks + 5):
            if prioritizer.start_task(priority):
                tasks_started += 1
            else:
                break

        # Verify that we didn't exceed reasonable limits for this priority
        total_active = sum(prioritizer.active_tasks.values())
        assert (
            total_active <= max_concurrent_tasks
        ), f"Total active tasks ({total_active}) should not exceed limit ({max_concurrent_tasks})"

        # Clean up
        for _ in range(tasks_started):
            prioritizer.finish_task(priority)


@given(articles=mixed_priority_articles_strategy())
@settings(max_examples=100, deadline=5000)
def test_error_handling_manager_prioritizes_correctly(articles):
    """
    **Feature: news-market-predictor, Property 22: Resource prioritization**

    Property: For any set of articles processed through ErrorHandlingManager,
    the articles should be ordered by priority before processing.

    **Validates: Requirements 5.5**

    This test verifies that the ErrorHandlingManager correctly orders articles
    by priority using the resource prioritizer.
    """
    # Create error handling manager with resource constraints
    constraints = ResourceConstraints(
        max_memory_mb=512, max_cpu_percent=80.0, max_concurrent_tasks=5
    )
    manager = ErrorHandlingManager(resource_constraints=constraints)

    # Get the ordered articles from the resource prioritizer
    ordered_articles = manager.resource_prioritizer.get_processing_order(articles)

    # Verify ordering respects priority
    def get_article_priority(article) -> Priority:
        content = article.content.lower()
        title = article.title.lower()
        text = f"{title} {content}"

        high_impact_keywords = [
            "earnings",
            "acquisition",
            "merger",
            "bankruptcy",
            "lawsuit",
            "fda approval",
            "clinical trial",
            "breakthrough",
            "partnership",
        ]

        medium_impact_keywords = [
            "revenue",
            "profit",
            "guidance",
            "upgrade",
            "downgrade",
            "analyst",
            "rating",
            "target price",
        ]

        if any(keyword in text for keyword in high_impact_keywords):
            return Priority.HIGH
        elif any(keyword in text for keyword in medium_impact_keywords):
            return Priority.MEDIUM
        else:
            return Priority.LOW

    # Get priorities for ordered articles
    ordered_priorities = [get_article_priority(article) for article in ordered_articles]

    # Verify that priorities are in order (HIGH=1, MEDIUM=2, LOW=3)
    priority_values = [p.value for p in ordered_priorities]
    assert priority_values == sorted(
        priority_values
    ), "Articles should be ordered by priority (HIGH=1, MEDIUM=2, LOW=3)"
    assert priority_values == sorted(
        priority_values
    ), "Articles should be processed in priority order (HIGH=1, MEDIUM=2, LOW=3)"


@given(
    priority=st.sampled_from([Priority.HIGH, Priority.MEDIUM, Priority.LOW]),
    max_concurrent=st.integers(min_value=2, max_value=8),
)
@settings(max_examples=50, deadline=5000)
def test_priority_based_resource_allocation(priority, max_concurrent):
    """
    **Feature: news-market-predictor, Property 22: Resource prioritization**

    Property: For any priority level, the resource allocator should reserve
    appropriate resources based on priority (HIGH gets more, LOW gets less).

    **Validates: Requirements 5.5**

    This test verifies that resource allocation favors high-priority tasks.
    """
    constraints = ResourceConstraints(
        max_memory_mb=512,
        max_cpu_percent=80.0,
        max_concurrent_tasks=max_concurrent,
    )
    prioritizer = ResourcePrioritizer(constraints)

    # Try to start tasks of the given priority
    tasks_started = 0
    for _ in range(max_concurrent + 5):  # Try more than the limit
        if prioritizer.start_task(priority):
            tasks_started += 1
        else:
            break

    # Verify resource allocation based on priority
    if priority == Priority.HIGH:
        # High priority should be able to use most/all resources
        assert (
            tasks_started >= max_concurrent // 2
        ), "High priority should get at least half of resources"
    elif priority == Priority.MEDIUM:
        # Medium priority should get moderate resources
        assert (
            tasks_started <= max_concurrent
        ), "Medium priority should not exceed total resources"
    else:  # LOW
        # Low priority should get limited resources
        assert (
            tasks_started <= max_concurrent // 2
        ), "Low priority should get at most half of resources"

    # Clean up
    for _ in range(tasks_started):
        prioritizer.finish_task(priority)


@given(articles=mixed_priority_articles_strategy())
@settings(max_examples=50, deadline=5000)
def test_low_resource_scenario_prioritizes_high_impact(articles):
    """
    **Feature: news-market-predictor, Property 22: Resource prioritization**

    Property: For any low-resource scenario (limited concurrent tasks),
    high-impact articles should be processed before lower-impact articles.

    **Validates: Requirements 5.5**

    This test specifically verifies the requirement that in low-resource
    scenarios, high-impact news is prioritized.
    """
    # Create a low-resource scenario
    constraints = ResourceConstraints(
        max_memory_mb=256,  # Low memory
        max_cpu_percent=50.0,  # Low CPU
        max_concurrent_tasks=2,  # Very limited concurrency
    )
    prioritizer = ResourcePrioritizer(constraints)

    # Get processing order
    ordered_articles = prioritizer.get_processing_order(articles)

    # Classify articles by priority
    def get_article_priority(article) -> Priority:
        content = article.content.lower()
        title = article.title.lower()
        text = f"{title} {content}"

        high_impact_keywords = [
            "earnings",
            "acquisition",
            "merger",
            "bankruptcy",
            "lawsuit",
            "fda approval",
            "clinical trial",
            "breakthrough",
            "partnership",
        ]

        medium_impact_keywords = [
            "revenue",
            "profit",
            "guidance",
            "upgrade",
            "downgrade",
            "analyst",
            "rating",
            "target price",
        ]

        if any(keyword in text for keyword in high_impact_keywords):
            return Priority.HIGH
        elif any(keyword in text for keyword in medium_impact_keywords):
            return Priority.MEDIUM
        else:
            return Priority.LOW

    # Get priorities for ordered articles
    ordered_priorities = [get_article_priority(article) for article in ordered_articles]

    # Count articles by priority
    high_count = sum(1 for p in ordered_priorities if p == Priority.HIGH)
    medium_count = sum(1 for p in ordered_priorities if p == Priority.MEDIUM)
    low_count = sum(1 for p in ordered_priorities if p == Priority.LOW)

    # If there are high priority articles, verify they come first
    if high_count > 0:
        # All high priority articles should be in the first high_count positions
        first_n_priorities = ordered_priorities[:high_count]
        assert all(
            p == Priority.HIGH for p in first_n_priorities
        ), "In low-resource scenario, all high-impact articles should be processed first"

    # If there are medium priority articles, they should come after high but before low
    if medium_count > 0 and high_count > 0:
        medium_start = high_count
        medium_end = high_count + medium_count
        medium_priorities = ordered_priorities[medium_start:medium_end]
        assert all(
            p == Priority.MEDIUM for p in medium_priorities
        ), "Medium priority articles should come after high priority"

    # Low priority articles should come last
    if low_count > 0 and (high_count > 0 or medium_count > 0):
        low_start = high_count + medium_count
        low_priorities = ordered_priorities[low_start:]
        assert all(
            p == Priority.LOW for p in low_priorities
        ), "Low priority articles should come last"


@given(
    articles=mixed_priority_articles_strategy(),
    max_concurrent=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=50, deadline=5000)
def test_resource_prioritization_under_varying_constraints(articles, max_concurrent):
    """
    **Feature: news-market-predictor, Property 22: Resource prioritization**

    Property: For any resource constraint level, the prioritization order
    should remain consistent (high before medium before low).

    **Validates: Requirements 5.5**

    This test verifies that regardless of resource constraints, the
    prioritization order is maintained.
    """
    # Create resource prioritizer with varying constraints
    constraints = ResourceConstraints(
        max_memory_mb=512,
        max_cpu_percent=80.0,
        max_concurrent_tasks=max_concurrent,
    )
    prioritizer = ResourcePrioritizer(constraints)

    # Get processing order
    ordered_articles = prioritizer.get_processing_order(articles)

    # Verify order is maintained regardless of constraint level
    def get_article_priority(article) -> Priority:
        content = article.content.lower()
        title = article.title.lower()
        text = f"{title} {content}"

        high_impact_keywords = [
            "earnings",
            "acquisition",
            "merger",
            "bankruptcy",
            "lawsuit",
            "fda approval",
            "clinical trial",
            "breakthrough",
            "partnership",
        ]

        medium_impact_keywords = [
            "revenue",
            "profit",
            "guidance",
            "upgrade",
            "downgrade",
            "analyst",
            "rating",
            "target price",
        ]

        if any(keyword in text for keyword in high_impact_keywords):
            return Priority.HIGH
        elif any(keyword in text for keyword in medium_impact_keywords):
            return Priority.MEDIUM
        else:
            return Priority.LOW

    # Get priorities for ordered articles
    ordered_priorities = [get_article_priority(article) for article in ordered_articles]

    # Verify sorted order
    priority_values = [p.value for p in ordered_priorities]
    assert priority_values == sorted(
        priority_values
    ), f"Articles should be sorted by priority regardless of max_concurrent={max_concurrent}"
