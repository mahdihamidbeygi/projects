"""
Main entry point for the News Market Predictor application.
"""

import sys
import argparse
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from .config import Config
from .logging_config import setup_logging, get_logger
from .exceptions import NewsMarketPredictorError
from .pipeline_manager import PipelineManager
from .fetcher.yahoo_finance_fetcher import YahooFinanceNewsFetcher
from .analyzer.content_processor import NewsContentProcessor
from .analyzer.sentiment_analyzer import VaderSentimentAnalyzer
from .analyzer.entity_extractor import FinancialEntityExtractor
from .predictor.market_predictor import BasicMarketPredictor


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the command-line argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="News Market Predictor - Analyze Yahoo Finance news and predict market impact",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run daily analysis for today
  python -m news_market_predictor.main

  # Run analysis for a specific date
  python -m news_market_predictor.main --date 2024-01-15

  # Run with custom configuration
  python -m news_market_predictor.main --log-level DEBUG --batch-size 20

  # Export results to JSON
  python -m news_market_predictor.main --output results.json --format json

  # Show system health status
  python -m news_market_predictor.main --health-check
        """,
    )

    # Analysis options
    parser.add_argument(
        "--date",
        type=str,
        help="Target date for analysis (YYYY-MM-DD format). Defaults to today.",
    )

    parser.add_argument(
        "--days-back",
        type=int,
        default=0,
        help="Number of days back from today to analyze (default: 0 for today only)",
    )

    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path for results (default: print to console)",
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "csv", "text"],
        default="text",
        help="Output format (default: text)",
    )

    # Configuration options
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--log-file", type=str, help="Log file path (default: news_predictor.log)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Number of articles to process in parallel (default: 10)",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        help="Maximum number of network retries (default: 3)",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        help="Minimum confidence threshold for predictions (default: 0.3)",
    )

    # Monitoring options
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Display system health status and exit",
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Display detailed statistics after analysis",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--version", action="version", version="News Market Predictor 0.1.0"
    )

    return parser


def apply_cli_config(config: Config, args: argparse.Namespace) -> Config:
    """
    Apply command-line arguments to configuration.

    Args:
        config: Base configuration
        args: Parsed command-line arguments

    Returns:
        Updated configuration
    """
    if args.log_level:
        config.log_level = args.log_level

    if args.log_file:
        config.log_file = args.log_file

    if args.batch_size:
        config.processing.batch_size = args.batch_size

    if args.max_retries:
        config.network.max_retries = args.max_retries

    if args.confidence_threshold:
        config.processing.min_confidence_threshold = args.confidence_threshold

    if args.verbose:
        config.log_level = "DEBUG"

    return config


def initialize_pipeline(config: Config) -> PipelineManager:
    """
    Initialize the complete analysis pipeline with all components.

    Args:
        config: Application configuration

    Returns:
        Configured PipelineManager instance
    """
    logger = get_logger(__name__)
    logger.info("Initializing pipeline components...")

    # Initialize components
    fetcher = YahooFinanceNewsFetcher(
        max_retries=config.network.max_retries,
        rate_limit_delay=config.network.rate_limit_delay,
        timeout=config.network.timeout,
    )

    processor = NewsContentProcessor()

    sentiment_analyzer = VaderSentimentAnalyzer()

    entity_extractor = FinancialEntityExtractor()

    predictor = BasicMarketPredictor()

    # Create pipeline manager
    pipeline = PipelineManager(
        fetcher=fetcher,
        processor=processor,
        sentiment_analyzer=sentiment_analyzer,
        entity_extractor=entity_extractor,
        predictor=predictor,
        storage=None,  # Storage will be added in task 12
        max_concurrent_tasks=config.processing.batch_size,
    )

    logger.info("Pipeline initialization complete")
    return pipeline


def format_results(results: Dict[str, Any], format_type: str) -> str:
    """
    Format analysis results for output.

    Args:
        results: Analysis results dictionary
        format_type: Output format (json, csv, text)

    Returns:
        Formatted results string
    """
    if format_type == "json":
        return json.dumps(results, indent=2, default=str)

    elif format_type == "csv":
        # Simple CSV format for predictions
        lines = ["Stock Symbol,Impact Direction,Confidence Level,Reasoning"]
        for pred in results.get("predictions", []):
            lines.append(
                f"{pred['stock_symbol']},{pred['impact_direction']},"
                f"{pred['confidence_level']:.2f},{pred['reasoning']}"
            )
        return "\n".join(lines)

    else:  # text format
        lines = []
        lines.append("=" * 80)
        lines.append("NEWS MARKET PREDICTOR - ANALYSIS RESULTS")
        lines.append("=" * 80)
        lines.append(f"Timestamp: {results.get('timestamp', 'N/A')}")
        lines.append(f"Status: {'SUCCESS' if results.get('success') else 'FAILED'}")
        lines.append(f"Articles Analyzed: {results.get('articles_count', 0)}")
        lines.append(f"Predictions Generated: {results.get('predictions_count', 0)}")
        lines.append("")

        if results.get("error"):
            lines.append(f"ERROR: {results['error']}")
            lines.append("")

        if results.get("predictions"):
            lines.append("TOP PREDICTIONS:")
            lines.append("-" * 80)
            for i, pred in enumerate(results["predictions"], 1):
                lines.append(f"\n{i}. {pred['stock_symbol']}")
                lines.append(f"   Impact: {pred['impact_direction'].upper()}")
                lines.append(f"   Confidence: {pred['confidence_level']:.1%}")
                lines.append(f"   Reasoning: {pred['reasoning']}")

        lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)


def display_health_status(pipeline: PipelineManager) -> None:
    """
    Display system health status.

    Args:
        pipeline: Pipeline manager instance
    """
    health = pipeline.get_health_status()

    print("\n" + "=" * 80)
    print("SYSTEM HEALTH STATUS")
    print("=" * 80)
    print(f"Status: {health['status'].upper()}")
    print(f"Article Success Rate: {health['article_success_rate']:.1%}")
    print(f"Prediction Success Rate: {health['prediction_success_rate']:.1%}")
    print(f"Storage Failures: {health['storage_failure_count']}")
    print(f"Rate Limit Hits: {health['rate_limit_hits']}")
    print("\nDetailed Statistics:")
    for key, value in health["statistics"].items():
        print(f"  {key}: {value}")
    print("=" * 80 + "\n")


def display_statistics(results: Dict[str, Any]) -> None:
    """
    Display detailed statistics from analysis results.

    Args:
        results: Analysis results dictionary
    """
    stats = results.get("statistics", {})

    print("\n" + "=" * 80)
    print("DETAILED STATISTICS")
    print("=" * 80)
    for key, value in stats.items():
        formatted_key = key.replace("_", " ").title()
        print(f"{formatted_key}: {value}")
    print("=" * 80 + "\n")


def parse_date(date_str: str) -> datetime:
    """
    Parse date string in YYYY-MM-DD format.

    Args:
        date_str: Date string

    Returns:
        Parsed datetime object

    Raises:
        ValueError: If date format is invalid
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"Invalid date format: {date_str}. Expected YYYY-MM-DD format."
        ) from exc


def main(argv: Optional[list] = None) -> int:
    """
    Main entry point for the News Market Predictor application.

    Args:
        argv: Command-line arguments (defaults to sys.argv)

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse command-line arguments
    parser = create_argument_parser()
    args = parser.parse_args(argv)

    try:
        # Load and configure
        config = Config.from_env()
        config = apply_cli_config(config, args)
        config.validate()

        # Setup logging
        setup_logging(config.log_level, config.log_file)
        logger = get_logger(__name__)

        logger.info("Starting News Market Predictor application")
        logger.debug(f"Configuration: {config}")
        logger.debug(f"Arguments: {args}")

        # Initialize pipeline
        pipeline = initialize_pipeline(config)

        # Handle health check
        if args.health_check:
            display_health_status(pipeline)
            return 0

        # Determine target date
        if args.date:
            target_date = parse_date(args.date)
            logger.info(f"Analyzing news for date: {target_date.strftime('%Y-%m-%d')}")
        else:
            target_date = datetime.now() - timedelta(days=args.days_back)
            logger.info(
                f"Analyzing news for: {target_date.strftime('%Y-%m-%d')} "
                f"({args.days_back} days back)"
            )

        # Run daily analysis
        logger.info("Running daily analysis...")
        results = pipeline.run_daily_analysis(target_date)

        # Format and output results
        formatted_output = format_results(results, args.format)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(formatted_output)
            logger.info(f"Results written to: {args.output}")
            print(f"\nResults saved to: {args.output}")
        else:
            print(formatted_output)

        # Display statistics if requested
        if args.stats:
            display_statistics(results)

        # Log completion
        if results.get("success"):
            logger.info("News Market Predictor application completed successfully")
            return 0
        else:
            logger.warning("News Market Predictor completed with errors")
            return 1

    except NewsMarketPredictorError as e:
        logger = get_logger(__name__)
        logger.error(f"Application error: {e}")
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1

    except ValueError as e:
        logger = get_logger(__name__)
        logger.error(f"Invalid input: {e}")
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1

    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\nUNEXPECTED ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
