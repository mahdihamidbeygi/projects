"""
Main entry point for the News Market Predictor application.
"""

import sys
from typing import Optional

from .config import Config
from .logging_config import setup_logging, get_logger
from .exceptions import NewsMarketPredictorError


def main(config_path: Optional[str] = None) -> int:
    """
    Main entry point for the News Market Predictor application.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Load configuration
        config = Config.from_env()
        config.validate()

        # Setup logging
        setup_logging(config.log_level, config.log_file)
        logger = get_logger(__name__)

        logger.info("Starting News Market Predictor application")
        logger.info(f"Configuration: {config}")

        # TODO: Initialize and run the main pipeline
        # This will be implemented in subsequent tasks

        logger.info("News Market Predictor application completed successfully")
        return 0

    except NewsMarketPredictorError as e:
        logger = get_logger(__name__)
        logger.error(f"Application error: {e}")
        return 1

    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Unexpected error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
