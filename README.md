# News Market Predictor

A system for analyzing Yahoo Finance news articles and predicting their potential impact on stock market movements.

## Project Structure

```
news_market_predictor/
├── __init__.py                 # Package initialization
├── main.py                     # Main application entry point
├── models.py                   # Core data models
├── interfaces.py               # Abstract base classes and interfaces
├── config.py                   # Configuration management
├── logging_config.py           # Logging setup
├── exceptions.py               # Custom exceptions
├── fetcher/                    # News fetching components
│   └── __init__.py
├── analyzer/                   # News analysis components
│   └── __init__.py
├── predictor/                  # Market prediction components
│   └── __init__.py
└── storage/                    # Data storage components
    └── __init__.py

tests/                          # Test suite
├── __init__.py
├── test_models.py              # Tests for data models
└── test_config.py              # Tests for configuration

requirements.txt                # Python dependencies
setup.py                        # Package setup configuration
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Command-Line Interface

The News Market Predictor provides a comprehensive CLI for running daily analysis:

#### Basic Usage

Run daily analysis for today:
```bash
python -m news_market_predictor.main
```

Run analysis for a specific date:
```bash
python -m news_market_predictor.main --date 2024-01-15
```

Run analysis for yesterday:
```bash
python -m news_market_predictor.main --days-back 1
```

#### Output Options

Export results to JSON:
```bash
python -m news_market_predictor.main --output results.json --format json
```

Export results to CSV:
```bash
python -m news_market_predictor.main --output results.csv --format csv
```

Display results in text format (default):
```bash
python -m news_market_predictor.main --format text
```

#### Configuration Options

Run with custom configuration:
```bash
python -m news_market_predictor.main --log-level DEBUG --batch-size 20 --max-retries 5
```

Set confidence threshold:
```bash
python -m news_market_predictor.main --confidence-threshold 0.5
```

Enable verbose output:
```bash
python -m news_market_predictor.main --verbose
```

#### Monitoring Options

Check system health status:
```bash
python -m news_market_predictor.main --health-check
```

Display detailed statistics:
```bash
python -m news_market_predictor.main --stats
```

#### Help and Version

Show help message:
```bash
python -m news_market_predictor.main --help
```

Show version:
```bash
python -m news_market_predictor.main --version
```

### Console Script

After installation, you can also use the console script:
```bash
news-predictor --help
```

## Configuration

The application can be configured using environment variables:

- `MAX_RETRIES`: Maximum number of retry attempts (default: 3)
- `RETRY_DELAY`: Delay between retries in seconds (default: 1.0)
- `TIMEOUT`: Network timeout in seconds (default: 30)
- `BATCH_SIZE`: Processing batch size (default: 10)
- `LOG_LEVEL`: Logging level (default: INFO)
- `DATABASE_URL`: Database connection URL

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

## Architecture

The system follows a modular pipeline architecture with the following components:

1. **News Fetcher**: Retrieves articles from Yahoo Finance
2. **Content Processor**: Cleans and normalizes article text
3. **Sentiment Analyzer**: Analyzes emotional tone and market sentiment
4. **Entity Extractor**: Identifies stock symbols and financial metrics
5. **Market Predictor**: Generates impact predictions
6. **Results Aggregator**: Combines and weights predictions
7. **Data Storage**: Persists articles and predictions

Each component implements abstract interfaces defined in `interfaces.py`, ensuring modularity and testability.