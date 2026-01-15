# Storage Module

This module provides comprehensive data persistence and storage capabilities for the News Market Predictor system.

## Features

- **Database Connection Management**: Support for SQLite (development) and PostgreSQL (production)
- **Schema Management**: Automated migrations and version control
- **Data Access Layer**: Clean interface for all database operations
- **Backup and Recovery**: Automated backups with compression and verification
- **Retention Management**: Automated data cleanup based on retention policies

## Quick Start

### 1. Initialize Database

```python
from news_market_predictor.storage import (
    DatabaseConnection,
    DatabaseConfig,
    SchemaManager
)

# Create database configuration
config = DatabaseConfig(
    db_type="sqlite",
    db_path="news_market_predictor.db"
)

# Initialize connection
db = DatabaseConnection(config)

# Run migrations
schema_manager = SchemaManager(db)
schema_manager.migrate_up()
```

### 2. Store and Retrieve Data

```python
from news_market_predictor.storage import DataAccessLayer
from news_market_predictor.models import NewsArticle
from datetime import datetime

# Create data access layer
dal = DataAccessLayer(db)

# Store an article
article = NewsArticle(
    id="article_1",
    title="Market Update",
    content="Stock prices rise...",
    url="https://finance.yahoo.com/news/...",
    published_at=datetime.now(),
    source="Yahoo Finance",
    category="Market News",
    raw_metadata={}
)

dal.store_article(article)

# Retrieve articles
articles = dal.retrieve_articles()
```

### 3. Create Backups

```python
from news_market_predictor.storage import BackupManager, BackupConfig

# Configure backup settings
backup_config = BackupConfig(
    backup_dir="backups",
    max_backups=7,
    compress_backups=True,
    verify_backups=True
)

# Create backup manager
backup_mgr = BackupManager(config.db_path, backup_config)

# Create backup
result = backup_mgr.create_backup(force=True)

# List backups
backups = backup_mgr.list_backups()

# Restore from backup
backup_mgr.restore_backup("backup_20260114_120000.db.gz")
```

## Using Environment Variables

You can configure the database using environment variables:

```bash
# SQLite
export DB_TYPE=sqlite
export DB_PATH=news_market_predictor.db

# PostgreSQL
export DB_TYPE=postgresql
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=news_predictor
export DB_USER=postgres
export DB_PASSWORD=your_password
export DB_POOL_SIZE=5
```

Then in your code:

```python
config = DatabaseConfig.from_env()
db = DatabaseConnection(config)
```

## Command-Line Interface

The module includes a CLI tool for database management:

```bash
# Initialize database
python -m news_market_predictor.storage.db_cli init

# Check migration status
python -m news_market_predictor.storage.db_cli status

# Run migrations
python -m news_market_predictor.storage.db_cli migrate --direction up

# Create backup
python -m news_market_predictor.storage.db_cli backup --compress

# List backups
python -m news_market_predictor.storage.db_cli list-backups

# Restore from backup
python -m news_market_predictor.storage.db_cli restore backup_20260114_120000.db.gz

# Show database statistics
python -m news_market_predictor.storage.db_cli stats
```

## Schema Migrations

The system uses a migration-based approach for schema management:

### Current Migrations

1. **Version 1**: Initial schema
   - Creates tables for articles, sentiment analysis, entities, predictions, outcomes, and accuracy metrics

2. **Version 2**: Add indexes
   - Adds indexes for improved query performance

### Adding New Migrations

To add a new migration, edit `schema_manager.py` and add a new `Migration` object to the `_get_migrations()` method:

```python
migrations.append(
    Migration(
        version=3,
        name="add_user_table",
        description="Add user management table",
        up_sql="""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );
        """,
        down_sql="""
            DROP TABLE IF EXISTS users;
        """
    )
)
```

## Data Access Layer API

The `DataAccessLayer` class provides methods for all database operations:

### Article Operations
- `store_article(article: NewsArticle) -> bool`
- `retrieve_articles(date_range: Optional[Tuple[datetime, datetime]]) -> List[NewsArticle]`
- `get_article_by_id(article_id: str) -> Optional[NewsArticle]`

### Sentiment Operations
- `store_sentiment(sentiment: SentimentAnalysis) -> bool`
- `get_sentiment_by_article(article_id: str) -> Optional[SentimentAnalysis]`

### Entity Operations
- `store_entities(entities: List[ExtractedEntity]) -> bool`
- `get_entities_by_article(article_id: str) -> List[ExtractedEntity]`

### Prediction Operations
- `store_prediction(prediction: MarketPrediction) -> bool`
- `retrieve_predictions(stock_symbol: Optional[str]) -> List[MarketPrediction]`

### Outcome Operations
- `store_outcome(outcome: MarketOutcome) -> bool`
- `get_outcomes_by_prediction(prediction_id: str) -> List[MarketOutcome]`

### Accuracy Operations
- `store_accuracy_metrics(accuracy: HistoricalAccuracy) -> bool`
- `calculate_historical_accuracy(stock_symbol: str, time_period_days: int) -> Optional[HistoricalAccuracy]`

### Maintenance Operations
- `cleanup_old_data(retention_days: int) -> bool`
- `get_database_stats() -> Dict[str, int]`

## Error Handling

All database operations include proper error handling:

```python
from news_market_predictor.exceptions import StorageError

try:
    dal.store_article(article)
except StorageError as e:
    logger.error(f"Failed to store article: {e}")
    # Handle error appropriately
```

## Best Practices

1. **Always use transactions** for operations that modify multiple tables
2. **Close connections** when done (or use context managers)
3. **Run migrations** before starting the application
4. **Create regular backups** in production environments
5. **Monitor database size** and run cleanup operations periodically
6. **Use connection pooling** for PostgreSQL in production

## PostgreSQL Setup

For production use with PostgreSQL:

1. Install psycopg2:
   ```bash
   pip install psycopg2-binary
   ```

2. Create database:
   ```sql
   CREATE DATABASE news_predictor;
   CREATE USER predictor_user WITH PASSWORD 'secure_password';
   GRANT ALL PRIVILEGES ON DATABASE news_predictor TO predictor_user;
   ```

3. Configure environment:
   ```bash
   export DB_TYPE=postgresql
   export DB_HOST=localhost
   export DB_PORT=5432
   export DB_NAME=news_predictor
   export DB_USER=predictor_user
   export DB_PASSWORD=secure_password
   ```

4. Run migrations:
   ```bash
   python -m news_market_predictor.storage.db_cli init --use-env
   ```

## Troubleshooting

### Migration Failures

If a migration fails, you can rollback:

```bash
python -m news_market_predictor.storage.db_cli migrate --direction down --version 1
```

### Backup Restoration

Always test backup restoration in a non-production environment first:

```python
# Create a test database
test_config = DatabaseConfig(db_type="sqlite", db_path="test_restore.db")
test_db = DatabaseConnection(test_config)

# Restore backup
backup_mgr = BackupManager("test_restore.db")
result = backup_mgr.restore_backup("backup_file.db.gz")

# Verify data
dal = DataAccessLayer(test_db)
stats = dal.get_database_stats()
print(f"Restored {stats['total_articles']} articles")
```

### Connection Issues

For PostgreSQL connection issues:

1. Check PostgreSQL is running: `pg_isready`
2. Verify credentials and permissions
3. Check firewall settings
4. Review PostgreSQL logs: `/var/log/postgresql/`

## Performance Considerations

- **Indexes**: The schema includes indexes on frequently queried columns
- **Connection Pooling**: PostgreSQL uses connection pooling for better performance
- **Batch Operations**: Use `execute_many()` for bulk inserts
- **Query Optimization**: Use appropriate WHERE clauses and LIMIT statements
- **Regular Maintenance**: Run VACUUM and ANALYZE on PostgreSQL periodically
