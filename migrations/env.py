"""
Alembic migration environment script.
Reads database URL from environment variables (never from alembic.ini directly).
"""
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Add project root to path so models can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, build_database_url
from src.models import Base

# Alembic Config object
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url() -> str:
    """Build database URL from environment variables."""
    # Load .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    try:
        app_config = load_config("config.json")
        return build_database_url(app_config.database)
    except Exception:
        # Fallback: read directly from environment
        host = os.getenv("SENTINEL_DB_HOST", "localhost")
        port = os.getenv("SENTINEL_DB_PORT", "5432")
        name = os.getenv("SENTINEL_DB_NAME", "sentineltrack")
        user = os.getenv("SENTINEL_DB_USER", "sentineluser")
        password = os.getenv("SENTINEL_DB_PASSWORD", "")
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (generate SQL without connecting)."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (connect and execute)."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
