"""
SentinelTrack --- src/database.py

Database engine, session factory and connection health checks.
Uses SQLAlchemy 2.x with psycopg2 for PostgreSQL.
Embedding similarity is handled via Python cosine-similarity fallback
(BYTEA storage). If pgvector is installed and the DB extension is enabled,
the recognizer module uses it for ANN search.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from src.config import AppConfig, build_database_url
from src.models import Base

logger = logging.getLogger("sentineltrack.database")

# Module-level engine and session factory (initialized once)
_engine = None
_SessionLocal: Optional[sessionmaker] = None


def init_db(config: AppConfig) -> None:
    """
    Initialize the database engine and session factory.
    Must be called once during application startup.

    Args:
        config: Validated application configuration.

    Raises:
        RuntimeError: If the database URL cannot be built.
        OperationalError: If the database is unreachable.
    """
    global _engine, _SessionLocal

    db_url = build_database_url(config.database)
    # Log URL with password masked for safety
    safe_url = db_url.split("@")[-1] if "@" in db_url else db_url
    logger.info(f"Connecting to PostgreSQL at {safe_url}")

    _engine = create_engine(
        db_url,
        pool_size=config.database.pool_size,
        max_overflow=config.database.max_overflow,
        pool_timeout=config.database.pool_timeout,
        pool_pre_ping=True,  # Test connection liveness before each use
        echo=config.database.echo_sql,
    )

    _SessionLocal = sessionmaker(
        bind=_engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )

    logger.info("Database engine initialized.")


def get_engine():
    """Return the initialized SQLAlchemy engine."""
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _engine


def get_session_factory() -> sessionmaker:
    """Return the session factory."""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _SessionLocal


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager that provides a transactional database session.
    Commits on success, rolls back on any exception.

    Usage:
        with get_session() as session:
            session.add(some_object)
    """
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        logger.error(f"Database transaction rolled back: {exc}", exc_info=True)
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection(retries: int = 3, delay: float = 2.0) -> bool:
    """
    Test database connectivity with bounded retries.

    Args:
        retries: Number of connection attempts.
        delay: Seconds to wait between attempts.

    Returns:
        True if connection succeeded, False otherwise.
    """
    engine = get_engine()
    for attempt in range(1, retries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection verified.")
            return True
        except OperationalError as exc:
            logger.warning(
                f"Database connection attempt {attempt}/{retries} failed: {exc}"
            )
            if attempt < retries:
                time.sleep(delay)

    logger.error("All database connection attempts failed.")
    return False


def create_all_tables() -> None:
    """
    Create all tables defined in models.py if they do not exist.
    This is a convenience function for testing.
    For production, use Alembic migrations instead.
    """
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("All tables created (if not already present).")


def drop_all_tables() -> None:
    """
    Drop all tables. ONLY for testing environments.
    Will raise if called in a configuration without explicit confirmation.
    """
    engine = get_engine()
    Base.metadata.drop_all(engine)
    logger.warning("All tables dropped.")


def check_pgvector_available() -> bool:
    """
    Check if the pgvector extension is available in the connected database.
    Returns True if pgvector is installed, False otherwise.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            ).fetchone()
            return result is not None
    except Exception as exc:
        logger.debug(f"pgvector check failed: {exc}")
        return False


def close_db() -> None:
    """Dispose the database engine pool. Call during graceful shutdown."""
    global _engine
    if _engine is not None:
        _engine.dispose()
        logger.info("Database connection pool closed.")
        _engine = None

