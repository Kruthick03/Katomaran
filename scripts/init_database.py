"""
SentinelTrack — scripts/init_database.py

Database initialization script:
  1. Creates the sentineltrack database and user (if they don't exist).
  2. Runs Alembic migrations.
  3. Verifies schema by querying all expected tables.

Usage:
    # As a privileged user (e.g., postgres):
    python scripts/init_database.py

    # With custom admin credentials:
    PGADMIN_USER=postgres PGADMIN_PASSWORD=secret python scripts/init_database.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_alembic_upgrade():
    """Run alembic upgrade head to apply all migrations."""
    print("Running Alembic migrations (alembic upgrade head)...")
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    if result.returncode != 0:
        print(f"Alembic migration FAILED:\n{result.stderr}")
        return False
    print(result.stdout)
    print("Migrations applied successfully.")
    return True


def verify_schema():
    """Verify that all expected tables exist."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from src.config import load_config
        from src.database import init_db, get_engine
        from sqlalchemy import inspect

        cfg = load_config("config.json")
        init_db(cfg)
        engine = get_engine()

        with engine.connect() as conn:
            inspector = inspect(conn)
            tables = inspector.get_table_names()

        expected = {"cameras", "faces", "visit_sessions", "events"}
        missing = expected - set(tables)

        if missing:
            print(f"SCHEMA ERROR: Missing tables: {missing}")
            return False

        print(f"Schema verified. Tables found: {sorted(tables)}")
        return True

    except Exception as exc:
        print(f"Schema verification failed: {exc}")
        return False


def create_db_user_if_possible():
    """
    Attempt to create the database and user using psql.
    Only works if psql is on PATH and the admin user has access.
    Gracefully skips if not available.
    """
    psql_path = r"C:\Program Files\PostgreSQL\18\bin\psql.exe"
    if not Path(psql_path).exists():
        print("psql not found on expected path. Skipping automated user creation.")
        print("Please create the database and user manually.")
        return False

    from dotenv import load_dotenv
    load_dotenv()

    db_name = os.getenv("SENTINEL_DB_NAME", "sentineltrack")
    db_user = os.getenv("SENTINEL_DB_USER", "sentineluser")
    db_pass = os.getenv("SENTINEL_DB_PASSWORD", "")
    admin_user = os.getenv("PGADMIN_USER", "postgres")
    admin_pass = os.getenv("PGADMIN_PASSWORD", "")

    sql_commands = [
        f"DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{db_user}') THEN CREATE USER {db_user} WITH PASSWORD '{db_pass}'; END IF; END $$;",
        f"SELECT 'CREATE DATABASE {db_name} OWNER {db_user}' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '{db_name}') \\gexec",
        f"GRANT ALL PRIVILEGES ON DATABASE {db_name} TO {db_user};",
    ]

    env = os.environ.copy()
    env["PGPASSWORD"] = admin_pass

    for sql in sql_commands:
        result = subprocess.run(
            [psql_path, "-U", admin_user, "-c", sql],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0 and "already exists" not in result.stderr:
            print(f"SQL warning: {result.stderr.strip()}")

    print(f"Database '{db_name}' and user '{db_user}' ready.")
    return True


def main():
    print("=" * 60)
    print("SentinelTrack — Database Initialization")
    print("=" * 60)

    print("\n[1] Creating database and user...")
    create_db_user_if_possible()

    print("\n[2] Running migrations...")
    if not run_alembic_upgrade():
        print("FAILED: Could not run migrations. Check .env and config.json.")
        sys.exit(1)

    print("\n[3] Verifying schema...")
    if not verify_schema():
        print("FAILED: Schema verification failed.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Database initialization complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
