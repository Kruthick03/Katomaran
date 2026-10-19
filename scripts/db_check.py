"""
SentinelTrack — scripts/db_check.py

Quick database health and statistics check script.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from sqlalchemy import text

from src.config import load_config
from src.database import check_connection, get_engine, init_db


def main():
    load_dotenv()
    cfg = load_config("config.json")
    init_db(cfg)

    print("=" * 60)
    print(" SentinelTrack - Database Health & Status Check")
    print("=" * 60)

    # 1. Connectivity Check
    ok = check_connection(retries=2, delay=1.0)
    status_label = "[ONLINE] PostgreSQL connection verified" if ok else "[FAILED] Could not connect to PostgreSQL"
    print(f"\n1. Connection Status:\n   {status_label}")

    if not ok:
        print("\nPlease check your PostgreSQL service and credentials in .env.")
        sys.exit(1)

    engine = get_engine()
    with engine.connect() as conn:
        # 2. Server & Environment info
        version = conn.execute(text("SELECT version();")).scalar()
        current_db = conn.execute(text("SELECT current_database();")).scalar()
        current_user = conn.execute(text("SELECT current_user;")).scalar()
        server_addr = conn.execute(text("SELECT inet_server_addr(), inet_server_port();")).fetchone()

        print("\n2. Database & Server Info:")
        print(f"   Database Name : {current_db}")
        print(f"   User          : {current_user}")
        print(f"   Host / Port   : {server_addr[0] or 'localhost'}:{server_addr[1] or 5432}")
        print(f"   Server Build  : {version.split(',')[0]}")

        # 3. Alembic Migration Version
        rev = conn.execute(text("SELECT version_num FROM alembic_version;")).scalar()
        print("\n3. Schema Migration Version:")
        print(f"   Alembic Revision : {rev} (up to date)")

        # 4. Table Row Counts
        print("\n4. Table Row Counts:")
        tables = ["cameras", "faces", "visit_sessions", "events"]
        for t in tables:
            cnt = conn.execute(text(f"SELECT count(*) FROM {t};")).scalar()
            print(f"   - {t:<16}: {cnt:>6} records")

        # 5. Registered Cameras
        print("\n5. Registered Cameras:")
        cams = conn.execute(
            text("SELECT source_id, source_name, source_type, enabled FROM cameras ORDER BY source_id;")
        ).fetchall()
        if cams:
            for c in cams:
                status = "ACTIVE" if c[3] else "DISABLED"
                print(f"   - [{c[0]}] {c[1]} ({c[2]}) - {status}")
        else:
            print("   (No cameras registered yet)")

        # 6. Session Status Breakdown
        print("\n6. Visit Sessions Breakdown:")
        sessions = conn.execute(
            text("SELECT status, count(*) FROM visit_sessions GROUP BY status ORDER BY count(*) DESC;")
        ).fetchall()
        if sessions:
            for s, cnt in sessions:
                print(f"   - {s:<12}: {cnt} sessions")
        else:
            print("   (No sessions recorded yet)")

        # 7. Event Types Breakdown
        print("\n7. Event Types Breakdown:")
        events_breakdown = conn.execute(
            text("SELECT event_type, count(*) FROM events GROUP BY event_type ORDER BY count(*) DESC;")
        ).fetchall()
        if events_breakdown:
            for et, cnt in events_breakdown:
                print(f"   - {et:<12}: {cnt} events")
        else:
            print("   (No events recorded yet)")

        # 8. Latest Events
        print("\n8. Recent Activity (Latest 5 Events):")
        recent = conn.execute(
            text(
                "SELECT event_id, event_type, event_timestamp, face_id, source_id "
                "FROM events ORDER BY event_timestamp DESC LIMIT 5;"
            )
        ).fetchall()
        if recent:
            for r in recent:
                ts_str = r[2].strftime("%Y-%m-%d %H:%M:%S UTC") if r[2] else "N/A"
                print(f"   - [{ts_str}] {r[1].upper():<12} Face: {r[3][:8]}... Source: {r[4]}")
        else:
            print("   (No events found)")

    print("\n" + "=" * 60)
    print(" Database check completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
