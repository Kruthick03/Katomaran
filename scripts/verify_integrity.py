"""
SentinelTrack — scripts/verify_integrity.py

Database and filesystem integrity checker.
Finds inconsistencies between:
  - Database event records and actual evidence image files on disk.
  - Incomplete sessions (ACTIVE/EXIT_PENDING with no exit event).
  - Orphaned events without a session.

Usage:
    python scripts/verify_integrity.py
    python scripts/verify_integrity.py --fix    # Attempt to fix stale sessions
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify database and evidence integrity.")
    parser.add_argument("--fix", action="store_true",
                        help="Attempt to close stale sessions.")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from src.config import load_config
    from src.database import init_db, get_session
    from src.recovery import RecoveryChecker
    from src.models import Event, EventType
    from sqlalchemy import select, func

    cfg = load_config("config.json")
    init_db(cfg)

    checker = RecoveryChecker()

    with get_session() as db:
        results = checker.run_full_check(db)

        # Check PostgreSQL vs events.log consistency
        db_entries = db.scalar(select(func.count(Event.event_id)).where(Event.event_type == EventType.ENTRY)) or 0
        db_exits = db.scalar(select(func.count(Event.event_id)).where(Event.event_type == EventType.EXIT)) or 0
        db_total_events = db.scalar(select(func.count(Event.event_id))) or 0

        log_path = Path("logs/events.log")
        log_entries = 0
        log_exits = 0
        log_registrations = 0
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "[ENTRY]" in line:
                        log_entries += 1
                    elif "[EXIT]" in line:
                        log_exits += 1
                    elif "[REGISTRATION]" in line:
                        log_registrations += 1

        results["database_events"] = {
            "total": db_total_events,
            "entries": db_entries,
            "exits": db_exits,
        }
        results["event_log_records"] = {
            "entries": log_entries,
            "exits": log_exits,
            "registrations": log_registrations,
        }
        results["mismatches"] = {
            "missing_evidence_files": results.get("missing_evidence", 0),
            "orphaned_events": results.get("orphaned_events", 0),
            "incomplete_sessions": results.get("incomplete_sessions", 0),
        }
        results["status"] = "HEALTHY" if (
            results.get("missing_evidence", 0) == 0 and
            results.get("orphaned_events", 0) == 0 and
            results.get("incomplete_sessions", 0) == 0
        ) else "NEEDS_ATTENTION"

        print("\n=== SentinelTrack Integrity & Reconciliation Audit ===")
        print(json.dumps(results, indent=2))

        if args.fix:
            print("\nAttempting to recover stale sessions...")
            checker.recover_incomplete_sessions(db, cfg.source.source_id)
            print("Recovery complete.")


if __name__ == "__main__":
    main()
