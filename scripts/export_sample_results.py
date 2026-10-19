"""
SentinelTrack — scripts/export_sample_results.py

Exports sample outputs for hackathon submission:
  - Database statistics (unique identities, sessions, events)
  - Recent event log entries
  - Evidence image listing
  - Run summary JSON

Usage:
    python scripts/export_sample_results.py
    python scripts/export_sample_results.py --output docs/sample_output.md
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/sample_output.md")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    try:
        from src.config import load_config
        from src.database import init_db, get_session
        from src.repositories import FaceRepository, SessionRepository, EventRepository
        from src.models import EventType

        cfg = load_config("config.json")
        init_db(cfg)

        lines = []
        lines.append("# SentinelTrack — Sample Output Report")
        lines.append(f"\nGenerated: {datetime.now(timezone.utc).isoformat()}\n")

        with get_session() as db:
            unique = FaceRepository.count_unique_identities(db)
            sessions = SessionRepository.count_sessions(db)
            entries = EventRepository.count_events(db, EventType.ENTRY)
            exits = EventRepository.count_events(db, EventType.EXIT)
            registrations = EventRepository.count_events(db, EventType.REGISTRATION)

        lines.append("## Database Statistics\n")
        lines.append(f"| Metric | Count |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Unique Identities | {unique} |")
        lines.append(f"| Total Sessions | {sessions} |")
        lines.append(f"| Entry Events | {entries} |")
        lines.append(f"| Exit Events | {exits} |")
        lines.append(f"| Registration Events | {registrations} |")

        # Evidence images
        lines.append("\n## Evidence Images\n")
        entries_dir = Path(cfg.evidence.entries_dir)
        exits_dir = Path(cfg.evidence.exits_dir)

        entry_images = list(entries_dir.rglob("*.jpg")) if entries_dir.exists() else []
        exit_images = list(exits_dir.rglob("*.jpg")) if exits_dir.exists() else []

        lines.append(f"- Entry images saved: {len(entry_images)}")
        lines.append(f"- Exit images saved: {len(exit_images)}")

        if entry_images[:3]:
            lines.append("\nSample entry images:")
            for img in entry_images[:3]:
                lines.append(f"  - `{img.relative_to(Path.cwd()) if img.is_absolute() else img}`")

        # Last run summary
        summary_path = Path("logs/last_run_summary.json")
        if summary_path.exists():
            lines.append("\n## Last Run Summary\n")
            lines.append("```json")
            lines.append(summary_path.read_text())
            lines.append("```")

        # Recent events.log
        log_path = Path(cfg.logging.log_file)
        if log_path.exists():
            lines.append("\n## Recent events.log (last 20 lines)\n")
            lines.append("```")
            log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            lines.extend(log_lines[-20:])
            lines.append("```")

        output = "\n".join(lines)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Sample output written to: {args.output}")
        print(output[:1000])

    except Exception as exc:
        print(f"Export failed: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
