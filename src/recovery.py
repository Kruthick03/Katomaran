"""
SentinelTrack — src/recovery.py

Database consistency, integrity checks, and crash recovery utilities.
Recovers orphaned or incomplete sessions on startup.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from src.event_logger import EventLogger, log_exit
from src.models import Event, EventType, SessionStatus, VisitSession
from src.repositories import EventRepository, SessionRepository

logger = logging.getLogger("sentineltrack.recovery")


class RecoveryChecker:
    """
    Consistency checker and crash-recovery engine for SentinelTrack.
    """

    def __init__(
        self,
        incomplete_threshold_hours: float = 1.0,
        event_logger: Optional[EventLogger] = None,
    ) -> None:
        self._threshold = timedelta(hours=incomplete_threshold_hours)
        self._event_logger = event_logger or EventLogger()

    def check_incomplete_sessions(self, db_session: Session) -> List[Dict[str, Any]]:
        """Find ACTIVE or EXIT_PENDING sessions older than the threshold."""
        cutoff = datetime.now(timezone.utc) - self._threshold
        stmt = select(VisitSession).where(
            VisitSession.status.in_([SessionStatus.ACTIVE, SessionStatus.EXIT_PENDING]),
            VisitSession.created_at < cutoff,
        )
        stale = db_session.execute(stmt).scalars().all()
        results = []
        for s in stale:
            results.append({
                "session_id": s.session_id,
                "face_id": s.face_id,
                "source_id": s.source_id,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "last_seen_at": s.last_seen_at.isoformat() if s.last_seen_at else None,
            })
            logger.warning(f"Incomplete session found: {s.session_id} (face: {s.face_id[:8]})")
        return results

    def check_orphaned_events(self, db_session: Session) -> List[Dict[str, Any]]:
        """Find events referencing non-existent sessions."""
        stmt = (
            select(Event)
            .outerjoin(VisitSession, Event.session_id == VisitSession.session_id)
            .where(
                Event.session_id.isnot(None),
                VisitSession.session_id.is_(None),
            )
        )
        orphans = db_session.execute(stmt).scalars().all()
        results = []
        for e in orphans:
            results.append({
                "event_id": e.event_id,
                "session_id": e.session_id,
                "face_id": e.face_id,
                "event_type": e.event_type,
            })
            logger.warning(f"Orphaned event found: {e.event_id}")
        return results

    def check_missing_evidence(self, db_session: Session) -> List[Dict[str, Any]]:
        """Find events whose recorded evidence file does not exist on disk."""
        stmt = select(Event).where(Event.evidence_image_path.isnot(None))
        events = db_session.execute(stmt).scalars().all()
        missing = []
        for e in events:
            if e.evidence_image_path and not Path(e.evidence_image_path).exists():
                missing.append({
                    "event_id": e.event_id,
                    "evidence_image_path": e.evidence_image_path,
                })
                logger.warning(f"Missing evidence image: {e.evidence_image_path}")
        return missing

    def run_full_check(self, db_session: Session) -> Dict[str, Any]:
        """Run all integrity checks and return summary counts."""
        incomplete = self.check_incomplete_sessions(db_session)
        orphaned = self.check_orphaned_events(db_session)
        missing_ev = self.check_missing_evidence(db_session)

        return {
            "incomplete_sessions": len(incomplete),
            "orphaned_events": len(orphaned),
            "missing_evidence_files": len(missing_ev),
            "status": "clean" if not (incomplete or orphaned or missing_ev) else "issues_found",
        }

    def recover_incomplete_sessions(
        self,
        db_session: Session,
        source_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Mark stale ACTIVE sessions from a previous run as EXITED.
        Prevents restart from associating new visits with abandoned sessions.
        """
        now = datetime.now(timezone.utc)
        stmt = select(VisitSession).where(
            VisitSession.status.in_([SessionStatus.ACTIVE, SessionStatus.EXIT_PENDING])
        )
        if source_id:
            stmt = stmt.where(VisitSession.source_id == source_id)

        active_sessions = db_session.execute(stmt).scalars().all()
        recovered = 0

        for sess in active_sessions:
            try:
                SessionRepository.confirm_exit(db_session, sess.session_id, now)

                # Persist exit event if not already present
                idempotency_key = f"{sess.session_id}:exit"
                if not EventRepository.has_exit_event(db_session, sess.session_id):
                    EventRepository.create_event(
                        db_session,
                        face_id=sess.face_id,
                        session_id=sess.session_id,
                        source_id=sess.source_id,
                        event_type=EventType.EXIT,
                        event_timestamp=now,
                        frame_number=None,
                        idempotency_key=idempotency_key,
                        meta={"recovery": "startup_reconciliation"},
                    )
                    log_exit(sess.face_id, sess.session_id, now.isoformat(), sess.source_id)

                recovered += 1
                logger.info(f"Recovered stale session: {sess.session_id}")
            except Exception as exc:
                logger.error(f"Error recovering session {sess.session_id}: {exc}", exc_info=True)

        return {"recovered": recovered, "source_id": source_id}
