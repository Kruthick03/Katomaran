"""
SentinelTrack — src/visit_session.py

Manages the lifecycle of visitor sessions via a state machine:
  NOT_PRESENT -> ACTIVE -> EXIT_PENDING -> EXITED

Guarantees:
  - Exactly one ENTRY event per session.
  - Exactly one EXIT event per session after absence confirmation.
  - Returning visitors create a new session without incrementing unique identity count.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
from sqlalchemy.orm import Session

from src.config import SessionConfig, get_config
from src.event_logger import EventLogger, log_entry, log_exit
from src.evidence_storage import EvidenceStorage
from src.models import EventType
from src.repositories import EventRepository, SessionRepository

logger = logging.getLogger("sentineltrack.session")


class VisitState(Enum):
    ACTIVE = auto()
    EXIT_PENDING = auto()


@dataclass
class ActiveSession:
    session_id: str
    face_id: str
    source_id: str
    state: VisitState
    last_seen_frame: int
    last_seen_time: datetime
    exit_pending_since_frame: int = -1
    last_crop: Optional[np.ndarray] = None


class VisitSessionManager:
    """
    Manages active visit sessions and coordinates entry/exit lifecycle.
    """

    def __init__(
        self,
        config: Optional[SessionConfig] = None,
        evidence_storage: Optional[EvidenceStorage] = None,
        event_logger: Optional[EventLogger] = None,
    ) -> None:
        if config is None:
            try:
                app_cfg = get_config()
                config = app_cfg.session
            except Exception:
                config = SessionConfig()

        self._config = config
        self._evidence_storage = evidence_storage
        self._event_logger = event_logger or EventLogger()

        self._active_sessions: Dict[str, ActiveSession] = {}  # face_id -> ActiveSession
        self._exit_confirmation_frames: int = getattr(
            config, "exit_confirmation_frames", 30
        )
        self._recent_events: List[Dict[str, Any]] = []

        # Stream/feed-specific visitor tracking (satisfies Katomaran unique visitor counting spec)
        self._feed_unique_face_ids: Set[str] = set()
        self._feed_total_sessions: int = 0

    def set_evidence_storage(self, storage: EvidenceStorage) -> None:
        """Set or update EvidenceStorage handler."""
        self._evidence_storage = storage

    def on_face_seen(
        self,
        db_session: Session,
        face_id: str,
        source_id: str,
        frame_number: int,
        timestamp: datetime,
        crop: Optional[np.ndarray] = None,
    ) -> Tuple[str, bool]:
        """
        Record that a face was observed in this frame.
        Creates a new VisitSession if face is not currently active.
        Returns (session_id, is_new_entry).
        """
        # Always register face as observed in this feed run
        self._feed_unique_face_ids.add(face_id)

        if face_id in self._active_sessions:
            sess = self._active_sessions[face_id]
            sess.last_seen_frame = frame_number
            sess.last_seen_time = timestamp
            if crop is not None:
                sess.last_crop = crop

            if sess.state == VisitState.EXIT_PENDING:
                logger.debug(
                    f"face_id={face_id[:8]} re-appeared at frame={frame_number}; resetting EXIT_PENDING to ACTIVE."
                )
                sess.state = VisitState.ACTIVE
                sess.exit_pending_since_frame = -1

            return sess.session_id, False

        # New visit session needed
        self._feed_total_sessions += 1

        # Check if an active session already exists in DB (e.g. from recovery or concurrent frame)
        existing = SessionRepository.get_active_session(db_session, face_id, source_id)
        if existing is not None:
            session_id = existing.session_id
            self._active_sessions[face_id] = ActiveSession(
                session_id=session_id,
                face_id=face_id,
                source_id=source_id,
                state=VisitState.ACTIVE,
                last_seen_frame=frame_number,
                last_seen_time=timestamp,
                last_crop=crop,
            )
            return session_id, False

        # Create new DB session record
        db_session_rec = SessionRepository.create_session(
            db_session,
            face_id=face_id,
            source_id=source_id,
            entry_timestamp=timestamp,
        )
        session_id = db_session_rec.session_id

        # Save entry evidence crop if available
        evidence_path = None
        if crop is not None and self._evidence_storage is not None:
            try:
                evidence_path = self._evidence_storage.save_evidence(
                    face_id=face_id,
                    session_id=session_id,
                    event_type="entry",
                    crop=crop,
                    timestamp=timestamp,
                    frame_number=frame_number,
                )
            except Exception as exc:
                logger.warning(f"Could not save entry evidence: {exc}")

        # Create ENTRY event in DB (idempotent)
        idempotency_key = f"{session_id}:entry"
        try:
            EventRepository.create_event(
                db_session,
                face_id=face_id,
                session_id=session_id,
                source_id=source_id,
                event_type=EventType.ENTRY,
                event_timestamp=timestamp,
                evidence_image_path=evidence_path,
                frame_number=frame_number,
                idempotency_key=idempotency_key,
            )
            log_entry(face_id, session_id, timestamp.isoformat(), source_id)
            self._recent_events.append({
                "type": "ENTRY",
                "face_id": face_id,
                "session_id": session_id,
                "timestamp": timestamp,
                "frame_number": frame_number,
            })
            if len(self._recent_events) > 10:
                self._recent_events.pop(0)
        except Exception as exc:
            logger.error(f"Failed to persist ENTRY event for session {session_id}: {exc}", exc_info=True)

        self._active_sessions[face_id] = ActiveSession(
            session_id=session_id,
            face_id=face_id,
            source_id=source_id,
            state=VisitState.ACTIVE,
            last_seen_frame=frame_number,
            last_seen_time=timestamp,
            last_crop=crop,
        )
        return session_id, True

    def on_faces_absent(
        self,
        db_session: Session,
        seen_face_ids: Set[str],
        frame_number: int,
        timestamp: datetime,
    ) -> Dict[str, Any]:
        """
        Advance state machine for identities not detected in the current frame.
        """
        absent_ids = [fid for fid in list(self._active_sessions.keys()) if fid not in seen_face_ids]
        exits_triggered = 0

        for face_id in absent_ids:
            sess = self._active_sessions.get(face_id)
            if sess is None:
                continue
            missed_frames = frame_number - sess.last_seen_frame

            if sess.state == VisitState.ACTIVE:
                # When threshold reached, transition to EXIT_PENDING
                if missed_frames >= self._exit_confirmation_frames:
                    sess.state = VisitState.EXIT_PENDING
                    sess.exit_pending_since_frame = frame_number
                    logger.debug(
                        f"face_id={face_id[:8]} entered EXIT_PENDING at frame={frame_number} "
                        f"(absent for {missed_frames} frames)"
                    )
                continue

            if sess.state == VisitState.EXIT_PENDING:
                # If still absent on the next absence cycle, confirm the exit
                # (ensures at least one frame in EXIT_PENDING before DB write)
                if frame_number <= sess.exit_pending_since_frame:
                    continue

                # Confirm exit
                try:
                    SessionRepository.confirm_exit(
                        db_session, sess.session_id, timestamp
                    )

                    # Save exit evidence crop if available
                    evidence_path = None
                    if sess.last_crop is not None and self._evidence_storage is not None:
                        try:
                            evidence_path = self._evidence_storage.save_evidence(
                                face_id=face_id,
                                session_id=sess.session_id,
                                event_type="exit",
                                crop=sess.last_crop,
                                timestamp=timestamp,
                                frame_number=frame_number,
                            )
                        except Exception as exc:
                            logger.warning(f"Could not save exit evidence: {exc}")

                    # Create EXIT event in DB
                    idempotency_key = f"{sess.session_id}:exit"
                    EventRepository.create_event(
                        db_session,
                        face_id=face_id,
                        session_id=sess.session_id,
                        source_id=sess.source_id,
                        event_type=EventType.EXIT,
                        event_timestamp=timestamp,
                        evidence_image_path=evidence_path,
                        frame_number=frame_number,
                        idempotency_key=idempotency_key,
                    )
                    log_exit(face_id, sess.session_id, timestamp.isoformat(), sess.source_id)
                    self._recent_events.append({
                        "type": "EXIT",
                        "face_id": face_id,
                        "session_id": sess.session_id,
                        "timestamp": timestamp,
                        "frame_number": frame_number,
                    })
                    if len(self._recent_events) > 10:
                        self._recent_events.pop(0)

                    del self._active_sessions[face_id]
                    exits_triggered += 1
                    logger.info(f"Confirmed exit for face_id={face_id[:8]} session={sess.session_id[:8]}")

                except Exception as exc:
                    logger.error(f"Failed to confirm exit for session {sess.session_id}: {exc}", exc_info=True)

        return {"exits_triggered": exits_triggered}

    def get_recent_events(self) -> List[Dict[str, Any]]:
        """Return list of recently confirmed entry and exit events."""
        return list(self._recent_events)

    def get_session_counts(self, db_session: Session) -> Dict[str, Any]:
        """Return unique identity count and session counts."""
        from src.repositories import FaceRepository
        unique = FaceRepository.count_unique_identities(db_session)
        total = SessionRepository.count_sessions(db_session)
        return {
            "unique_identities": unique,
            "total_sessions": total,
            "currently_active": len(self._active_sessions),
            "feed_unique_visitors": len(self._feed_unique_face_ids),
            "feed_total_sessions": self._feed_total_sessions,
        }

    @property
    def feed_unique_visitors_count(self) -> int:
        """Count of distinct unique identities observed during this video feed/run."""
        return len(self._feed_unique_face_ids)

    @property
    def feed_total_sessions_count(self) -> int:
        """Count of total visit sessions started during this video feed/run."""
        return self._feed_total_sessions

    @property
    def active_sessions_count(self) -> int:
        """Count of currently active (non-exited) visit sessions."""
        return len(self._active_sessions)
