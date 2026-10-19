"""
SentinelTrack --- src/repositories.py

Repository layer: all database read/write operations.
No SQL queries should appear outside this module.

Each repository method:
  - Accepts a SQLAlchemy Session as its first argument.
  - Is transactional (the caller manages session lifecycle).
  - Uses idempotency keys where required to make retries safe.
  - Raises specific exceptions on constraint violations rather than
    silently swallowing errors.

Repositories:
  FaceRepository   --- faces table
  SessionRepository --- visit_sessions table
  EventRepository  --- events table
  CameraRepository --- cameras table
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
from sqlalchemy import select, update, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.models import (
    Camera,
    Event,
    EventType,
    Face,
    RegistrationStatus,
    SessionStatus,
    SourceType,
    VisitSession,
)
from src.face_recognizer import (
    RegisteredFace,
    embedding_to_bytes,
    bytes_to_embedding,
    normalize_embedding,
)

logger = logging.getLogger("sentineltrack.repositories")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------- Camera repository ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class CameraRepository:
    """Operations on the cameras table."""

    @staticmethod
    def get_or_create(
        session: Session,
        source_id: str,
        source_type: str,
        source_name: str,
        config_meta: Optional[dict] = None,
    ) -> Camera:
        """
        Fetch existing camera record or create a new one.
        Never stores passwords in config_meta.
        """
        camera = session.get(Camera, source_id)
        if camera is None:
            camera = Camera(
                source_id=source_id,
                source_type=source_type,
                source_name=source_name,
                config_meta=config_meta,
                enabled=True,
            )
            session.add(camera)
            session.flush()
            logger.info(f"Camera registered: {source_id} ({source_type})")
        return camera


# --------- Face repository ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class FaceRepository:
    """Operations on the faces table."""

    @staticmethod
    def get_by_id(session: Session, face_id: str) -> Optional[Face]:
        """Fetch a face by its stable UUID."""
        return session.get(Face, face_id)

    @staticmethod
    def get_all_registered(session: Session) -> List[RegisteredFace]:
        """
        Load all confirmed registered faces with their embeddings.
        Used for similarity matching at runtime.
        Only loads faces with non-null embeddings.
        """
        stmt = select(Face).where(
            Face.embedding.isnot(None),
            Face.registration_status == RegistrationStatus.CONFIRMED,
        )
        faces = session.execute(stmt).scalars().all()

        result = []
        for face in faces:
            try:
                emb = bytes_to_embedding(face.embedding, face.embedding_dimension or 512)
                emb = normalize_embedding(emb)
                result.append(RegisteredFace(face_id=face.face_id, embedding=emb))
            except Exception as exc:
                logger.warning(f"Could not deserialize embedding for {face.face_id}: {exc}")

        return result

    @staticmethod
    def register_new(
        session: Session,
        embedding: np.ndarray,
        recognition_model: str,
        first_seen_at: Optional[datetime] = None,
    ) -> Face:
        """
        Create a new identity record. Assigns a fresh UUID as face_id.

        Args:
            session: Active database session.
            embedding: Normalized float32 embedding array.
            recognition_model: Name of the recognition model used.
            first_seen_at: Timestamp of first detection.

        Returns:
            Newly created Face record.
        """
        now = first_seen_at or _utcnow()
        face = Face(
            face_id=str(uuid.uuid4()),
            first_seen_at=now,
            last_seen_at=now,
            embedding=embedding_to_bytes(embedding),
            embedding_dimension=len(embedding),
            recognition_model=recognition_model,
            registration_status=RegistrationStatus.CONFIRMED,
        )
        session.add(face)
        session.flush()
        logger.info(f"New face registered: {face.face_id}")
        return face

    @staticmethod
    def update_last_seen(
        session: Session,
        face_id: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Update last_seen_at for a returning visitor."""
        ts = timestamp or _utcnow()
        session.execute(
            update(Face)
            .where(Face.face_id == face_id)
            .values(last_seen_at=ts, updated_at=ts)
            .execution_options(synchronize_session="fetch")
        )
        session.flush()

    @staticmethod
    def count_unique_identities(session: Session) -> int:
        """Count total distinct registered identities."""
        return session.execute(
            select(func.count()).select_from(Face).where(
                Face.registration_status == RegistrationStatus.CONFIRMED
            )
        ).scalar_one()


# --------- Session repository ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

class SessionRepository:
    """Operations on the visit_sessions table."""

    @staticmethod
    def get_by_id(session: Session, session_id: str) -> Optional[VisitSession]:
        return session.get(VisitSession, session_id)

    @staticmethod
    def get_active_session(
        session: Session,
        face_id: str,
        source_id: str,
    ) -> Optional[VisitSession]:
        """
        Find an active (non-exited) session for this face at this source.
        Returns None if no active session exists.
        """
        stmt = select(VisitSession).where(
            VisitSession.face_id == face_id,
            VisitSession.source_id == source_id,
            VisitSession.status.in_([SessionStatus.ACTIVE, SessionStatus.EXIT_PENDING]),
        ).order_by(VisitSession.created_at.desc())
        return session.execute(stmt).scalars().first()

    @staticmethod
    def create_session(
        session: Session,
        face_id: str,
        source_id: str,
        entry_timestamp: datetime,
    ) -> VisitSession:
        """Create a new visit session in ACTIVE state."""
        vs = VisitSession(
            session_id=str(uuid.uuid4()),
            face_id=face_id,
            source_id=source_id,
            entry_timestamp=entry_timestamp,
            last_seen_at=entry_timestamp,
            status=SessionStatus.ACTIVE,
        )
        session.add(vs)
        session.flush()
        logger.info(
            f"Visit session created: {vs.session_id} face={face_id}"
        )
        return vs

    @staticmethod
    def update_last_seen(
        session: Session,
        session_id: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Update last_seen_at on an active session."""
        ts = timestamp or _utcnow()
        session.execute(
            update(VisitSession)
            .where(VisitSession.session_id == session_id)
            .values(last_seen_at=ts, updated_at=ts)
            .execution_options(synchronize_session="fetch")
        )
        session.flush()

    @staticmethod
    def mark_exit_pending(session: Session, session_id: str) -> None:
        """Transition session to EXIT_PENDING state."""
        session.execute(
            update(VisitSession)
            .where(VisitSession.session_id == session_id)
            .values(status=SessionStatus.EXIT_PENDING, updated_at=_utcnow())
            .execution_options(synchronize_session="fetch")
        )
        session.flush()

    @staticmethod
    def confirm_exit(
        session: Session,
        session_id: str,
        exit_timestamp: datetime,
    ) -> None:
        """Confirm exit: set EXITED status and exit_timestamp."""
        vs = session.get(VisitSession, session_id)
        ts = exit_timestamp
        if vs and vs.entry_timestamp and ts < vs.entry_timestamp:
            ts = vs.entry_timestamp

        session.execute(
            update(VisitSession)
            .where(VisitSession.session_id == session_id)
            .values(
                status=SessionStatus.EXITED,
                exit_timestamp=ts,
                updated_at=_utcnow(),
            )
            .execution_options(synchronize_session="fetch")
        )
        session.flush()
        logger.info(f"Session exited: {session_id}")

    @staticmethod
    def count_sessions(session: Session) -> int:
        """Count all visit sessions."""
        return session.execute(
            select(func.count()).select_from(VisitSession)
        ).scalar_one()

    @staticmethod
    def get_incomplete_sessions(session: Session) -> List[VisitSession]:
        """
        Return sessions that are ACTIVE or EXIT_PENDING but have no exit event.
        Used by recovery checker.
        """
        stmt = select(VisitSession).where(
            VisitSession.status.in_([SessionStatus.ACTIVE, SessionStatus.EXIT_PENDING])
        )
        return session.execute(stmt).scalars().all()


# --------- Event repository ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class EventRepository:
    """Operations on the events table."""

    @staticmethod
    def create_event(
        session: Session,
        face_id: str,
        session_id: Optional[str],
        source_id: str,
        event_type: EventType,
        event_timestamp: datetime,
        evidence_image_path: Optional[str] = None,
        frame_number: Optional[int] = None,
        meta: Optional[dict] = None,
        idempotency_key: Optional[str] = None,
    ) -> Optional[Event]:
        """
        Create an event record with idempotency protection.
        If an event with the same idempotency_key already exists, return None
        (idempotent retry --- not an error).

        Raises IntegrityError for constraint violations other than duplicates.
        """
        idem_key = idempotency_key or str(uuid.uuid4())
        event = Event(
            event_id=str(uuid.uuid4()),
            face_id=face_id,
            session_id=session_id,
            source_id=source_id,
            event_type=event_type,
            event_timestamp=event_timestamp,
            evidence_image_path=evidence_image_path,
            frame_number=frame_number,
            meta=meta,
            idempotency_key=idem_key,
        )
        try:
            with session.begin_nested():
                session.add(event)
                session.flush()
            return event
        except IntegrityError as exc:
            if "idempotency_key" in str(exc) or "uq_events" in str(exc):
                logger.info(
                    f"Duplicate event suppressed (idempotency_key={idem_key}): {exc}"
                )
                return None
            raise

    @staticmethod
    def has_entry_event(session: Session, session_id: str) -> bool:
        """Check if an entry event already exists for the session."""
        stmt = select(Event).where(
            Event.session_id == session_id,
            Event.event_type == EventType.ENTRY,
        )
        return session.execute(stmt).scalars().first() is not None

    @staticmethod
    def has_exit_event(session: Session, session_id: str) -> bool:
        """Check if an exit event already exists for the session."""
        stmt = select(Event).where(
            Event.session_id == session_id,
            Event.event_type == EventType.EXIT,
        )
        return session.execute(stmt).scalars().first() is not None

    @staticmethod
    def count_events(session: Session, event_type: Optional[EventType] = None) -> int:
        """Count events, optionally filtered by type."""
        stmt = select(func.count()).select_from(Event)
        if event_type:
            stmt = stmt.where(Event.event_type == event_type)
        return session.execute(stmt).scalar_one()

    @staticmethod
    def get_events_by_session(session: Session, session_id: str) -> List[Event]:
        """Fetch all events for a session ordered by timestamp."""
        stmt = (
            select(Event)
            .where(Event.session_id == session_id)
            .order_by(Event.event_timestamp)
        )
        return list(session.execute(stmt).scalars().all())


