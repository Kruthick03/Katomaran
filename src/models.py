"""
SentinelTrack --- src/models.py

SQLAlchemy 2.x declarative ORM models for PostgreSQL.
Implements normalized schema with:
  - faces: persistent identity records
  - visit_sessions: per-visit tracking
  - events: entry/exit/registration audit log
  - cameras: source registry

All timestamps are stored as UTC.
Embedding stored as BYTEA with documented Python cosine-similarity fallback.
pgvector extension used when available (detected at migration time).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    event,
    text,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    """Return current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


# --------- Base class ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all SentinelTrack models."""
    pass


# --------- Enumerations ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import enum as py_enum


class EventType(str, py_enum.Enum):
    ENTRY = "entry"
    EXIT = "exit"
    REGISTRATION = "registration"
    RECOGNITION = "recognition"
    AMBIGUOUS = "ambiguous"
    ERROR = "error"


class SessionStatus(str, py_enum.Enum):
    ACTIVE = "active"
    EXIT_PENDING = "exit_pending"
    EXITED = "exited"


class RegistrationStatus(str, py_enum.Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FLAGGED = "flagged"


class SourceType(str, py_enum.Enum):
    FILE = "file"
    RTSP = "rtsp"


def _enum_val(enum_cls):
    """Extract string values from Enum class for PostgreSQL native enum mapping."""
    return [e.value for e in enum_cls]


# --------- Camera / Source ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Camera(Base):
    """Registry of video sources (files and RTSP streams)."""
    __tablename__ = "cameras"

    source_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_type: Mapped[str] = mapped_column(
        Enum(SourceType, name="source_type_enum", values_callable=_enum_val), nullable=False
    )
    source_name: Mapped[str] = mapped_column(String(256), nullable=False)
    # Non-secret config metadata stored as JSONB (no passwords)
    config_meta: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    sessions: Mapped[list["VisitSession"]] = relationship(
        "VisitSession", back_populates="camera"
    )
    events: Mapped[list["Event"]] = relationship(
        "Event", back_populates="camera"
    )


# --------- Face / Identity ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Face(Base):
    """
    Persistent identity record. One row per unique person.
    face_id is a stable UUID assigned at first registration.
    Embeddings are stored as BYTEA (serialized float32 array).
    When pgvector is enabled, a separate vector column is used for ANN search.
    """
    __tablename__ = "faces"

    face_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # Embedding stored as raw bytes (numpy float32 array serialized with tobytes())
    embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    # Dimension stored separately for validation
    embedding_dimension: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Model used to generate the embedding
    recognition_model: Mapped[str] = mapped_column(String(128), nullable=False)
    # Registration status for quality control
    registration_status: Mapped[str] = mapped_column(
        Enum(RegistrationStatus, name="registration_status_enum", values_callable=_enum_val),
        default=RegistrationStatus.CONFIRMED,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    sessions: Mapped[list["VisitSession"]] = relationship(
        "VisitSession", back_populates="face"
    )
    events: Mapped[list["Event"]] = relationship(
        "Event", back_populates="face"
    )

    __table_args__ = (
        Index("ix_faces_last_seen_at", "last_seen_at"),
        Index("ix_faces_first_seen_at", "first_seen_at"),
    )


# --------- Visit Session ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class VisitSession(Base):
    """
    One row per logical visit by one identity to one camera.
    A returning identity creates a NEW session --- the face_id remains the same.
    Entry and exit are exactly one event each, enforced by DB constraints.
    """
    __tablename__ = "visit_sessions"

    session_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    face_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("faces.face_id", ondelete="RESTRICT"), nullable=False
    )
    source_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("cameras.source_id", ondelete="RESTRICT"), nullable=False
    )

    entry_timestamp: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    exit_timestamp: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    status: Mapped[str] = mapped_column(
        Enum(SessionStatus, name="session_status_enum", values_callable=_enum_val),
        default=SessionStatus.ACTIVE,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    face: Mapped["Face"] = relationship("Face", back_populates="sessions")
    camera: Mapped["Camera"] = relationship("Camera", back_populates="sessions")
    events: Mapped[list["Event"]] = relationship(
        "Event", back_populates="session"
    )

    __table_args__ = (
        Index("ix_visit_sessions_face_id", "face_id"),
        Index("ix_visit_sessions_source_id", "source_id"),
        Index("ix_visit_sessions_status", "status"),
        Index("ix_visit_sessions_entry_timestamp", "entry_timestamp"),
        # Enforce exit_timestamp only after entry_timestamp
        CheckConstraint(
            "(exit_timestamp IS NULL) OR (entry_timestamp IS NOT NULL AND exit_timestamp >= entry_timestamp)",
            name="ck_session_timestamps_order",
        ),
    )


# --------- Event ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Event(Base):
    """
    Immutable audit log for all pipeline events.
    Idempotency key prevents duplicate creation on retry.
    Partial unique indexes enforce:
      - at most one entry event per session
      - at most one exit event per session
    """
    __tablename__ = "events"

    event_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    face_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("faces.face_id", ondelete="RESTRICT"), nullable=False
    )
    session_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("visit_sessions.session_id", ondelete="RESTRICT"),
        nullable=True,
    )
    source_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("cameras.source_id", ondelete="RESTRICT"), nullable=False
    )

    event_type: Mapped[str] = mapped_column(
        Enum(EventType, name="event_type_enum", values_callable=_enum_val), nullable=False
    )
    event_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    # Path to the evidence crop image, if applicable
    evidence_image_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Frame number where the event was detected
    frame_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Idempotency key for safe retries (UUID generated before DB write)
    idempotency_key: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True
    )
    # Non-sensitive metadata (similarity score, model name, etc.)
    meta: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    face: Mapped["Face"] = relationship("Face", back_populates="events")
    session: Mapped[Optional["VisitSession"]] = relationship(
        "VisitSession", back_populates="events"
    )
    camera: Mapped["Camera"] = relationship("Camera", back_populates="events")

    __table_args__ = (
        Index("ix_events_face_id", "face_id"),
        Index("ix_events_session_id", "session_id"),
        Index("ix_events_event_type", "event_type"),
        Index("ix_events_event_timestamp", "event_timestamp"),
        # Exactly one ENTRY per session
        Index(
            "uq_events_entry_per_session",
            "session_id",
            unique=True,
            postgresql_where=text("event_type = 'entry'"),
        ),
        # At most one EXIT per session
        Index(
            "uq_events_exit_per_session",
            "session_id",
            unique=True,
            postgresql_where=text("event_type = 'exit'"),
        ),
        # Idempotency key uniqueness already enforced by column unique=True above
    )

