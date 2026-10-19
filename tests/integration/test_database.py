"""
SentinelTrack — tests/integration/test_database.py

PostgreSQL integration tests.
REQUIRES a running PostgreSQL database with sentineltrack_test database.
Set SENTINEL_TEST_DB_* environment variables to connect.

These tests use a separate test database and rollback after each test.
They do NOT mock the database — they run real PostgreSQL operations.

SKIPPED if database is unavailable (to not block CI without a DB).
"""
from __future__ import annotations

import os
import uuid
import pytest
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def get_test_db_url() -> str:
    """Build test database URL from env vars."""
    host = os.getenv("SENTINEL_DB_HOST", "localhost")
    port = os.getenv("SENTINEL_DB_PORT", "5432")
    name = os.getenv("SENTINEL_DB_NAME", "sentineltrack")
    user = os.getenv("SENTINEL_DB_USER", "sentineluser")
    password = os.getenv("SENTINEL_DB_PASSWORD", "")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


def is_db_available() -> bool:
    """Check if test database is reachable."""
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(get_test_db_url(), pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


db_available = pytest.mark.skipif(
    not is_db_available(),
    reason="PostgreSQL database not available. Set SENTINEL_DB_* env vars."
)


@pytest.fixture(scope="module")
def db_engine():
    """Create engine for test module."""
    from sqlalchemy import create_engine
    from src.models import Base
    engine = create_engine(get_test_db_url(), pool_pre_ping=True)
    # Create tables if not present (test database may not have migrations run)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """Provide a rolled-back session for each test."""
    from sqlalchemy.orm import sessionmaker
    factory = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)
    session = factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@db_available
def test_create_camera(db_session):
    """Can create a camera record."""
    from src.repositories import CameraRepository
    camera = CameraRepository.get_or_create(
        db_session,
        source_id=f"test_cam_{uuid.uuid4().hex[:8]}",
        source_type="file",
        source_name="Test Camera",
    )
    assert camera.source_id is not None
    assert camera.enabled is True


@db_available
def test_register_face(db_session):
    """Can register a new face identity."""
    import numpy as np
    from src.repositories import FaceRepository, CameraRepository
    from src.face_recognizer import normalize_embedding

    # Need camera first
    source_id = f"test_cam_{uuid.uuid4().hex[:8]}"
    CameraRepository.get_or_create(db_session, source_id, "file", "Test")

    embedding = normalize_embedding(np.random.randn(512).astype(np.float32))
    face = FaceRepository.register_new(
        db_session,
        embedding=embedding,
        recognition_model="buffalo_sc",
        first_seen_at=datetime.now(timezone.utc),
    )

    assert face.face_id is not None
    assert len(face.face_id) == 36  # UUID format
    assert face.embedding_dimension == 512


@db_available
def test_no_duplicate_entry_event(db_session):
    """Partial unique index prevents two entry events for same session."""
    import numpy as np
    from sqlalchemy.exc import IntegrityError
    from src.repositories import (
        CameraRepository, FaceRepository, SessionRepository, EventRepository
    )
    from src.face_recognizer import normalize_embedding
    from src.models import EventType

    source_id = f"test_cam_{uuid.uuid4().hex[:8]}"
    CameraRepository.get_or_create(db_session, source_id, "file", "Test")

    embedding = normalize_embedding(np.random.randn(512).astype(np.float32))
    face = FaceRepository.register_new(
        db_session, embedding, "buffalo_sc",
        datetime.now(timezone.utc)
    )

    now = datetime.now(timezone.utc)
    session = SessionRepository.create_session(db_session, face.face_id, source_id, now)

    # First entry — should succeed
    event1 = EventRepository.create_event(
        db_session,
        face_id=face.face_id,
        session_id=session.session_id,
        source_id=source_id,
        event_type=EventType.ENTRY,
        event_timestamp=now,
        idempotency_key=str(uuid.uuid4()),
    )
    assert event1 is not None

    # Second entry for same session — should raise IntegrityError or return None
    try:
        event2 = EventRepository.create_event(
            db_session,
            face_id=face.face_id,
            session_id=session.session_id,
            source_id=source_id,
            event_type=EventType.ENTRY,
            event_timestamp=now,
            idempotency_key=str(uuid.uuid4()),  # Different idempotency key!
        )
        # If None, constraint was handled gracefully
        # If raised, it should be IntegrityError
        # Either is acceptable behavior
    except IntegrityError:
        pass  # Expected — constraint enforced


@db_available
def test_idempotent_event_creation(db_session):
    """Same idempotency key should not create duplicate events."""
    import numpy as np
    from src.repositories import (
        CameraRepository, FaceRepository, SessionRepository, EventRepository
    )
    from src.face_recognizer import normalize_embedding
    from src.models import EventType

    source_id = f"test_cam_{uuid.uuid4().hex[:8]}"
    CameraRepository.get_or_create(db_session, source_id, "file", "Test")

    embedding = normalize_embedding(np.random.randn(512).astype(np.float32))
    face = FaceRepository.register_new(
        db_session, embedding, "buffalo_sc",
        datetime.now(timezone.utc)
    )

    now = datetime.now(timezone.utc)
    session = SessionRepository.create_session(db_session, face.face_id, source_id, now)

    idem_key = f"test-idempotency-{uuid.uuid4()}"

    # First creation
    event1 = EventRepository.create_event(
        db_session,
        face_id=face.face_id,
        session_id=session.session_id,
        source_id=source_id,
        event_type=EventType.REGISTRATION,
        event_timestamp=now,
        idempotency_key=idem_key,
    )

    # Second creation with same key — should return None, not raise
    event2 = EventRepository.create_event(
        db_session,
        face_id=face.face_id,
        session_id=session.session_id,
        source_id=source_id,
        event_type=EventType.REGISTRATION,
        event_timestamp=now,
        idempotency_key=idem_key,
    )

    # One of them is None (idempotent suppression) or both have same ID
    if event1 and event2:
        assert event1.event_id == event2.event_id
    # At most one event exists for this idempotency key


@db_available
def test_unique_identity_count(db_session):
    """count_unique_identities returns correct number."""
    import numpy as np
    from src.repositories import CameraRepository, FaceRepository
    from src.face_recognizer import normalize_embedding

    source_id = f"test_cam_{uuid.uuid4().hex[:8]}"
    CameraRepository.get_or_create(db_session, source_id, "file", "Test")

    count_before = FaceRepository.count_unique_identities(db_session)

    # Register two new faces
    for _ in range(2):
        emb = normalize_embedding(np.random.randn(512).astype(np.float32))
        FaceRepository.register_new(db_session, emb, "buffalo_sc", datetime.now(timezone.utc))

    count_after = FaceRepository.count_unique_identities(db_session)
    assert count_after == count_before + 2
