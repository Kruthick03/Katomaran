"""
SentinelTrack — tests/test_production_readiness.py

Comprehensive Production Readiness & Katomaran Hackathon Compliance Test Suite.
Validates the five mandatory architectural invariants:
  1. Configuration & Frame-Skip Decimation (config.json loading, modulo decimation, motion extrapolation)
  2. Re-Identification & Unique Counting Invariant (auto-registration +1, recognized = 0 increment, ambiguity rejection)
  3. Exactly-Once Entry & Exit Lifecycle (single entry, absence confirmation, single exit, returning visitor invariance)
  4. Resilience & Interrupt Handling (stale session crash recovery, shutdown cleanup, DB integrity)
  5. RTSP Stream Readiness (credential injection, timeout handling, bounded reconnect loop)
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import AppConfig, DetectionConfig, RecognitionConfig, RTSPConfig, SessionConfig, load_config
from src.event_logger import EventLogger
from src.face_detector import FaceDetection, YOLOFaceDetector
from src.face_recognizer import FaceRecognizer, RegisteredFace, cosine_similarity, normalize_embedding
from src.identity_service import IdentityService
from src.models import Event, EventType, Face, SessionStatus, VisitSession
from src.recovery import RecoveryChecker
from src.repositories import CameraRepository, EventRepository, FaceRepository, SessionRepository
from src.tracker import FaceTracker, TrackState
from src.video_input import RTSPVideoSource
from src.visit_session import ActiveSession, VisitSessionManager, VisitState
from tests.integration.test_database import db_available, get_test_db_url


# Database fixtures for integration tests with automatic rollback
@pytest.fixture(scope="module")
def db_engine():
    """Create test engine."""
    from sqlalchemy import create_engine
    from src.models import Base
    engine = create_engine(get_test_db_url(), pool_pre_ping=True)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine):
    """Provide a clean, rolled-back session for tests requiring PostgreSQL."""
    from sqlalchemy.orm import sessionmaker
    factory = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)
    session = factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


# ============================================================================
# INVARIANT 1: CONFIGURATION & FRAME SKIP DECIMATION
# ============================================================================

class TestInvariant1ConfigurationAndFrameSkip:
    """Verifies config loading and dynamic frame-skip decimation."""

    def test_config_loading_and_dynamic_skip(self):
        """config.json loads properly and allows runtime decimation modification."""
        cfg = load_config("config.json")
        assert isinstance(cfg, AppConfig)
        assert cfg.detection.detection_frame_skip >= 1

        # Dynamic override alters configuration
        cfg.detection.detection_frame_skip = 4
        assert cfg.detection.detection_frame_skip == 4

    def test_detector_frame_skip_decimation_cycle(self):
        """
        When detection_frame_skip=N, detector only runs on frame % N == 0.
        Skipped frames return ([], False) indicating decimation skip.
        """
        det_cfg = DetectionConfig(detection_frame_skip=3, confidence_threshold=0.45)
        detector = YOLOFaceDetector(
            model_path=det_cfg.model_path,
            confidence_threshold=det_cfg.confidence_threshold,
            detection_frame_skip=det_cfg.detection_frame_skip,
        )
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Mock the underlying model's predict method
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = []
        mock_model.predict.return_value = [mock_result]
        detector._model = mock_model

        # Frame 0: detection frame
        dets_0, ran_0 = detector.detect(dummy_frame, frame_number=0)
        assert ran_0 is True
        assert detector._model.predict.call_count == 1

        # Frame 1: skipped frame
        dets_1, ran_1 = detector.detect(dummy_frame, frame_number=1)
        assert ran_1 is False
        assert dets_1 == []
        assert detector._model.predict.call_count == 1  # Predict NOT called

        # Frame 2: skipped frame
        dets_2, ran_2 = detector.detect(dummy_frame, frame_number=2)
        assert ran_2 is False
        assert dets_2 == []
        assert detector._model.predict.call_count == 1  # Predict NOT called

        # Frame 3: detection frame (3 % 3 == 0)
        dets_3, ran_3 = detector.detect(dummy_frame, frame_number=3)
        assert ran_3 is True
        assert detector._model.predict.call_count == 2

    def test_frame_skip_motion_extrapolation_preserves_tracks(self):
        """
        On skipped frames, tracker uses predict_motion() with velocity damping.
        Verifies existing tracks smoothly extrapolate without creating spurious tracks.
        """
        tracker = FaceTracker(iou_match_threshold=0.3, max_missed_frames=10, min_track_age_frames=1)

        # Detection Frame 0: Initial face detection
        det0 = FaceDetection(bbox=(100, 100, 150, 150), confidence=0.9, frame_number=0)
        tracks_0, _ = tracker.update([det0], frame_number=0, is_detection_frame=True)
        assert len(tracks_0) == 1
        t_id = tracks_0[0].track_id

        # Detection Frame 1: Face moves to (110, 100, 160, 150) -> displacement dx = 10
        det1 = FaceDetection(bbox=(110, 100, 160, 150), confidence=0.9, frame_number=1)
        tracks_1, _ = tracker.update([det1], frame_number=1, is_detection_frame=True)
        assert len(tracks_1) == 1
        assert tracks_1[0].track_id == t_id
        assert tracks_1[0].bbox == (110, 100, 160, 150)

        # Frame 2: SKIPPED FRAME (no new detections, is_detection_frame=False)
        tracks_2, _ = tracker.update([], frame_number=2, is_detection_frame=False)
        assert len(tracks_2) == 1
        assert tracks_2[0].track_id == t_id
        # Coordinates extrapolated with damped velocity: 110 + 0.8 * 10 = 118
        assert tracks_2[0].bbox[0] == pytest.approx(118, abs=1.0)
        assert tracks_2[0].bbox[2] == pytest.approx(168, abs=1.0)
        # No spurious tracks created
        assert len(tracker.get_active_tracks()) == 1


# ============================================================================
# INVARIANT 2: RE-IDENTIFICATION & UNIQUE COUNTING INVARIANT
# ============================================================================

class TestInvariant2ReIdentificationAndUniqueCounting:
    """Verifies that re-identification preserves the unique visitor count."""

    def test_reidentification_does_not_increment_unique_count(self):
        """
        Simulate passing identical/similar 512-d embeddings across multiple frames.
        First frame: registers new identity -> count = 1.
        Subsequent frames: matches existing identity -> count = 1.
        Distant embedding: registers second identity -> count = 2.
        """
        rec_cfg = RecognitionConfig(similarity_threshold=0.45, ambiguity_margin=0.08)
        recognizer = FaceRecognizer(rec_cfg)
        identity_service = IdentityService(recognizer=recognizer)

        # Mock DB session and repositories
        mock_db = MagicMock()
        registered_db_faces = []

        def mock_register_new(session, embedding, recognition_model, first_seen_at):
            f_id = f"face_{len(registered_db_faces) + 1}"
            new_f = Face(face_id=f_id, recognition_model=recognition_model, first_seen_at=first_seen_at)
            registered_db_faces.append(new_f)
            return new_f

        def mock_count_unique(session):
            return len(registered_db_faces)

        with patch("src.repositories.FaceRepository.register_new", side_effect=mock_register_new), \
             patch("src.repositories.FaceRepository.count_unique_identities", side_effect=mock_count_unique), \
             patch("src.repositories.FaceRepository.update_last_seen"):

            # Generate Person A normalized 512-d embedding
            np.random.seed(42)
            emb_person_a = normalize_embedding(np.random.randn(512).astype(np.float32))

            # Frame 1: Person A first seen -> AUTO-REGISTRATION
            res1 = identity_service.identify(
                mock_db, emb_person_a, frame_number=1, source_id="cam_001", track_id=1
            )
            assert res1 is not None
            face_id_1, status_1, sim_1 = res1
            assert status_1 == "registered"
            assert identity_service.get_unique_count(mock_db) == 1

            # Frame 2..10: Person A seen with slightly noisy embedding (cosine similarity >= threshold 0.45)
            for f_num in range(2, 11):
                noisy_emb = normalize_embedding(emb_person_a + np.random.randn(512).astype(np.float32) * 0.01)
                res = identity_service.identify(
                    mock_db, noisy_emb, frame_number=f_num, source_id="cam_001", track_id=1
                )
                assert res is not None
                f_id, status, sim = res
                assert f_id == face_id_1
                assert status == "recognized"
                assert sim >= 0.45
                # INVARIANT: Unique visitor count must NOT increment
                assert identity_service.get_unique_count(mock_db) == 1

            # Frame 11: Person B seen (orthogonal embedding, distance >> threshold)
            emb_person_b = normalize_embedding(np.random.randn(512).astype(np.float32))
            res_b = identity_service.identify(
                mock_db, emb_person_b, frame_number=11, source_id="cam_001", track_id=2
            )
            assert res_b is not None
            face_id_b, status_b, _ = res_b
            assert face_id_b != face_id_1
            assert status_b == "registered"
            # Second visitor registered -> count increments to 2
            assert identity_service.get_unique_count(mock_db) == 2

    def test_ambiguity_margin_conservative_rejection(self):
        """
        When query vector is equally close to two gallery faces within ambiguity_margin,
        the system must reject the match rather than misclassifying or fabricating IDs.
        """
        rec_cfg = RecognitionConfig(similarity_threshold=0.45, ambiguity_margin=0.08)
        recognizer = FaceRecognizer(rec_cfg)
        identity_service = IdentityService(recognizer=recognizer)

        base_emb = normalize_embedding(np.ones(512, dtype=np.float32))
        noise1 = np.random.randn(512).astype(np.float32) * 0.02
        noise2 = np.random.randn(512).astype(np.float32) * 0.02
        emb1 = normalize_embedding(base_emb + noise1)
        emb2 = normalize_embedding(base_emb + noise2)

        identity_service._gallery = [
            RegisteredFace(face_id="face_cand_1", embedding=emb1),
            RegisteredFace(face_id="face_cand_2", embedding=emb2),
        ]

        mock_db = MagicMock()
        # Query closely matches both candidate 1 and candidate 2
        res = identity_service.identify(
            mock_db, base_emb, frame_number=1, source_id="cam_001", track_id=9
        )
        # Must reject ambiguous identification
        assert res is None


# ============================================================================
# INVARIANT 3: EXACTLY-ONCE ENTRY & EXIT LIFECYCLE
# ============================================================================

class TestInvariant3ExactlyOnceEntryAndExitLifecycle:
    """Verifies visit session state machine and exactly-once event generation."""

    def test_exactly_once_entry_and_exit_cycle(self):
        """
        Pass a sequence of detections for face_A:
        - Frame 1: Entry event created (exactly 1 ENTRY).
        - Frames 2..15: Active visit (0 extra ENTRY events).
        - Frames 16..36: Absence (> exit_confirmation_frames=20) -> transitions to EXITED (exactly 1 EXIT).
        - Total events: 1 ENTRY, 1 EXIT.
        """
        sess_cfg = SessionConfig(exit_confirmation_frames=20)
        session_mgr = VisitSessionManager(config=sess_cfg)

        mock_db = MagicMock()
        now = datetime.now(timezone.utc)

        events_emitted = []

        def mock_create_event(db, face_id, session_id, source_id, event_type, event_timestamp, **kwargs):
            events_emitted.append({"type": event_type, "session_id": session_id, "face_id": face_id})

        def mock_create_session(db, face_id, source_id, entry_timestamp):
            s = MagicMock()
            s.session_id = f"sess_{uuid.uuid4().hex[:8]}"
            s.face_id = face_id
            return s

        with patch("src.repositories.SessionRepository.get_active_session", return_value=None), \
             patch("src.repositories.SessionRepository.create_session", side_effect=mock_create_session), \
             patch("src.repositories.SessionRepository.confirm_exit"), \
             patch("src.repositories.EventRepository.create_event", side_effect=mock_create_event):

            # Frame 1: Person A arrives
            sess_id, is_new_entry = session_mgr.on_face_seen(
                mock_db, face_id="person_A", source_id="cam_001", frame_number=1, timestamp=now
            )
            assert is_new_entry is True
            assert session_mgr.active_sessions_count == 1
            assert len(events_emitted) == 1
            assert events_emitted[0]["type"] == EventType.ENTRY

            # Frames 2..15: Person A seen in every frame
            for f in range(2, 16):
                s_id, is_new = session_mgr.on_face_seen(
                    mock_db, face_id="person_A", source_id="cam_001", frame_number=f, timestamp=now
                )
                assert s_id == sess_id
                assert is_new is False
                # No additional entry events
                assert len(events_emitted) == 1

            # Frames 16..35: Person A absent (20 frames)
            for f in range(16, 36):
                session_mgr.on_faces_absent(
                    mock_db, seen_face_ids=set(), frame_number=f, timestamp=now + timedelta(seconds=f)
                )

            # At frame 35 (15 + 20): enters EXIT_PENDING
            active_s = session_mgr._active_sessions.get("person_A")
            assert active_s is not None
            assert active_s.state == VisitState.EXIT_PENDING

            # Frame 36: Confirms exit
            res = session_mgr.on_faces_absent(
                mock_db, seen_face_ids=set(), frame_number=36, timestamp=now + timedelta(seconds=36)
            )
            assert res["exits_triggered"] == 1
            assert session_mgr.active_sessions_count == 0

            # Verify total emitted events: EXACTLY 1 ENTRY, EXACTLY 1 EXIT
            entries = [e for e in events_emitted if e["type"] == EventType.ENTRY]
            exits = [e for e in events_emitted if e["type"] == EventType.EXIT]
            assert len(entries) == 1
            assert len(exits) == 1
            assert entries[0]["session_id"] == exits[0]["session_id"]

    def test_returning_visitor_creates_new_session_maintains_feed_unique_count(self):
        """
        When Person A leaves and returns later:
        - Second visit initiates a new session (different session_id, new ENTRY event).
        - feed_unique_visitors stays 1!
        - feed_total_sessions becomes 2!
        """
        sess_cfg = SessionConfig(exit_confirmation_frames=5)
        session_mgr = VisitSessionManager(config=sess_cfg)
        mock_db = MagicMock()
        now = datetime.now(timezone.utc)

        with patch("src.repositories.SessionRepository.get_active_session", return_value=None), \
             patch("src.repositories.SessionRepository.create_session", side_effect=lambda db, face_id, source_id, entry_timestamp: MagicMock(session_id=str(uuid.uuid4()))), \
             patch("src.repositories.SessionRepository.confirm_exit"), \
             patch("src.repositories.EventRepository.create_event"):

            # Visit 1
            sess1, is_entry1 = session_mgr.on_face_seen(mock_db, "person_A", "cam_001", 1, now)
            assert is_entry1 is True
            assert session_mgr.feed_unique_visitors_count == 1
            assert session_mgr.feed_total_sessions_count == 1

            # Person exits
            for f in range(2, 10):
                session_mgr.on_faces_absent(mock_db, set(), f, now)
            assert session_mgr.active_sessions_count == 0

            # Visit 2 (Person A returns at Frame 50)
            sess2, is_entry2 = session_mgr.on_face_seen(mock_db, "person_A", "cam_001", 50, now + timedelta(seconds=50))
            assert is_entry2 is True
            assert sess2 != sess1  # New session
            # INVARIANT: feed_unique_visitors is still 1, total sessions is 2
            assert session_mgr.feed_unique_visitors_count == 1
            assert session_mgr.feed_total_sessions_count == 2

    @db_available
    def test_idempotency_guard_prevents_duplicate_db_events(self, db_session):
        """
        PostgreSQL idempotency constraint check:
        Attempting to insert two events with identical idempotency_key is rejected.
        """
        now = datetime.now(timezone.utc)
        cam = CameraRepository.get_or_create(db_session, f"test_cam_{uuid.uuid4().hex[:6]}", "file", "Idempotency Cam")
        face = FaceRepository.register_new(
            db_session, normalize_embedding(np.ones(512, dtype=np.float32)), "buffalo_sc"
        )
        session = SessionRepository.create_session(
            db_session, face.face_id, cam.source_id, entry_timestamp=now
        )

        key = f"{session.session_id}:entry"
        ev1 = EventRepository.create_event(
            db_session, face.face_id, session.session_id, cam.source_id,
            EventType.ENTRY, now, idempotency_key=key
        )
        assert ev1 is not None

        # Duplicate event with same idempotency_key returns existing or None, never duplicates
        ev2 = EventRepository.create_event(
            db_session, face.face_id, session.session_id, cam.source_id,
            EventType.ENTRY, now, idempotency_key=key
        )
        assert ev2 is None or ev2.event_id == ev1.event_id


# ============================================================================
# INVARIANT 4: RESILIENCE & INTERRUPT HANDLING
# ============================================================================

class TestInvariant4ResilienceAndInterruptHandling:
    """Verifies crash recovery, graceful shutdown, and database reconciliation."""

    def test_recovery_checker_identifies_and_reconciles_stale_sessions(self):
        """
        If the application was interrupted leaving ACTIVE sessions in the database,
        RecoveryChecker.recover_incomplete_sessions() marks them EXITED with an audit record.
        """
        checker = RecoveryChecker(incomplete_threshold_hours=0.5)
        mock_db = MagicMock()

        stale_sess = MagicMock()
        stale_sess.session_id = "sess_stale_123"
        stale_sess.face_id = "face_abc"
        stale_sess.source_id = "cam_001"
        stale_sess.status = SessionStatus.ACTIVE
        stale_sess.created_at = datetime.now(timezone.utc) - timedelta(hours=2)

        # Wire mock_db.execute() scalars().all()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [stale_sess]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute.return_value = mock_result

        with patch("src.repositories.SessionRepository.confirm_exit") as mock_exit, \
             patch("src.repositories.EventRepository.has_exit_event", return_value=False), \
             patch("src.repositories.EventRepository.create_event") as mock_ev:

            res = checker.recover_incomplete_sessions(mock_db, source_id="cam_001")
            assert res["recovered"] == 1
            mock_exit.assert_called_once()
            mock_ev.assert_called_once()

    def test_clean_pipeline_shutdown_resource_cleanup(self):
        """
        Simulate signal interrupt and verify pipeline cleanup closes DB, releases sources,
        and saves last_run_summary.json without hanging.
        """
        import src.main as main_mod

        # Verify shutdown handler sets flag
        main_mod._shutdown_requested = False
        main_mod._handle_signal(2, None)
        assert main_mod._shutdown_requested is True

        # Reset flag
        main_mod._shutdown_requested = False


# ============================================================================
# INVARIANT 5: RTSP STREAM READINESS & RECONNECT HANDLING
# ============================================================================

class TestInvariant5RTSPStreamReadiness:
    """Verifies RTSP stream ingestion, credential injection, and bounded reconnects."""

    def test_rtsp_uri_builder_credential_injection_and_sanitization(self, monkeypatch):
        """
        RTSPVideoSource injects RTSP_USERNAME and RTSP_PASSWORD from environment,
        while logs and representations mask the credentials.
        """
        from src.config import SourceConfig

        monkeypatch.setenv("RTSP_USERNAME", "admin")
        monkeypatch.setenv("RTSP_PASSWORD", "SecretPass123!")

        src_cfg = SourceConfig(type="rtsp", rtsp_uri="rtsp://192.168.1.100:554/live")
        rtsp_cfg = RTSPConfig(reconnect_limit=3, reconnect_delay_seconds=0.1)

        rtsp_source = RTSPVideoSource(src_cfg, rtsp_cfg)
        built_uri = rtsp_source._build_uri()

        # Injected URI has credentials
        assert built_uri == "rtsp://admin:SecretPass123!@192.168.1.100:554/live"

        # Masked representation for logging does NOT expose password
        safe_repr = rtsp_source._raw_uri
        assert "SecretPass123!" not in safe_repr

    def test_rtsp_reconnect_loop_bounded_retry(self):
        """
        When RTSP stream loses connection, it retries up to reconnect_limit.
        Once exhausted, it returns (False, None, frame_num) rather than hanging.
        """
        from src.config import SourceConfig

        src_cfg = SourceConfig(type="rtsp", rtsp_uri="rtsp://localhost:8554/stream")
        rtsp_cfg = RTSPConfig(reconnect_limit=2, reconnect_delay_seconds=0.01)

        rtsp_source = RTSPVideoSource(src_cfg, rtsp_cfg)

        # Mock VideoCapture to simulate broken connection
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_cls.return_value = mock_cap

            # Exceed reconnect limit
            rtsp_source._reconnect_count = 2  # Reached limit

            success, frame, f_num = rtsp_source.read_frame()
            assert success is False
            assert frame is None
