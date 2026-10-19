"""
Tests specifically validating the incremental fixes:
- No premature track expiry while actively detected (max_age bugfix)
- Global best-first greedy IoU matching preventing track swap
- Temporary detection gaps (occlusion) not causing immediate exit
- Multiple tracks maintain separate identities
- Single entry and single exit per session lifecycle
"""
import pytest
from datetime import datetime, timezone
import numpy as np

from src.tracker import FaceTracker, TrackState, compute_iou
from src.face_detector import FaceDetection
from src.visit_session import VisitSessionManager, VisitState
from src.config import TrackerConfig, SessionConfig


def test_active_track_not_killed_after_30_frames():
    """Verify that an actively detected track is not removed after 30 frames."""
    tracker = FaceTracker(max_missed_frames=15, min_track_age_frames=2, max_age=30)
    
    # Simulate a face visible for 50 continuous frames
    for frame_idx in range(1, 51):
        det = FaceDetection(bbox=(100, 100, 200, 200), confidence=0.9, frame_number=frame_idx)
        active, removed = tracker.update([det], frame_number=frame_idx)
        assert len(active) == 1
        assert len(removed) == 0
        assert active[0].track_id == 1
        assert active[0].age == frame_idx
        if frame_idx >= 2:
            assert active[0].state == TrackState.ACTIVE


def test_iou_matching_two_simultaneous_faces():
    """Verify two distinct visible faces get separate track IDs without swapping."""
    tracker = FaceTracker(max_missed_frames=15, min_track_age_frames=1, iou_match_threshold=0.2)
    
    # Frame 1: Person A at left, Person B at right
    det_a = FaceDetection(bbox=(50, 50, 150, 150), confidence=0.9, frame_number=1)
    det_b = FaceDetection(bbox=(400, 50, 500, 150), confidence=0.9, frame_number=1)
    active, _ = tracker.update([det_a, det_b], frame_number=1)
    assert len(active) == 2
    id_a = [t.track_id for t in active if t.bbox[0] == 50][0]
    id_b = [t.track_id for t in active if t.bbox[0] == 400][0]
    assert id_a != id_b

    # Frame 2: Slight movement
    det_a2 = FaceDetection(bbox=(55, 52, 155, 152), confidence=0.9, frame_number=2)
    det_b2 = FaceDetection(bbox=(405, 52, 505, 152), confidence=0.9, frame_number=2)
    active2, _ = tracker.update([det_a2, det_b2], frame_number=2)
    assert len(active2) == 2
    track_map = {t.track_id: t.bbox[0] for t in active2}
    assert track_map[id_a] == 55
    assert track_map[id_b] == 405


def test_temporary_detection_gap_preserves_session():
    """Verify that a brief 2-frame detection gap does NOT cause session exit or duplicate entry."""
    from unittest.mock import MagicMock
    from src.models import VisitSession, SessionStatus
    
    mock_db = MagicMock()
    # Mock SessionRepository.get_active_session and create_session
    mock_session_obj = MagicMock()
    mock_session_obj.session_id = "test-session-uuid-1"
    
    from unittest.mock import patch
    with patch("src.visit_session.SessionRepository.create_session", return_value=mock_session_obj), \
         patch("src.visit_session.SessionRepository.get_active_session", return_value=None), \
         patch("src.visit_session.EventRepository.create_event", return_value=MagicMock()):
        
        session_cfg = SessionConfig(exit_confirmation_frames=10)
        manager = VisitSessionManager(config=session_cfg)
        now = datetime.now(timezone.utc)
        
        face_id = "test-face-gap-1234"
        # Frame 1: Face seen
        sess_id, is_new = manager.on_face_seen(mock_db, face_id, "cam_001", 1, now)
        assert is_new is True

        # Frame 2 & 3: Absent (temporary occlusion)
        manager.on_faces_absent(mock_db, set(), 2, now)
        manager.on_faces_absent(mock_db, set(), 3, now)

        # Frame 4: Re-appears
        sess_id_2, is_new_2 = manager.on_face_seen(mock_db, face_id, "cam_001", 4, now)
        assert is_new_2 is False  # NOT a new entry!
        assert sess_id_2 == sess_id  # Same visit session!
        assert manager._active_sessions[face_id].state == VisitState.ACTIVE


def test_cosine_similarity_safety_and_bounds():
    """Cosine similarity must strictly remain in [-1.0, 1.0] even with unnormalized inputs."""
    from src.face_recognizer import cosine_similarity
    
    # Large unnormalized vectors (simulating raw model outputs)
    v1 = np.random.randn(512).astype(np.float32) * 25.0
    v2 = np.random.randn(512).astype(np.float32) * 20.0
    sim = cosine_similarity(v1, v2)
    assert -1.0 <= sim <= 1.0

    # Identical vectors (even if scaled) must equal exactly 1.0
    assert abs(cosine_similarity(v1, v1) - 1.0) < 1e-5

    # Orthogonal vectors
    o1 = np.array([1.0, 0.0, 0.0], dtype=np.float32) * 15.0
    o2 = np.array([0.0, 1.0, 0.0], dtype=np.float32) * 12.0
    assert abs(cosine_similarity(o1, o2)) < 1e-5

    # Zero vectors must return 0.0 without division by zero crash
    z = np.zeros(512, dtype=np.float32)
    assert cosine_similarity(v1, z) == 0.0


def test_secondary_matching_recovers_fast_moving_face():
    """Verify that secondary pass matches a fast-moving face where IoU is ~0.15."""
    tracker = FaceTracker(max_missed_frames=15, min_track_age_frames=1, iou_match_threshold=0.20)
    
    # Frame 1: Face at (100, 100, 150, 150) (size 50x50)
    det1 = FaceDetection(bbox=(100, 100, 150, 150), confidence=0.9, frame_number=1)
    active, _ = tracker.update([det1], frame_number=1)
    assert len(active) == 1
    assert active[0].track_id == 1

    # Frame 2: Fast movement to (125, 120, 175, 170) (IoU is ~0.18, center moved 28px)
    det2 = FaceDetection(bbox=(125, 120, 175, 170), confidence=0.9, frame_number=2)
    active2, _ = tracker.update([det2], frame_number=2)
    assert len(active2) == 1
    # Must preserve the same track ID 1 rather than creating track ID 2!
    assert active2[0].track_id == 1
    assert active2[0].missed_frames == 0


def test_frame_skip_extrapolation_no_stale_replays():
    """Verify that on skipped frames, tracker predicts motion without replaying stale boxes or creating new tracks."""
    tracker = FaceTracker(max_missed_frames=15, min_track_age_frames=1, iou_match_threshold=0.20)
    
    # Frame 1: Detection at (100, 100, 200, 200)
    det1 = FaceDetection(bbox=(100, 100, 200, 200), confidence=0.9, frame_number=1)
    active1, _ = tracker.update([det1], frame_number=1, is_detection_frame=True)
    assert len(active1) == 1
    assert active1[0].track_id == 1

    # Frame 2: Face moved to (110, 105, 210, 205) -> velocity is (+10, +5, +10, +5)
    det2 = FaceDetection(bbox=(110, 105, 210, 205), confidence=0.9, frame_number=2)
    active2, _ = tracker.update([det2], frame_number=2, is_detection_frame=True)
    assert len(active2) == 1
    assert active2[0].track_id == 1
    assert active2[0].bbox == (110, 105, 210, 205)

    # Frame 3: SKIPPED FRAME (no detector run, empty detections, is_detection_frame=False)
    active3, removed3 = tracker.update([], frame_number=3, is_detection_frame=False)
    assert len(active3) == 1
    assert len(removed3) == 0
    assert active3[0].track_id == 1
    # Age advances correctly
    assert active3[0].age == 3
    # Missed frames counter must NOT increment on a skipped frame
    assert active3[0].missed_frames == 0
    # Coordinates must be extrapolated forward with velocity damping (0.8 * 10 = +8, 0.8 * 5 = +4)
    # Expected: (118, 109, 218, 209), definitely NOT frozen at (110, 105, 210, 205)
    assert active3[0].bbox[0] > 110
    assert active3[0].bbox[1] > 105


def test_detector_returns_empty_on_skipped_frames():
    """Verify that YOLOFaceDetector returns empty detections on skipped frames rather than replaying stale boxes."""
    from unittest.mock import MagicMock
    from src.face_detector import YOLOFaceDetector
    
    detector = YOLOFaceDetector(
        model_path="models/yolov8n-face.pt",
        confidence_threshold=0.35,
        iou_threshold=0.45,
        detection_frame_skip=3,
        min_face_width=20,
        min_face_height=20,
    )
    detector._model = MagicMock()
    
    # Mock box output for frame 1
    mock_box = MagicMock()
    mock_box.xyxy = [MagicMock()]
    mock_box.xyxy[0].cpu().numpy.return_value = np.array([50, 50, 100, 100])
    mock_box.conf = [MagicMock()]
    mock_box.conf[0].cpu().numpy.return_value = np.array(0.85)
    
    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    detector._model.predict.return_value = [mock_result]
    
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Frame 1: Detection frame (runs model)
    dets1, ran1 = detector.detect(dummy_frame, 1)
    assert ran1 is True
    assert len(dets1) == 1
    assert dets1[0].bbox == (50, 50, 100, 100)
    
    # Frame 2: Skipped frame (must return empty list, ran=False, NOT cached detections)
    dets2, ran2 = detector.detect(dummy_frame, 2)
    assert ran2 is False
    assert len(dets2) == 0
    
    # Frame 3: Skipped frame (must return empty list, ran=False)
    dets3, ran3 = detector.detect(dummy_frame, 3)
    assert ran3 is False
    assert len(dets3) == 0
    
    # Frame 4: Detection frame (runs model again)
    dets4, ran4 = detector.detect(dummy_frame, 4)
    assert ran4 is True
    assert len(dets4) == 1


