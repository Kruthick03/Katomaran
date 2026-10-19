"""
SentinelTrack — tests/unit/test_tracker.py
Unit tests for the IoU-based face tracker.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.tracker import FaceTracker, Track, TrackState, compute_iou
from src.face_detector import FaceDetection


def make_detection(x1, y1, x2, y2, frame=1, conf=0.9):
    return FaceDetection(bbox=(x1, y1, x2, y2), confidence=conf, frame_number=frame)


def test_compute_iou_perfect_overlap():
    """Identical boxes have IoU of 1.0."""
    box = (10, 10, 100, 100)
    assert abs(compute_iou(box, box) - 1.0) < 1e-6


def test_compute_iou_no_overlap():
    """Non-overlapping boxes have IoU of 0.0."""
    a = (0, 0, 50, 50)
    b = (100, 100, 200, 200)
    assert compute_iou(a, b) == 0.0


def test_compute_iou_partial():
    """Partially overlapping boxes have IoU in (0, 1)."""
    a = (0, 0, 100, 100)
    b = (50, 50, 150, 150)
    iou = compute_iou(a, b)
    assert 0.0 < iou < 1.0


def test_new_track_created_on_detection():
    """New detection creates a new tentative track."""
    tracker = FaceTracker(max_missed_frames=5)
    det = make_detection(10, 10, 100, 100, frame=1)
    active, removed = tracker.update([det], frame_number=1)

    assert len(active) == 1
    assert active[0].state == TrackState.TENTATIVE
    assert active[0].bbox == (10, 10, 100, 100)


def test_track_promoted_to_active():
    """Track becomes active after min_track_age_frames detections."""
    tracker = FaceTracker(max_missed_frames=5, min_track_age_frames=3)
    for i in range(3):
        det = make_detection(10, 10, 100, 100, frame=i+1)
        active, _ = tracker.update([det], frame_number=i+1)

    assert any(t.state == TrackState.ACTIVE for t in active)


def test_track_expires_on_missed_frames():
    """Track is removed after exceeding max_missed_frames."""
    tracker = FaceTracker(max_missed_frames=3)
    det = make_detection(10, 10, 100, 100, frame=1)
    tracker.update([det], frame_number=1)

    # No detections for max_missed_frames + 1 frames
    removed_ids = set()
    for i in range(2, 7):
        active, removed = tracker.update([], frame_number=i)
        for r in removed:
            removed_ids.add(r.track_id)

    assert len(removed_ids) > 0


def test_two_tracks_not_merged():
    """Two spatially separated faces create two separate tracks."""
    tracker = FaceTracker(max_missed_frames=5)
    det1 = make_detection(10, 10, 80, 80, frame=1)
    det2 = make_detection(300, 300, 400, 400, frame=1)
    active, _ = tracker.update([det1, det2], frame_number=1)

    assert len(active) == 2
    assert active[0].track_id != active[1].track_id


def test_face_id_assignment():
    """Persistent face_id can be assigned to a track."""
    tracker = FaceTracker(max_missed_frames=5)
    det = make_detection(10, 10, 100, 100, frame=1)
    active, _ = tracker.update([det], frame_number=1)

    track_id = active[0].track_id
    tracker.assign_face_id(track_id, "face-abc-123")

    track = tracker.get_track(track_id)
    assert track.face_id == "face-abc-123"


def test_track_id_differs_from_face_id():
    """Track ID is an integer; face ID is a string UUID."""
    tracker = FaceTracker()
    det = make_detection(10, 10, 100, 100, frame=1)
    active, _ = tracker.update([det], frame_number=1)
    track = active[0]
    assert isinstance(track.track_id, int)
    assert track.face_id is None  # Not yet assigned
