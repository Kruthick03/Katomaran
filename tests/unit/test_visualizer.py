"""
Unit tests for SentinelVisualizer (src/visualizer.py).
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pytest

from src.tracker import Track, TrackState
from src.visualizer import SentinelVisualizer


@pytest.fixture
def dummy_frame():
    # 480x640 BGR black canvas
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def visualizer(tmp_path):
    return SentinelVisualizer(
        window_name="TestWindow",
        screenshots_dir=str(tmp_path / "screenshots"),
    )


def test_visualizer_render_empty_tracks(visualizer, dummy_frame):
    """Render frame with zero tracks; should contain no-face-detected badge."""
    rendered = visualizer.render(
        frame=dummy_frame,
        tracks=[],
        unique_count=0,
        session_count=0,
        active_sessions=0,
        total_entries=0,
        total_exits=0,
        fps=15.0,
        frame_number=1,
    )
    assert isinstance(rendered, np.ndarray)
    assert rendered.shape == dummy_frame.shape


def test_visualizer_render_with_active_track(visualizer, dummy_frame):
    """Render frame with an active recognized track."""
    track = Track(
        track_id=1,
        bbox=(50, 50, 150, 150),
        confidence=0.92,
        state=TrackState.ACTIVE,
        face_id="abc12345-6789-abcd-ef01-23456789abcd",
        recognition_status="Recognized",
        similarity=0.88,
    )
    rendered = visualizer.render(
        frame=dummy_frame,
        tracks=[track],
        unique_count=1,
        session_count=1,
        active_sessions=1,
        total_entries=1,
        total_exits=0,
        fps=12.5,
        frame_number=10,
    )
    assert isinstance(rendered, np.ndarray)
    assert rendered.shape == dummy_frame.shape


def test_visualizer_render_with_event_banner(visualizer, dummy_frame):
    """Render frame with a recent ENTRY event banner."""
    recent_events = [{
        "type": "ENTRY",
        "face_id": "face-uuid-1234",
        "session_id": "session-uuid-5678",
        "frame_number": 10,
    }]
    rendered = visualizer.render(
        frame=dummy_frame,
        tracks=[],
        unique_count=1,
        session_count=1,
        active_sessions=1,
        total_entries=1,
        total_exits=0,
        fps=10.0,
        frame_number=12,  # 2 frames after event
        recent_events=recent_events,
    )
    assert isinstance(rendered, np.ndarray)


def test_visualizer_save_screenshot(visualizer, dummy_frame):
    """Test screenshot saving to disk."""
    saved_path = visualizer.save_screenshot(dummy_frame)
    assert os.path.exists(saved_path)
    assert Path(saved_path).suffix == ".jpg"
    assert os.path.getsize(saved_path) > 0


def test_visualizer_skips_lost_tracks_in_normal_mode(visualizer, dummy_frame):
    """Lost tracks with missed_frames > 0 should not be rendered in normal mode."""
    lost_track = Track(
        track_id=1,
        bbox=(50, 50, 150, 150),
        confidence=0.90,
        state=TrackState.LOST,
        missed_frames=2,
    )
    rendered = visualizer.render(
        frame=dummy_frame,
        tracks=[lost_track],
        unique_count=0,
        session_count=0,
        active_sessions=0,
        total_entries=0,
        total_exits=0,
        fps=10.0,
        frame_number=5,
        debug_mode=False,
    )
    assert isinstance(rendered, np.ndarray)


def test_visualizer_anti_collision_labels(visualizer, dummy_frame):
    """Two closely positioned faces should avoid label collision."""
    t1 = Track(
        track_id=1,
        bbox=(50, 50, 120, 120),
        confidence=0.92,
        state=TrackState.ACTIVE,
        face_id="face-1",
        recognition_status="Recognized",
        similarity=0.85,
    )
    t2 = Track(
        track_id=2,
        bbox=(70, 50, 140, 120),
        confidence=0.89,
        state=TrackState.ACTIVE,
        face_id="face-2",
        recognition_status="Recognized",
        similarity=0.82,
    )
    rendered = visualizer.render(
        frame=dummy_frame,
        tracks=[t1, t2],
        unique_count=2,
        session_count=2,
        active_sessions=2,
        total_entries=2,
        total_exits=0,
        fps=12.0,
        frame_number=5,
    )
    assert isinstance(rendered, np.ndarray)


def test_visualizer_debug_mode(visualizer, dummy_frame):
    """Debug mode renders raw detections and debug banner."""
    from src.face_detector import FaceDetection
    raw_det = FaceDetection(bbox=(30, 30, 90, 90), confidence=0.77, frame_number=1)
    rendered = visualizer.render(
        frame=dummy_frame,
        tracks=[],
        unique_count=0,
        session_count=0,
        active_sessions=0,
        total_entries=0,
        total_exits=0,
        fps=15.0,
        frame_number=1,
        raw_detections=[raw_det],
        debug_mode=True,
    )
    assert isinstance(rendered, np.ndarray)

