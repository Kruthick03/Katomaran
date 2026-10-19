"""
SentinelTrack — tests/unit/test_detector.py
Unit tests for face detection, bounding-box validation, and frame skipping.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.face_detector import validate_bbox, FaceDetection


def test_validate_bbox_valid():
    """Valid bounding box within frame bounds passes."""
    assert validate_bbox((10, 20, 100, 150), (480, 640), 30, 30) is True


def test_validate_bbox_degenerate():
    """Degenerate box (x1 >= x2) fails validation."""
    assert validate_bbox((100, 20, 50, 150), (480, 640)) is False


def test_validate_bbox_out_of_frame():
    """Box extending beyond frame fails validation."""
    assert validate_bbox((600, 20, 700, 100), (480, 640)) is False  # x2 > width


def test_validate_bbox_too_small():
    """Box smaller than minimum face size fails."""
    assert validate_bbox((0, 0, 15, 15), (480, 640), min_width=30, min_height=30) is False


def test_validate_bbox_negative_coords():
    """Box with negative coordinates fails validation."""
    assert validate_bbox((-5, 10, 100, 150), (480, 640)) is False


def test_face_detection_properties():
    """FaceDetection computed properties are correct."""
    det = FaceDetection(bbox=(10, 20, 110, 170), confidence=0.9, frame_number=5)
    assert det.width == 100
    assert det.height == 150
    assert det.area == 15000
    assert det.center == (60.0, 95.0)


def test_face_detection_x1_y1():
    """FaceDetection coordinate accessors work."""
    det = FaceDetection(bbox=(5, 10, 50, 80), confidence=0.8, frame_number=1)
    assert det.x1 == 5
    assert det.y1 == 10
    assert det.x2 == 50
    assert det.y2 == 80
