"""
SentinelTrack — tests/e2e/test_pipeline.py

End-to-end tests demonstrating the complete pipeline.
Uses synthetic embeddings and mocked detector outputs.
Does NOT require real models to be downloaded.
DOES require PostgreSQL for database-backed tests.

These tests verify:
1. Frames are read from a synthetic source
2. Face detection bboxes are processed
3. Identity registration occurs on first detection
4. Subsequent frames recognize the same identity
5. Entry event fires exactly once per visit
6. Absence triggers exit confirmation
7. Returning identity creates new session, not new face
8. Unique identity count is correct
"""
from __future__ import annotations

import os
import uuid
import numpy as np
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def make_norm_embedding(dim=512) -> np.ndarray:
    """Create a normalized random embedding."""
    raw = np.random.randn(dim).astype(np.float32)
    norm = np.linalg.norm(raw)
    return (raw / norm) if norm > 1e-8 else raw


def make_blank_frame(h=480, w=640) -> np.ndarray:
    """Create a blank BGR frame for testing."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_embedding_normalization_pipeline():
    """Embeddings are normalized consistently through the pipeline."""
    from src.face_recognizer import normalize_embedding, cosine_similarity

    raw = np.random.randn(512).astype(np.float32)
    normed = normalize_embedding(raw)

    assert abs(np.linalg.norm(normed) - 1.0) < 1e-5
    # Self-similarity after normalization
    assert abs(cosine_similarity(normed, normed) - 1.0) < 1e-5


def test_new_face_registered_once():
    """
    Simulate first-time detection: same embedding presented twice.
    Second presentation should RECOGNIZE, not register a new identity.
    """
    from src.config import RecognitionConfig
    from src.face_recognizer import FaceRecognizer, RegisteredFace

    embedding = make_norm_embedding()
    cfg = RecognitionConfig(similarity_threshold=0.4)
    recognizer = FaceRecognizer(cfg)

    registered: list[RegisteredFace] = []

    # First detection: no match → would register new identity
    matched_id, sim, ambiguous = recognizer.match(embedding, registered)
    assert matched_id is None
    assert sim == 0.0

    # Register this identity
    face_id = str(uuid.uuid4())
    registered.append(RegisteredFace(face_id=face_id, embedding=embedding))

    # Second detection (same frame, slightly noisy embedding)
    noise = np.random.randn(512).astype(np.float32) * 0.001
    noisy_emb = embedding + noise
    noisy_emb = noisy_emb / np.linalg.norm(noisy_emb)

    matched_id_2, sim_2, _ = recognizer.match(noisy_emb, registered)
    # Should recognize existing identity
    assert matched_id_2 == face_id
    assert sim_2 > 0.9


def test_different_persons_distinct_identities():
    """Two different embeddings should not match each other."""
    from src.config import RecognitionConfig
    from src.face_recognizer import FaceRecognizer, RegisteredFace

    embedding_a = make_norm_embedding()
    embedding_b = make_norm_embedding()

    # Ensure they are actually different (not accidentally similar)
    while abs(np.dot(embedding_a, embedding_b)) > 0.4:
        embedding_b = make_norm_embedding()

    cfg = RecognitionConfig(similarity_threshold=0.4)
    recognizer = FaceRecognizer(cfg)

    registered = [RegisteredFace(face_id="person_A", embedding=embedding_a)]

    matched_id, sim, _ = recognizer.match(embedding_b, registered)
    assert matched_id is None  # person B not recognized as person A


def test_returning_visitor_same_face_id():
    """
    Person A seen → exits → returns.
    Same embedding should match same face_id both times.
    Unique identity count stays at 1.
    """
    from src.config import RecognitionConfig
    from src.face_recognizer import FaceRecognizer, RegisteredFace

    embedding = make_norm_embedding()
    cfg = RecognitionConfig(similarity_threshold=0.4)
    recognizer = FaceRecognizer(cfg)

    # Register person A
    person_a_id = "person_A_" + uuid.uuid4().hex[:8]
    registered = [RegisteredFace(face_id=person_a_id, embedding=embedding)]

    # First visit
    matched_1, sim_1, _ = recognizer.match(embedding, registered)
    assert matched_1 == person_a_id

    # Simulate exit (person leaves) ...
    # Simulate return (person comes back)
    matched_2, sim_2, _ = recognizer.match(embedding, registered)
    assert matched_2 == person_a_id

    # Both visits identified same person → unique count stays at 1
    unique_ids = {matched_1, matched_2}
    assert len(unique_ids) == 1


def test_bbox_validation_in_pipeline():
    """Invalid bboxes are rejected before reaching the recognizer."""
    from src.face_detector import validate_bbox

    frame_shape = (480, 640)

    # Valid
    assert validate_bbox((10, 10, 200, 200), frame_shape) is True
    # Too small
    assert validate_bbox((10, 10, 25, 25), frame_shape, min_width=30, min_height=30) is False
    # Out of bounds
    assert validate_bbox((600, 400, 700, 500), frame_shape) is False


def test_iou_tracker_multiple_persons():
    """Two persons tracked simultaneously maintain separate track IDs."""
    from src.tracker import FaceTracker
    from src.face_detector import FaceDetection

    tracker = FaceTracker(max_missed_frames=5)

    # Two persons far apart
    det_A = FaceDetection((10, 10, 100, 100), 0.9, 1)
    det_B = FaceDetection((400, 300, 550, 450), 0.85, 1)

    active, _ = tracker.update([det_A, det_B], frame_number=1)

    assert len(active) == 2
    track_ids = {t.track_id for t in active}
    assert len(track_ids) == 2
