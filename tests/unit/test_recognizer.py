"""
SentinelTrack — tests/unit/test_recognizer.py
Unit tests for embedding normalization, cosine similarity, and matching logic.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.face_recognizer import (
    normalize_embedding,
    cosine_similarity,
    embedding_to_bytes,
    bytes_to_embedding,
    RegisteredFace,
)


def make_random_embedding(dim=512) -> np.ndarray:
    """Create a random normalized embedding."""
    raw = np.random.randn(dim).astype(np.float32)
    return normalize_embedding(raw)


def test_normalize_unit_norm():
    """Normalized embedding should have L2 norm of ~1.0."""
    raw = np.array([3.0, 4.0], dtype=np.float32)
    norm = normalize_embedding(raw)
    assert abs(np.linalg.norm(norm) - 1.0) < 1e-6


def test_normalize_zero_vector():
    """Zero vector normalization should not crash."""
    raw = np.zeros(512, dtype=np.float32)
    result = normalize_embedding(raw)
    # Should return as-is without division by zero error
    assert result is not None


def test_cosine_similarity_identical():
    """Same vector has cosine similarity of 1.0."""
    v = make_random_embedding()
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-5


def test_cosine_similarity_orthogonal():
    """Orthogonal vectors have cosine similarity near 0."""
    a = normalize_embedding(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    b = normalize_embedding(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_range():
    """Cosine similarity between random vectors in [-1, 1]."""
    for _ in range(20):
        a = make_random_embedding()
        b = make_random_embedding()
        sim = cosine_similarity(a, b)
        assert -1.01 <= sim <= 1.01


def test_embedding_serialization_roundtrip():
    """Embedding serialized to bytes and back should be identical."""
    original = make_random_embedding(512)
    serialized = embedding_to_bytes(original)
    recovered = bytes_to_embedding(serialized, 512)
    np.testing.assert_array_almost_equal(original, recovered, decimal=6)


def test_match_returns_best():
    """Match should return the registered face with highest similarity."""
    from src.config import RecognitionConfig
    from src.face_recognizer import FaceRecognizer

    query = make_random_embedding()
    similar = query.copy()  # Identical → similarity=1.0
    different = make_random_embedding()

    # Ensure different is not accidentally similar
    while cosine_similarity(query, different) > 0.7:
        different = make_random_embedding()

    registered = [
        RegisteredFace(face_id="face_similar", embedding=similar),
        RegisteredFace(face_id="face_different", embedding=different),
    ]

    cfg = RecognitionConfig(similarity_threshold=0.3)
    recognizer = FaceRecognizer(cfg)

    matched_id, sim, ambiguous = recognizer.match(query, registered)
    assert matched_id == "face_similar"
    assert sim > 0.9


def test_match_no_registered():
    """Match with empty registered list returns None."""
    from src.config import RecognitionConfig
    from src.face_recognizer import FaceRecognizer

    query = make_random_embedding()
    cfg = RecognitionConfig(similarity_threshold=0.4)
    recognizer = FaceRecognizer(cfg)

    matched_id, sim, ambiguous = recognizer.match(query, [])
    assert matched_id is None
    assert sim == 0.0


def test_match_below_threshold():
    """Match below threshold returns None face_id."""
    from src.config import RecognitionConfig
    from src.face_recognizer import FaceRecognizer

    query = make_random_embedding()
    # Force a very different embedding
    anti_query = normalize_embedding(-query)
    registered = [RegisteredFace(face_id="face_A", embedding=anti_query)]

    cfg = RecognitionConfig(similarity_threshold=0.9)  # Very strict
    recognizer = FaceRecognizer(cfg)

    matched_id, sim, _ = recognizer.match(query, registered)
    assert matched_id is None


def test_ambiguity_detection():
    """Ambiguous match flagged when top-2 are within ambiguity_margin."""
    from src.config import RecognitionConfig
    from src.face_recognizer import FaceRecognizer

    base = make_random_embedding()
    # Create two very similar embeddings
    noise = np.random.randn(512).astype(np.float32) * 0.01
    emb1 = normalize_embedding(base + noise)
    emb2 = normalize_embedding(base + noise * 1.1)
    query = base.copy()

    registered = [
        RegisteredFace(face_id="A", embedding=emb1),
        RegisteredFace(face_id="B", embedding=emb2),
    ]

    # Large ambiguity margin to ensure it's flagged
    cfg = RecognitionConfig(similarity_threshold=0.3, ambiguity_margin=0.5)
    recognizer = FaceRecognizer(cfg)

    _, _, is_ambiguous = recognizer.match(query, registered)
    assert is_ambiguous is True
