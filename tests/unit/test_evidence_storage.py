"""
SentinelTrack — tests/unit/test_evidence_storage.py
Unit tests for evidence storage path construction and validation.
"""
import os
import pytest
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def test_evidence_path_collision_resistance(tmp_path):
    """Evidence filenames include face_id, session_id, frame, and timestamp."""
    from src.config import EvidenceConfig
    from src.evidence_storage import EvidenceStorage

    cfg = EvidenceConfig(
        entries_dir=str(tmp_path / "entries"),
        exits_dir=str(tmp_path / "exits"),
    )
    storage = EvidenceStorage(cfg)

    # Two different frames should produce different filenames
    ts = datetime.now(timezone.utc)
    face_id = "face-abc-12345678"
    session_id = "sess-def-12345678"

    path1 = storage._build_path("entry", face_id, session_id, 1, ts)
    path2 = storage._build_path("entry", face_id, session_id, 2, ts)

    assert path1 != path2


def test_save_evidence_valid_crop(tmp_path):
    """Valid crop saves successfully and returns a valid path."""
    from src.config import EvidenceConfig
    from src.evidence_storage import EvidenceStorage

    cfg = EvidenceConfig(
        entries_dir=str(tmp_path / "entries"),
        exits_dir=str(tmp_path / "exits"),
    )
    storage = EvidenceStorage(cfg)

    # Create a valid 100x100 BGR image
    crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    ts = datetime.now(timezone.utc)

    result = storage.save_evidence(
        face_id="face-aaaabbbb1234",
        session_id="sess-ccccdddd5678",
        event_type="entry",
        crop=crop,
        timestamp=ts,
        frame_number=42,
    )

    assert result is not None
    assert Path(result).exists()
    assert Path(result).stat().st_size > 0


def test_save_evidence_none_crop(tmp_path):
    """None crop returns None without crashing."""
    from src.config import EvidenceConfig
    from src.evidence_storage import EvidenceStorage

    cfg = EvidenceConfig(
        entries_dir=str(tmp_path / "entries"),
        exits_dir=str(tmp_path / "exits"),
    )
    storage = EvidenceStorage(cfg)

    result = storage.save_evidence(
        face_id="test",
        session_id="test",
        event_type="entry",
        crop=None,
        timestamp=datetime.now(timezone.utc),
        frame_number=1,
    )
    assert result is None


def test_save_evidence_too_small_crop(tmp_path):
    """A 5x5 crop below minimum size returns None."""
    from src.config import EvidenceConfig
    from src.evidence_storage import EvidenceStorage

    cfg = EvidenceConfig(
        entries_dir=str(tmp_path / "entries"),
        exits_dir=str(tmp_path / "exits"),
    )
    storage = EvidenceStorage(cfg)

    tiny_crop = np.zeros((5, 5, 3), dtype=np.uint8)
    result = storage.save_evidence(
        face_id="test",
        session_id="test",
        event_type="exit",
        crop=tiny_crop,
        timestamp=datetime.now(timezone.utc),
        frame_number=1,
    )
    assert result is None


def test_verify_path_exists(tmp_path):
    """verify_path_exists returns True for existing file, False for missing."""
    from src.config import EvidenceConfig
    from src.evidence_storage import EvidenceStorage

    cfg = EvidenceConfig(
        entries_dir=str(tmp_path / "entries"),
        exits_dir=str(tmp_path / "exits"),
    )
    storage = EvidenceStorage(cfg)

    existing = tmp_path / "test.jpg"
    existing.write_bytes(b"fake_jpg")

    assert storage.verify_path_exists(str(existing)) is True
    assert storage.verify_path_exists("/nonexistent/path/file.jpg") is False
