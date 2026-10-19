"""
SentinelTrack — tests/unit/test_config.py
Unit tests for configuration loading and validation.
"""
import json
import os
import pytest
from pathlib import Path


def _write_config(tmp_path: Path, data: dict) -> Path:
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(data))
    return cfg_file


def _minimal_config() -> dict:
    return {
        "source": {"type": "file", "video_path": "data/sample.mp4"},
    }


def test_load_minimal_config(tmp_path):
    """Config loads with minimal fields, defaults fill the rest."""
    cfg_file = _write_config(tmp_path, _minimal_config())
    os.chdir(tmp_path)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.config import load_config

    cfg = load_config(str(cfg_file))
    assert cfg.source.type == "file"
    assert cfg.detection.detection_frame_skip >= 1


def test_missing_config_file():
    """Loading a non-existent config raises FileNotFoundError."""
    from src.config import load_config
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.json")


def test_rtsp_requires_uri(tmp_path):
    """RTSP source without rtsp_uri should fail validation."""
    from pydantic import ValidationError
    from src.config import AppConfig
    with pytest.raises((ValidationError, ValueError)):
        AppConfig.model_validate({
            "source": {"type": "rtsp", "rtsp_uri": ""}
        })


def test_frame_skip_range(tmp_path):
    """detection_frame_skip must be between 1 and 60."""
    from pydantic import ValidationError
    from src.config import AppConfig

    with pytest.raises(ValidationError):
        AppConfig.model_validate({
            "detection": {"detection_frame_skip": 0}
        })

    with pytest.raises(ValidationError):
        AppConfig.model_validate({
            "detection": {"detection_frame_skip": 100}
        })


def test_confidence_threshold_range():
    """Confidence thresholds must be in [0.01, 1.0]."""
    from pydantic import ValidationError
    from src.config import AppConfig
    with pytest.raises(ValidationError):
        AppConfig.model_validate({
            "detection": {"confidence_threshold": 1.5}
        })
    with pytest.raises(ValidationError):
        AppConfig.model_validate({
            "detection": {"confidence_threshold": 0.0}
        })
