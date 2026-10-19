"""
SentinelTrack --- src/config.py

Configuration loading and validation using Pydantic.
All database credentials are read exclusively from environment variables.
Settings are validated on startup; the application refuses to run with
invalid configuration rather than silently using unsafe defaults.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# --------- Sub-models ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class SourceConfig(BaseModel):
    """Video source configuration."""
    type: Literal["file", "rtsp"] = "file"
    video_path: str = "data/sample.mp4"
    rtsp_uri: str = ""
    source_id: str = "cam_001"
    source_name: str = "Main Entrance"

    @field_validator("rtsp_uri")
    @classmethod
    def rtsp_uri_no_password(cls, v: str) -> str:
        """Warn if password appears embedded in RTSP URI (insecure)."""
        if v and "@" in v and "://" in v:
            # Check for password in URI --- log warning but don't block
            import warnings
            warnings.warn(
                "RTSP URI contains credentials. Use RTSP_USERNAME/RTSP_PASSWORD "
                "environment variables instead and set rtsp_uri without credentials.",
                UserWarning,
                stacklevel=2,
            )
        return v


class DetectionConfig(BaseModel):
    """YOLO face detector configuration."""
    model_config = ConfigDict(protected_namespaces=())

    model_path: str = "models/yolov8n-face.pt"
    confidence_threshold: float = Field(0.25, ge=0.01, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.01, le=1.0)
    detection_frame_skip: int = Field(2, ge=1, le=60,
        description="Detect faces every N frames. Tracking runs every frame.")
    min_face_width: int = Field(16, ge=10)
    min_face_height: int = Field(16, ge=10)
    input_size: int = Field(640, ge=64)


class RecognitionConfig(BaseModel):
    """InsightFace/ArcFace recognition configuration."""
    model_config = ConfigDict(protected_namespaces=())

    model_name: str = "buffalo_sc"
    similarity_threshold: float = Field(0.45, ge=0.0, le=1.0,
        description="Cosine similarity threshold. Lower = stricter (fewer false matches).")
    ambiguity_margin: float = Field(0.08, ge=0.0, le=1.0,
        description="If top-2 matches are within this margin, treat as ambiguous.")
    embedding_dimension: int = Field(512, ge=128)
    min_detection_score: float = Field(0.35, ge=0.0, le=1.0)
    normalize_embeddings: bool = True


class TrackerConfig(BaseModel):
    """Tracker configuration for ByteTrack-style tracking."""
    max_missed_frames: int = Field(30, ge=1,
        description="Frames without detection before track is dropped.")
    min_track_age_frames: int = Field(2, ge=1,
        description="Minimum track age before recognition is attempted.")
    iou_match_threshold: float = Field(0.25, ge=0.0, le=1.0)
    max_age: int = Field(300, ge=1)


class SessionConfig(BaseModel):
    """Visit-session state-machine configuration."""
    exit_confirmation_frames: int = Field(20, ge=1,
        description="Frames of absence required to confirm exit.")
    min_session_duration_seconds: float = Field(1.0, ge=0.0)
    reentry_cooldown_seconds: float = Field(5.0, ge=0.0)


class EvidenceConfig(BaseModel):
    """Evidence storage configuration."""
    entries_dir: str = "logs/entries"
    exits_dir: str = "logs/exits"
    image_format: Literal["jpg", "png", "bmp"] = "jpg"
    image_quality: int = Field(90, ge=1, le=100)
    crop_padding_fraction: float = Field(0.25, ge=0.0, le=1.0)
    save_evidence: bool = True
    retention_days: int = Field(90, ge=1)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_file: str = "logs/events.log"
    max_bytes: int = Field(10_485_760, ge=1024)  # 10 MB
    backup_count: int = Field(5, ge=0)
    console_output: bool = True


class RTSPConfig(BaseModel):
    """RTSP connection and retry configuration."""
    reconnect_limit: int = Field(10, ge=0)
    reconnect_delay_seconds: float = Field(3.0, ge=0.0)
    connection_timeout_seconds: float = Field(10.0, ge=1.0)
    read_timeout_seconds: float = Field(5.0, ge=1.0)


class MetricsConfig(BaseModel):
    """Performance metrics configuration."""
    enabled: bool = True
    report_interval_frames: int = Field(100, ge=1)
    log_memory: bool = True
    log_cpu: bool = True


class DatabaseEnvConfig(BaseModel):
    """Maps database credential settings to environment variable names."""
    host_env: str = "SENTINEL_DB_HOST"
    port_env: str = "SENTINEL_DB_PORT"
    name_env: str = "SENTINEL_DB_NAME"
    user_env: str = "SENTINEL_DB_USER"
    password_env: str = "SENTINEL_DB_PASSWORD"
    pool_size: int = Field(5, ge=1)
    max_overflow: int = Field(10, ge=0)
    pool_timeout: int = Field(30, ge=5)
    echo_sql: bool = False


# --------- Root configuration ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

class AppConfig(BaseModel):
    """
    Root application configuration model.
    Loaded from config.json; secrets loaded from environment variables.
    """
    source: SourceConfig = SourceConfig()
    detection: DetectionConfig = DetectionConfig()
    recognition: RecognitionConfig = RecognitionConfig()
    tracker: TrackerConfig = TrackerConfig()
    session: SessionConfig = SessionConfig()
    evidence: EvidenceConfig = EvidenceConfig()
    logging: LoggingConfig = LoggingConfig()
    rtsp: RTSPConfig = RTSPConfig()
    metrics: MetricsConfig = MetricsConfig()
    database: DatabaseEnvConfig = DatabaseEnvConfig()
    headless: bool = False

    @model_validator(mode="after")
    def validate_source(self) -> "AppConfig":
        """Ensure required fields are set based on source type."""
        if self.source.type == "rtsp" and not self.source.rtsp_uri:
            raise ValueError(
                "source.rtsp_uri must be set when source.type is 'rtsp'."
            )
        if self.source.type == "file" and not self.source.video_path:
            raise ValueError(
                "source.video_path must be set when source.type is 'file'."
            )
        return self


# --------- Database URL builder ---------------------------------------------------------------------------------------------------------------------------------------------------------------

def build_database_url(db_cfg: DatabaseEnvConfig) -> str:
    """
    Construct the PostgreSQL DSN from environment variables.
    Raises RuntimeError if any required variable is missing.
    Never returns a URL with hardcoded credentials.
    """
    # Check for full DSN override first
    full_dsn = os.getenv("DATABASE_URL")
    if full_dsn:
        return full_dsn

    missing = []
    values = {}
    for field_name, env_name in [
        ("host", db_cfg.host_env),
        ("port", db_cfg.port_env),
        ("dbname", db_cfg.name_env),
        ("user", db_cfg.user_env),
        ("password", db_cfg.password_env),
    ]:
        val = os.getenv(env_name)
        if not val:
            missing.append(f"{field_name} (env: {env_name})")
        else:
            values[field_name] = val

    if missing:
        raise RuntimeError(
            f"Missing required database environment variables: {missing}. "
            f"Copy .env.example to .env and fill in your credentials."
        )

    return (
        f"postgresql+psycopg2://{values['user']}:{values['password']}"
        f"@{values['host']}:{values['port']}/{values['dbname']}"
    )


# --------- Loader ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_config(config_path: str | Path = "config.json") -> AppConfig:
    """
    Load and validate application configuration from a JSON file.

    Args:
        config_path: Path to config.json (absolute or relative to cwd).

    Returns:
        Validated AppConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If configuration values fail validation.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path.resolve()}\n"
            f"Copy config.example.json to config.json and adjust settings."
        )

    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    # Strip comment keys (keys starting with _)
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}

    config = AppConfig.model_validate(raw)
    return config


# Singleton holder --- populated by main.py on startup
_current_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Return the current global config. Raises if not yet initialized."""
    if _current_config is None:
        raise RuntimeError(
            "Configuration has not been loaded. Call load_config() first."
        )
    return _current_config


def set_config(cfg: AppConfig) -> None:
    """Set the global config singleton (called from main.py)."""
    global _current_config
    _current_config = cfg

