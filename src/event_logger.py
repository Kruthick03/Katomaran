"""
SentinelTrack --- src/event_logger.py

Mandatory events.log writer.
Wraps Python's standard logging module with:
  - Rotating file handler (events.log)
  - Console handler
  - Structured log messages for every major pipeline event
  - Safe handling of logging failures

Log categories (used as 'event' field in structured messages):
  REGISTRATION, RECOGNITION, TRACKING, ENTRY, EXIT,
  EMBEDDING_GENERATION, ERROR, STARTUP, SHUTDOWN, METRICS
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional


# --------- Module-level logger ------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Other modules import `get_logger()` or use module-level `logger`.

_APP_LOGGER_NAME = "sentineltrack"
_initialized = False


def setup_logging(
    level: str = "INFO",
    log_file: str = "logs/events.log",
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
    console_output: bool = True,
) -> logging.Logger:
    """
    Configure the application-wide rotating logger.
    Must be called once during application startup (main.py).

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to the rotating events.log file.
        max_bytes: Maximum size of a single log file before rotation.
        backup_count: Number of backup log files to retain.
        console_output: Whether to also print logs to stdout.

    Returns:
        Configured root application logger.
    """
    global _initialized

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(_APP_LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid adding duplicate handlers if called multiple times
    if _initialized:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Rotating file handler
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError as exc:
        # Cannot open log file --- print to stderr but continue
        print(f"[WARN] Could not open log file {log_path}: {exc}", file=sys.stderr)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)

    _initialized = True
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a child logger of the application root logger.

    Args:
        name: Sub-logger name (e.g., 'detector', 'recognizer').
              Defaults to the application root logger.

    Returns:
        Logger instance.
    """
    if name:
        return logging.getLogger(f"{_APP_LOGGER_NAME}.{name}")
    return logging.getLogger(_APP_LOGGER_NAME)


# --------- Structured event helpers ---------------------------------------------------------------------------------------------------------------------------------------------------
# Each function emits a log line with a standardized prefix for easy grep.

def log_registration(face_id: str, frame_number: int, source_id: str) -> None:
    """Log a new identity registration event."""
    get_logger("identity").info(
        f"[REGISTRATION] face_id={face_id} frame={frame_number} source={source_id}"
    )


def log_recognition(face_id: str, similarity: float, frame_number: int) -> None:
    """Log a returning identity recognition event."""
    get_logger("identity").info(
        f"[RECOGNITION] face_id={face_id} similarity={similarity:.4f} frame={frame_number}"
    )


def log_tracking(track_id: int, face_id: Optional[str], frame_number: int) -> None:
    """Log a tracker update."""
    get_logger("tracker").debug(
        f"[TRACKING] track_id={track_id} face_id={face_id or 'unknown'} frame={frame_number}"
    )


def log_entry(face_id: str, session_id: str, timestamp: str, source_id: str) -> None:
    """Log a confirmed entry event."""
    get_logger("session").info(
        f"[ENTRY] face_id={face_id} session_id={session_id} "
        f"timestamp={timestamp} source={source_id}"
    )


def log_exit(face_id: str, session_id: str, timestamp: str, source_id: str) -> None:
    """Log a confirmed exit event."""
    get_logger("session").info(
        f"[EXIT] face_id={face_id} session_id={session_id} "
        f"timestamp={timestamp} source={source_id}"
    )


def log_embedding_generation(face_id: str, dimension: int, frame_number: int) -> None:
    """Log embedding generation (no raw embedding values)."""
    get_logger("recognizer").debug(
        f"[EMBEDDING_GENERATION] face_id={face_id} dim={dimension} frame={frame_number}"
    )


def log_error(component: str, message: str, exc: Optional[Exception] = None) -> None:
    """Log a critical error with optional exception details."""
    logger = get_logger(component)
    if exc:
        logger.error(f"[ERROR] {message}", exc_info=exc)
    else:
        logger.error(f"[ERROR] {message}")


def log_startup(config_path: str, source_type: str) -> None:
    """Log application startup."""
    get_logger().info(
        f"[STARTUP] SentinelTrack starting | config={config_path} source_type={source_type}"
    )


def log_shutdown(processed_frames: int, unique_identities: int, sessions: int) -> None:
    """Log application shutdown summary."""
    get_logger().info(
        f"[SHUTDOWN] frames_processed={processed_frames} "
        f"unique_identities={unique_identities} sessions={sessions}"
    )


class EventLogger:
    """
    Object-oriented structured logger for pipeline events.
    Supports dependency injection and standalone usage.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or get_logger()

    def log(
        self,
        event_type: str,
        face_id: Optional[str] = None,
        session_id: Optional[str] = None,
        frame_number: Optional[int] = None,
        detail: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Generic event logger."""
        parts = [f"[{event_type.upper()}]"]
        if face_id:
            parts.append(f"face_id={face_id}")
        if session_id:
            parts.append(f"session_id={session_id}")
        if frame_number is not None:
            parts.append(f"frame={frame_number}")
        if detail:
            parts.append(f"detail={detail}")
        for k, v in kwargs.items():
            parts.append(f"{k}={v}")
        self._logger.info(" ".join(parts))

    def log_entry(
        self,
        face_id: str,
        session_id: str,
        frame_number: Optional[int] = None,
        timestamp: Optional[object] = None,
        evidence_path: Optional[str] = None,
        source_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        ts_str = str(timestamp) if timestamp is not None else ""
        parts = [
            f"[ENTRY] face_id={face_id} session_id={session_id}",
        ]
        if frame_number is not None:
            parts.append(f"frame={frame_number}")
        if ts_str:
            parts.append(f"timestamp={ts_str}")
        if source_id:
            parts.append(f"source={source_id}")
        if evidence_path:
            parts.append(f"evidence={evidence_path}")
        get_logger("session").info(" ".join(parts))

    def log_exit(
        self,
        face_id: str,
        session_id: str,
        frame_number: Optional[int] = None,
        timestamp: Optional[object] = None,
        duration: Optional[float] = None,
        source_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        ts_str = str(timestamp) if timestamp is not None else ""
        parts = [
            f"[EXIT] face_id={face_id} session_id={session_id}",
        ]
        if frame_number is not None:
            parts.append(f"frame={frame_number}")
        if ts_str:
            parts.append(f"timestamp={ts_str}")
        if duration is not None:
            parts.append(f"duration={duration:.1f}s")
        if source_id:
            parts.append(f"source={source_id}")
        get_logger("session").info(" ".join(parts))

    def log_presence(
        self,
        face_id: str,
        session_id: str,
        frame_number: Optional[int] = None,
        timestamp: Optional[object] = None,
        **kwargs,
    ) -> None:
        get_logger("session").debug(
            f"[PRESENCE] face_id={face_id} session_id={session_id} frame={frame_number}"
        )

    def log_registration(
        self,
        face_id: str,
        frame_number: Optional[int] = None,
        source_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        log_registration(face_id, frame_number or 0, source_id or "unknown")

    def log_recognition(
        self,
        face_id: str,
        similarity: float,
        frame_number: Optional[int] = None,
        **kwargs,
    ) -> None:
        log_recognition(face_id, similarity, frame_number or 0)

    def log_tracking(
        self,
        track_id: int,
        face_id: Optional[str] = None,
        frame_number: Optional[int] = None,
        **kwargs,
    ) -> None:
        log_tracking(track_id, face_id, frame_number or 0)

    def log_recovery(
        self,
        face_id: str,
        session_id: str,
        timestamp: Optional[object] = None,
        reason: str = "recovery",
        **kwargs,
    ) -> None:
        get_logger("recovery").warning(
            f"[RECOVERY] face_id={face_id} session_id={session_id} reason={reason}"
        )

    def log_error(
        self, component: str, message: str, exc: Optional[Exception] = None
    ) -> None:
        log_error(component, message, exc)


