"""
SentinelTrack --- src/video_input.py

Unified video source layer supporting:
  A. Local video files (MP4, AVI, etc.)
  B. RTSP streams with bounded reconnect logic

Both source types expose the same interface (VideoSource base class)
so downstream modules do not need to know which source is active.

Key design decisions:
  - RTSP passwords are read from RTSP_USERNAME/RTSP_PASSWORD env vars
    and injected into the URI at runtime; they are never logged.
  - Reconnect attempts are bounded by config.rtsp.reconnect_limit.
  - Frame read failures on RTSP are counted as dropped frames and
    trigger a reconnect attempt after a configurable threshold.
  - End-of-file on a local video raises StopIteration (handled in main).
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

from src.config import RTSPConfig, SourceConfig

logger = logging.getLogger("sentineltrack.video_input")

# Type alias for a single video frame
Frame = np.ndarray
FrameResult = Tuple[bool, Optional[Frame], int]  # (success, frame, frame_number)


# --------- Abstract base ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class VideoSource(ABC):
    """
    Abstract video source interface.
    Concrete implementations: FileVideoSource, RTSPVideoSource.
    """

    @abstractmethod
    def open(self) -> None:
        """Open the video source. Raises on failure."""
        ...

    @abstractmethod
    def read_frame(self) -> FrameResult:
        """
        Read the next frame.

        Returns:
            Tuple (success, frame_or_None, frame_number).
            success=False and frame=None means source ended or error.
        """
        ...

    @abstractmethod
    def release(self) -> None:
        """Release all resources associated with this source."""
        ...

    @abstractmethod
    def is_open(self) -> bool:
        """Return True if the source is currently open and readable."""
        ...

    def frame_generator(self) -> Generator[Tuple[Frame, int], None, None]:
        """
        Convenience generator that yields (frame, frame_number) tuples.
        Stops when read_frame() returns success=False.
        """
        while True:
            success, frame, frame_num = self.read_frame()
            if not success or frame is None:
                break
            yield frame, frame_num


# --------- File source ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class FileVideoSource(VideoSource):
    """
    Reads frames sequentially from a local video file.
    Raises FileNotFoundError if the path does not exist.
    """

    def __init__(self, config: SourceConfig) -> None:
        self._path = config.video_path
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_number: int = 0

    def open(self) -> None:
        """Open the video file. Raises FileNotFoundError if missing."""
        if not os.path.exists(self._path):
            raise FileNotFoundError(
                f"Video file not found: {self._path}\n"
                "Place a sample video at data/sample.mp4 or update config.json."
            )
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"OpenCV could not open video file: {self._path}"
            )
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(
            f"Opened video file: {self._path} | fps={fps:.1f} | total_frames={total}"
        )

    def read_frame(self) -> FrameResult:
        """Read the next frame from the file."""
        if self._cap is None or not self._cap.isOpened():
            return False, None, self._frame_number

        ret, frame = self._cap.read()
        if not ret:
            logger.info(f"End of video file reached at frame {self._frame_number}.")
            return False, None, self._frame_number

        self._frame_number += 1
        return True, frame, self._frame_number

    def release(self) -> None:
        """Release the OpenCV capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("FileVideoSource released.")

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        if self._cap is None:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS) or 25.0


# --------- RTSP source ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class RTSPVideoSource(VideoSource):
    """
    Reads frames from an RTSP stream with bounded reconnect logic.

    Password injection:
        If RTSP_USERNAME and RTSP_PASSWORD environment variables are set,
        they are injected into the URI at runtime (e.g., rtsp://user:pass@host/path).
        Never log the constructed URI.

    Reconnect policy:
        On connection loss, waits reconnect_delay_seconds and retries.
        Stops after reconnect_limit failures. Raises RuntimeError if exhausted.
    """

    def __init__(self, source_cfg: SourceConfig, rtsp_cfg: RTSPConfig) -> None:
        self._raw_uri = source_cfg.rtsp_uri
        self._rtsp_cfg = rtsp_cfg
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_number: int = 0
        self._reconnect_count: int = 0
        self._consecutive_failures: int = 0
        self._MAX_CONSECUTIVE_FAILURES = 10  # Before triggering reconnect

    def _build_uri(self) -> str:
        """Inject credentials from environment into the RTSP URI."""
        username = os.getenv("RTSP_USERNAME", "")
        password = os.getenv("RTSP_PASSWORD", "")
        uri = self._raw_uri

        if username and password and "://" in uri:
            protocol, rest = uri.split("://", 1)
            # Remove any existing credentials from URI
            if "@" in rest:
                rest = rest.split("@", 1)[1]
            uri = f"{protocol}://{username}:{password}@{rest}"

        return uri

    def open(self) -> None:
        """Open the RTSP stream."""
        self._connect()

    def _connect(self) -> None:
        """Attempt to open/reopen the RTSP stream."""
        if self._cap is not None:
            self._cap.release()

        uri = self._build_uri()
        # Log only the host part, never credentials
        safe_log = self._raw_uri.split("@")[-1] if "@" in self._raw_uri else self._raw_uri

        logger.info(f"Connecting to RTSP source: ...@{safe_log}")
        self._cap = cv2.VideoCapture(uri)

        # Set OpenCV buffer and timeout
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            raise RuntimeError(f"Could not connect to RTSP source: ...@{safe_log}")

        self._consecutive_failures = 0
        logger.info("RTSP connection established.")

    def read_frame(self) -> FrameResult:
        """
        Read from RTSP with automatic reconnect on failure.
        Returns (False, None, frame_num) when reconnect limit is exhausted.
        """
        if self._cap is None or not self._cap.isOpened():
            return self._try_reconnect()

        ret, frame = self._cap.read()
        if not ret or frame is None:
            self._consecutive_failures += 1
            logger.warning(
                f"RTSP frame read failed (consecutive={self._consecutive_failures})"
            )
            if self._consecutive_failures >= self._MAX_CONSECUTIVE_FAILURES:
                return self._try_reconnect()
            return False, None, self._frame_number

        self._consecutive_failures = 0
        self._frame_number += 1
        return True, frame, self._frame_number

    def _try_reconnect(self) -> FrameResult:
        """Attempt to reconnect to the RTSP stream."""
        if self._reconnect_count >= self._rtsp_cfg.reconnect_limit:
            logger.error(
                f"RTSP reconnect limit reached ({self._rtsp_cfg.reconnect_limit}). "
                "Stopping stream."
            )
            return False, None, self._frame_number

        self._reconnect_count += 1
        logger.info(
            f"RTSP reconnect attempt {self._reconnect_count}/"
            f"{self._rtsp_cfg.reconnect_limit} --- "
            f"waiting {self._rtsp_cfg.reconnect_delay_seconds}s"
        )
        time.sleep(self._rtsp_cfg.reconnect_delay_seconds)

        try:
            self._connect()
            return self.read_frame()
        except Exception as exc:
            logger.error(f"Reconnect failed: {exc}")
            return False, None, self._frame_number

    def release(self) -> None:
        """Release the RTSP capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info(
                f"RTSPVideoSource released. Total reconnects: {self._reconnect_count}"
            )

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count


# --------- Factory ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def create_video_source(config) -> VideoSource:
    """
    Factory function --- creates the appropriate VideoSource from config.

    Args:
        config: AppConfig instance.

    Returns:
        FileVideoSource or RTSPVideoSource.
    """
    if config.source.type == "rtsp":
        logger.info("Creating RTSP video source.")
        return RTSPVideoSource(config.source, config.rtsp)
    else:
        logger.info(f"Creating file video source: {config.source.video_path}")
        return FileVideoSource(config.source)

