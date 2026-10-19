"""
evidence_storage.py
===================
Manages on-disk storage of face-crop evidence images that are captured at
ENTRY and EXIT events.

Design goals
------------
* **Atomic writes** --- crops are written to a ``.tmp`` file and then renamed
  into place with :func:`os.replace` so a crash never leaves a partial JPEG.
* **Collision-resistant filenames** --- filename encodes event_type, truncated
  face_id, session_id, frame number, and timestamp so two concurrent writers
  cannot overwrite each other.
* **Date-partitioned directories** --- each day gets its own sub-folder under
  the configured entries / exits root, making manual inspection and retention
  cleanup straightforward.
* **Retention cleanup** --- :meth:`cleanup_old_evidence` removes files older
  than *N* days without touching directories that still contain newer files.
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config import get_config
from src.event_logger import EventLogger

logger = logging.getLogger(__name__)


class EvidenceStorage:
    """
    Saves and manages face-crop evidence images for audit purposes.

    Parameters
    ----------
    event_logger : EventLogger
        Structured event logger for audit output.

    Directory layout produced
    -------------------------
    ::

        <evidence.entries_dir>/
            YYYY-MM-DD/
                entry_<face8>_<sess8>_<frame>_<ts>.jpg
                ...
        <evidence.exits_dir>/
            YYYY-MM-DD/
                exit_<face8>_<sess8>_<frame>_<ts>.jpg
                ...

    Notes
    -----
    JPEG quality is read from ``config.evidence.jpeg_quality`` (default 90).
    Directory paths are read from ``config.evidence.entries_dir`` and
    ``config.evidence.exits_dir``.
    """

    # Fallback JPEG encode quality if not specified in config.
    _DEFAULT_JPEG_QUALITY: int = 90
    # Minimum pixel dimension for a crop to be considered valid.
    _MIN_CROP_DIM: int = 10

    def __init__(
        self,
        config_or_logger: Optional[object] = None,
        event_logger: Optional[EventLogger] = None,
    ) -> None:
        if isinstance(config_or_logger, EventLogger):
            self._event_logger = config_or_logger
            evidence_cfg = None
        else:
            evidence_cfg = config_or_logger
            self._event_logger = event_logger or EventLogger()

        if evidence_cfg is None:
            try:
                self._config = get_config()
                evidence_cfg = getattr(self._config, "evidence", None)
            except Exception:
                evidence_cfg = None

        self._entries_dir: Path = Path(
            getattr(evidence_cfg, "entries_dir", "logs/entries")
        )
        self._exits_dir: Path = Path(
            getattr(evidence_cfg, "exits_dir", "logs/exits")
        )
        self._jpeg_quality: int = int(
            getattr(evidence_cfg, "image_quality", getattr(evidence_cfg, "jpeg_quality", self._DEFAULT_JPEG_QUALITY))
        )

        logger.info(
            "EvidenceStorage initialised | entries=%s exits=%s quality=%d",
            self._entries_dir,
            self._exits_dir,
            self._jpeg_quality,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _build_path(
        self,
        event_type: str,
        face_id: str,
        session_id: str,
        frame_number: int,
        timestamp: object,
    ) -> Path:
        if isinstance(timestamp, datetime):
            ts_dt = timestamp.astimezone(timezone.utc) if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        elif isinstance(timestamp, (int, float)):
            ts_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        else:
            ts_dt = datetime.now(timezone.utc)

        date_str = ts_dt.strftime("%Y-%m-%d")
        base_dir = self._entries_dir if event_type.lower() == "entry" else self._exits_dir
        target_dir = base_dir / date_str

        ts_str = ts_dt.strftime("%H%M%S%f")[:12]
        safe_event = "".join(c if c.isalnum() else "_" for c in event_type.lower())
        filename = (
            f"{safe_event}"
            f"_{face_id[:8]}"
            f"_{session_id[:8]}"
            f"_{frame_number:07d}"
            f"_{ts_str}.jpg"
        )
        return target_dir / filename

    def save_evidence(
        self,
        face_id: str,
        session_id: str,
        event_type: str,
        crop: Optional[np.ndarray],
        timestamp: object,
        frame_number: int,
    ) -> Optional[str]:
        # ------------------------------------------------------------------ #
        # 1. Validate crop                                                     #
        # ------------------------------------------------------------------ #
        if not self._is_valid_crop(crop):
            logger.error(
                "save_evidence: invalid crop for face_id=%s frame=%d.",
                face_id,
                frame_number,
            )
            self._event_logger.log(
                event_type="evidence_error",
                face_id=face_id,
                session_id=session_id,
                frame_number=frame_number,
                detail="Invalid or empty crop rejected.",
            )
            return None

        # ------------------------------------------------------------------ #
        # 2. Build path and ensure directory                                   #
        # ------------------------------------------------------------------ #
        final_path = self._build_path(
            event_type=event_type,
            face_id=face_id,
            session_id=session_id,
            frame_number=frame_number,
            timestamp=timestamp,
        )
        target_dir = final_path.parent

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.exception(
                "save_evidence: failed to create directory %s.", target_dir
            )
            self._event_logger.log(
                event_type="evidence_error",
                face_id=face_id,
                session_id=session_id,
                frame_number=frame_number,
                detail=f"Directory creation failed: {target_dir}",
            )
            return None

        # ------------------------------------------------------------------ #
        # 4. Encode JPEG                                                       #
        # ------------------------------------------------------------------ #
        try:
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
            success, encoded = cv2.imencode(".jpg", crop, encode_params)
            if not success or encoded is None:
                raise RuntimeError("cv2.imencode returned success=False")
            image_bytes: bytes = encoded.tobytes()
        except Exception:
            logger.exception(
                "save_evidence: JPEG encoding failed for face_id=%s frame=%d.",
                face_id,
                frame_number,
            )
            self._event_logger.log(
                event_type="evidence_error",
                face_id=face_id,
                session_id=session_id,
                frame_number=frame_number,
                detail="JPEG encoding failure.",
            )
            return None

        # ------------------------------------------------------------------ #
        # 5. Atomic write via temp file --- os.replace                          #
        # ------------------------------------------------------------------ #
        tmp_path: Optional[Path] = None
        try:
            # Write to a temp file in the same directory so os.replace is
            # guaranteed to be atomic (same filesystem / mount point).
            fd, tmp_str = tempfile.mkstemp(dir=str(target_dir), suffix=".tmp")
            tmp_path = Path(tmp_str)
            with os.fdopen(fd, "wb") as fh:
                fh.write(image_bytes)

            os.replace(str(tmp_path), str(final_path))
            tmp_path = None  # Successfully renamed; don't clean up.
        except Exception:
            logger.exception(
                "save_evidence: write/rename failed for %s.", final_path
            )
            self._event_logger.log(
                event_type="evidence_error",
                face_id=face_id,
                session_id=session_id,
                frame_number=frame_number,
                detail=f"Write failure: {final_path}",
            )
            # Clean up temp file if rename didn't happen.
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            return None

        # ------------------------------------------------------------------ #
        # 6. Validate the written file                                         #
        # ------------------------------------------------------------------ #
        try:
            file_size = final_path.stat().st_size
            if file_size == 0:
                raise ValueError("Written file has zero size.")
        except Exception:
            logger.exception(
                "save_evidence: post-write validation failed for %s.", final_path
            )
            return None

        abs_path = str(final_path.resolve())
        self._event_logger.log(
            event_type="evidence_saved",
            face_id=face_id,
            session_id=session_id,
            frame_number=frame_number,
            detail=f"Saved {file_size} bytes --- {abs_path}",
        )
        logger.info(
            "Evidence saved | face_id=%s frame=%d size=%d path=%s",
            face_id,
            frame_number,
            file_size,
            abs_path,
        )
        return abs_path

    # ------------------------------------------------------------------

    def verify_path_exists(self, image_path: str) -> bool:
        """
        Check whether a path that was stored in the database actually exists
        on disk.

        Useful for integrity audits --- call this before displaying or serving
        an evidence image.

        Parameters
        ----------
        image_path : str
            Absolute or relative path recorded in the database.

        Returns
        -------
        bool
            ``True`` if the file exists and is a regular file.
        """
        if not image_path:
            return False
        try:
            p = Path(image_path)
            return p.is_file()
        except (TypeError, ValueError, OSError):
            logger.debug(
                "verify_path_exists: path check raised for %r.", image_path
            )
            return False

    # ------------------------------------------------------------------

    def cleanup_old_evidence(self, retention_days: int) -> dict[str, int]:
        """
        Delete evidence files that are older than *retention_days* days.

        The method walks both the entries and exits directory trees.  Only
        regular ``.jpg`` files are deleted; directories are never removed
        (even empty ones), so the date structure is preserved.

        Parameters
        ----------
        retention_days : int
            Files whose modification time is older than this many days will
            be deleted.  Must be --- 1.

        Returns
        -------
        dict
            ``{'deleted': int, 'errors': int, 'skipped': int}`` counts.

        Raises
        ------
        ValueError
            If *retention_days* is less than 1.
        """
        if retention_days < 1:
            raise ValueError(f"retention_days must be >= 1, got {retention_days}.")

        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=retention_days)
        cutoff_ts = cutoff.timestamp()

        counts = {"deleted": 0, "errors": 0, "skipped": 0}

        for root_dir in (self._entries_dir, self._exits_dir):
            if not root_dir.exists():
                logger.debug("cleanup_old_evidence: %s does not exist, skipping.", root_dir)
                continue

            for path in root_dir.rglob("*.jpg"):
                if not path.is_file():
                    counts["skipped"] += 1
                    continue
                try:
                    mtime = path.stat().st_mtime
                    if mtime < cutoff_ts:
                        path.unlink()
                        counts["deleted"] += 1
                        logger.debug("Deleted old evidence: %s", path)
                    else:
                        counts["skipped"] += 1
                except Exception:
                    logger.exception("cleanup_old_evidence: failed to process %s.", path)
                    counts["errors"] += 1

        logger.info(
            "Evidence cleanup complete | retention=%dd deleted=%d errors=%d skipped=%d",
            retention_days,
            counts["deleted"],
            counts["errors"],
            counts["skipped"],
        )
        self._event_logger.log(
            event_type="evidence_cleanup",
            detail=(
                f"retention={retention_days}d "
                f"deleted={counts['deleted']} "
                f"errors={counts['errors']} "
                f"skipped={counts['skipped']}"
            ),
        )
        return counts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_valid_crop(self, crop: object) -> bool:
        """
        Return ``True`` if *crop* is a non-empty numpy ndarray with sane
        dimensions.

        Parameters
        ----------
        crop : object
            The value to validate.

        Returns
        -------
        bool
        """
        if crop is None:
            return False
        if not isinstance(crop, np.ndarray):
            return False
        if crop.ndim not in (2, 3):
            return False
        h, w = crop.shape[:2]
        if h < self._MIN_CROP_DIM or w < self._MIN_CROP_DIM:
            return False
        if crop.size == 0:
            return False
        return True

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<EvidenceStorage entries={self._entries_dir} "
            f"exits={self._exits_dir} quality={self._jpeg_quality}>"
        )


