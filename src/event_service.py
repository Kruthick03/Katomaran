"""
SentinelTrack — src/event_service.py

Orchestrates the per-frame processing pipeline for tracked faces.
Coordinates FaceRecognizer, IdentityService, VisitSessionManager, and EvidenceStorage.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Set

import numpy as np
from sqlalchemy.orm import Session

from src.config import AppConfig, get_config
from src.event_logger import EventLogger
from src.evidence_storage import EvidenceStorage
from src.face_detector import FaceDetection
from src.face_recognizer import FaceRecognizer
from src.identity_service import IdentityService
from src.tracker import Track
from src.visit_session import VisitSessionManager

logger = logging.getLogger("sentineltrack.event_service")


class EventService:
    """
    Coordinates recognition, identity assignment, and visit sessions
    for active tracks in each frame.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        recognizer: Optional[FaceRecognizer] = None,
        identity_service: Optional[IdentityService] = None,
        session_manager: Optional[VisitSessionManager] = None,
        evidence_storage: Optional[EvidenceStorage] = None,
        event_logger: Optional[EventLogger] = None,
        metrics: Optional[Any] = None,
    ) -> None:
        if config is None:
            try:
                config = get_config()
            except Exception:
                config = None

        self._config = config
        self._recognizer = recognizer
        self._identity_service = identity_service
        self._session_manager = session_manager
        self._evidence_storage = evidence_storage
        self._event_logger = event_logger or EventLogger()
        self._metrics = metrics

        if self._session_manager and self._evidence_storage:
            self._session_manager.set_evidence_storage(self._evidence_storage)

        self._source_id = "cam_001"
        if config and hasattr(config, "source") and hasattr(config.source, "source_id"):
            self._source_id = config.source.source_id

        self._min_track_age = 2
        if config and hasattr(config, "tracker") and hasattr(config.tracker, "min_track_age_frames"):
            self._min_track_age = config.tracker.min_track_age_frames

    def process_tracked_face(
        self,
        db_session: Session,
        frame: np.ndarray,
        track: Track,
        frame_number: int,
        timestamp: datetime,
    ) -> bool:
        """
        Process an active track through recognition, identification, and session tracking.

        Returns:
            True if processed successfully, False if skipped or errored.
        """
        # Step 1: Check track age gate
        if track.age < self._min_track_age:
            return False

        # Step 2: Build detection object
        detection = FaceDetection(
            bbox=track.bbox,
            confidence=track.confidence,
            frame_number=frame_number,
        )

        # Step 3: Extract embedding via ArcFace
        if self._recognizer is None:
            return False

        emb_result = self._recognizer.generate_embedding(frame, detection)
        if emb_result is None or not emb_result.quality_ok:
            return False

        # Step 4: Identify or register identity
        if track.face_id is not None:
            # Identity is already confirmed for this continuous track.
            # Optionally update similarity if a high-quality embedding is extracted:
            if emb_result is not None and emb_result.quality_ok and self._identity_service is not None:
                ident_result = self._identity_service.identify(
                    session=db_session,
                    embedding=emb_result.embedding,
                    frame_number=frame_number,
                    source_id=self._source_id,
                    track_id=track.track_id,
                    timestamp=timestamp,
                )
                if ident_result is not None:
                    matched_id, status, similarity = ident_result
                    if matched_id == track.face_id:
                        track.similarity = similarity

            # Always maintain session continuity for the active face
            if self._session_manager is not None:
                crop_to_save = emb_result.crop if (emb_result and emb_result.quality_ok) else None
                session_id, is_new_entry = self._session_manager.on_face_seen(
                    db_session=db_session,
                    face_id=track.face_id,
                    source_id=self._source_id,
                    frame_number=frame_number,
                    timestamp=timestamp,
                    crop=crop_to_save,
                )
                if is_new_entry and self._metrics is not None:
                    self._metrics.increment_entries()
            return True

        # Unidentified track: perform initial identification or registration
        if self._identity_service is None:
            return False

        ident_result = self._identity_service.identify(
            session=db_session,
            embedding=emb_result.embedding,
            frame_number=frame_number,
            source_id=self._source_id,
            track_id=track.track_id,
            timestamp=timestamp,
        )

        if ident_result is None:
            return False

        face_id, status, similarity = ident_result
        track.face_id = face_id
        track.recognition_status = "New identity" if status == "registered" else ("Recognized" if status == "recognized" else "Unknown")
        track.similarity = similarity

        if self._metrics is not None:
            if status == "registered":
                self._metrics.increment_registrations()
            elif status == "recognized":
                self._metrics.increment_recognitions()

        # Step 5: Update or create visit session
        if self._session_manager is not None:
            session_id, is_new_entry = self._session_manager.on_face_seen(
                db_session=db_session,
                face_id=face_id,
                source_id=self._source_id,
                frame_number=frame_number,
                timestamp=timestamp,
                crop=emb_result.crop,
            )
            if is_new_entry and self._metrics is not None:
                self._metrics.increment_entries()

        return True

    def process_session_exits(
        self,
        db_session: Session,
        active_face_ids: Set[str],
        frame_number: int,
        timestamp: datetime,
    ) -> Dict[str, Any]:
        """
        Notify session manager of absent faces to trigger exit confirmation.
        """
        if self._session_manager is None:
            return {"exits_triggered": 0}

        res = self._session_manager.on_faces_absent(
            db_session=db_session,
            seen_face_ids=active_face_ids,
            frame_number=frame_number,
            timestamp=timestamp,
        )
        if self._metrics is not None and res.get("exits_triggered"):
            self._metrics.increment_exits(res["exits_triggered"])
        return res
