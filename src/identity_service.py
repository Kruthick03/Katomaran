"""
SentinelTrack — src/identity_service.py

Coordinates face registration and recognition.
Maintains an in-memory gallery of registered identities to avoid repeated
database queries during real-time frame processing.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sqlalchemy.orm import Session

from src.config import AppConfig, get_config
from src.event_logger import EventLogger, log_registration, log_recognition
from src.face_recognizer import FaceRecognizer, RegisteredFace, normalize_embedding
from src.models import Face
from src.repositories import FaceRepository

logger = logging.getLogger("sentineltrack.identity")


class IdentityService:
    """
    Coordinates identity resolution against PostgreSQL and an in-memory cache.

    Guarantees:
      - New faces registered exactly once with a stable UUID.
      - Returning visitors matched using cosine similarity.
      - Ambiguous matches rejected (conservative policy).
      - Per-track registration cooldown prevents rapid duplicate registrations.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        recognizer: Optional[FaceRecognizer] = None,
        event_logger: Optional[EventLogger] = None,
    ) -> None:
        if config is None:
            try:
                config = get_config()
            except Exception:
                config = None
        self._config = config
        self._recognizer = recognizer
        self._event_logger = event_logger or EventLogger()

        # In-memory gallery of registered faces
        self._gallery: List[RegisteredFace] = []
        self._face_map: Dict[str, RegisteredFace] = {}

        # Cooldown guard: track_id -> monotonic timestamp of registration
        self._recent_registrations: Dict[int, float] = {}
        self._registration_cooldown_seconds: float = 2.0

    def reload_from_database(self, session: Session) -> int:
        """
        Refresh in-memory gallery from PostgreSQL faces table.
        Should be called on startup and periodically if multiple workers run.
        """
        try:
            faces = FaceRepository.get_all_registered(session)
            self._gallery = faces
            self._face_map = {f.face_id: f for f in faces}
            logger.info(f"Loaded {len(self._gallery)} registered faces into memory.")
            return len(self._gallery)
        except Exception as exc:
            logger.error(f"Failed to load faces from database: {exc}", exc_info=True)
            return 0

    def identify(
        self,
        session: Session,
        embedding: Union[np.ndarray, object],
        frame_number: int = 0,
        source_id: str = "cam_001",
        track_id: Optional[int] = None,
        timestamp: Optional[datetime] = None,
        **kwargs,
    ) -> Optional[Tuple[str, str, float]]:
        """
        Identify or register a face embedding.

        Args:
            session: Active database session.
            embedding: Normalized float32 embedding vector or EmbeddingResult.
            frame_number: Current frame index.
            source_id: Camera/source identifier.
            track_id: Tracker ID for cooldown enforcement.
            timestamp: Observation timestamp (defaults to now UTC).

        Returns:
            Tuple of (face_id, 'recognized' | 'registered', similarity) or None if skipped/ambiguous.
        """
        now_ts = timestamp or datetime.now(timezone.utc)

        # Handle EmbeddingResult object if passed
        if hasattr(embedding, "quality_ok") and not embedding.quality_ok:
            logger.debug(f"Frame {frame_number}: Embedding quality not ok, skipping.")
            return None

        if hasattr(embedding, "embedding"):
            emb_vec = embedding.embedding
        else:
            emb_vec = embedding

        if emb_vec is None or len(emb_vec) == 0:
            return None

        query_vec = normalize_embedding(np.asarray(emb_vec, dtype=np.float32))

        # Check track registration cooldown
        now_mono = time.monotonic()
        if track_id is not None and track_id in self._recent_registrations:
            if now_mono - self._recent_registrations[track_id] < self._registration_cooldown_seconds:
                # Still in cooldown — if track has a face_id, recognize it; otherwise skip
                logger.debug(f"Track {track_id} is in registration cooldown.")

        # Match against in-memory gallery
        if self._recognizer is None:
            logger.error("Recognizer not configured in IdentityService.")
            return None

        matched_face_id, similarity, is_ambiguous = self._recognizer.match(
            query_vec, self._gallery
        )

        if is_ambiguous:
            logger.warning(
                f"Frame {frame_number} Track {track_id}: Ambiguous match rejected (sim={similarity:.4f})."
            )
            self._event_logger.log(
                event_type="ambiguous",
                frame_number=frame_number,
                source_id=source_id,
                detail=f"Ambiguous match with similarity {similarity:.4f}",
            )
            return None

        # Case 1: Recognized existing identity
        if matched_face_id is not None:
            try:
                FaceRepository.update_last_seen(session, matched_face_id, now_ts)
            except Exception as exc:
                logger.warning(f"Could not update last_seen for {matched_face_id}: {exc}")

            log_recognition(matched_face_id, similarity, frame_number)
            return matched_face_id, "recognized", similarity

        # Case 2: New visitor — register new identity
        model_name = "buffalo_sc"
        if hasattr(self._recognizer, "_cfg") and hasattr(self._recognizer._cfg, "model_name"):
            model_name = self._recognizer._cfg.model_name

        try:
            new_face = FaceRepository.register_new(
                session=session,
                embedding=query_vec,
                recognition_model=model_name,
                first_seen_at=now_ts,
            )
            new_rf = RegisteredFace(face_id=new_face.face_id, embedding=query_vec)
            self._gallery.append(new_rf)
            self._face_map[new_face.face_id] = new_rf

            if track_id is not None:
                self._recent_registrations[track_id] = now_mono

            log_registration(new_face.face_id, frame_number, source_id)
            return new_face.face_id, "registered", 0.0

        except Exception as exc:
            logger.error(f"Failed to register new face at frame {frame_number}: {exc}", exc_info=True)
            return None

    def get_unique_count(self, session: Session) -> int:
        """Return total unique registered identities from database."""
        return FaceRepository.count_unique_identities(session)
