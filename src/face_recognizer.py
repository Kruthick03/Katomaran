"""
SentinelTrack --- src/face_recognizer.py

InsightFace ArcFace face recognition and embedding generation.

Responsibilities:
  1. Load InsightFace (buffalo_sc or similar ArcFace model) on CPU.
  2. Extract a face crop from a detected bounding box.
  3. Generate and L2-normalize a 512-dimensional embedding.
  4. Compare against registered embeddings using cosine similarity.
  5. Return the best match above the configured threshold, or None.

Threshold calibration notes:
  - Cosine similarity 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite.
  - A threshold of ~0.45 on ArcFace buffalo_sc balances security vs convenience.
  - Poor lighting, extreme angles (>45--), and heavy occlusion increase false
    non-matches and may produce similarity scores below threshold.
  - Two very similar-looking people may score above threshold (false match).
  - The ambiguity_margin setting flags cases where top-2 candidates are close.
  - Never use similarity scores as probabilities.

Privacy:
  - Raw embedding bytes are NEVER written to logs.
  - Crops are only written to disk by the evidence_storage module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config import RecognitionConfig
from src.face_detector import FaceDetection

logger = logging.getLogger("sentineltrack.recognizer")


# --------- Data types ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class EmbeddingResult:
    """Result of embedding extraction for one face crop."""
    embedding: np.ndarray       # Normalized float32 array, shape (512,)
    face_id: Optional[str]      # Matched face ID, or None if new
    similarity: float           # Cosine similarity to best match (0.0 if new)
    is_ambiguous: bool          # True if top-2 matches are within ambiguity_margin
    crop: Optional[np.ndarray]  # BGR face crop (for evidence storage)
    quality_ok: bool            # False if crop is too small or has issues


@dataclass
class RegisteredFace:
    """An in-memory record of a registered face (loaded from database)."""
    face_id: str
    embedding: np.ndarray   # Normalized float32, shape (512,)


# --------- Embedding utilities ------------------------------------------------------------------------------------------------------------------------------------------------------------------

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    L2-normalize an embedding vector.
    After normalization, cosine similarity == dot product.

    Args:
        embedding: Raw float32 array from the model.

    Returns:
        Unit-norm float32 array of the same shape.
    """
    norm = np.linalg.norm(embedding)
    if norm < 1e-8:
        logger.warning("Embedding has near-zero norm; returning as-is.")
        return embedding.astype(np.float32)
    return (embedding / norm).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embedding vectors.
    Safely normalizes both inputs and bounds output to [-1.0, 1.0].

    Args:
        a, b: Float32 arrays of the same shape.

    Returns:
        Cosine similarity strictly bounded in [-1.0, 1.0].
    """
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    sim = float(np.dot(a, b) / (norm_a * norm_b))
    return float(np.clip(sim, -1.0, 1.0))


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Serialize a float32 embedding array to bytes for BYTEA storage."""
    return embedding.astype(np.float32).tobytes()


def bytes_to_embedding(data: bytes, dimension: int = 512) -> np.ndarray:
    """Deserialize a float32 embedding from BYTEA bytes."""
    return np.frombuffer(data, dtype=np.float32).reshape(dimension)


def extract_face_crop(
    frame: np.ndarray,
    detection: FaceDetection,
    padding_fraction: float = 0.25,
) -> Optional[np.ndarray]:
    """
    Extract and validate a face crop from a frame with optional padding.

    Args:
        frame: BGR source frame.
        detection: Detected face bounding box.
        padding_fraction: Expand box by this fraction on each side.

    Returns:
        Cropped BGR image, or None if crop is invalid.
    """
    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = detection.bbox

    # Add padding
    pad_w = int((x2 - x1) * padding_fraction)
    pad_h = int((y2 - y1) * padding_fraction)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return crop


# --------- Recognizer ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class FaceRecognizer:
    """
    InsightFace ArcFace-based face recognizer.

    Loads the specified InsightFace model pack on CPU.
    Provides embedding extraction and matching against registered identities.

    InsightFace model packs:
      - 'buffalo_sc': Small, fast, ArcFace-based (recommended for CPU).
      - 'buffalo_l': Larger, more accurate, slower.
    """

    def __init__(self, config: RecognitionConfig) -> None:
        self._cfg = config
        self._app = None  # InsightFace FaceAnalysis app

    def load(self) -> None:
        """
        Load InsightFace model. Downloads model files on first run to
        ~/.insightface/models/<model_name>/.
        """
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "insightface is not installed. Run: pip install insightface onnxruntime"
            ) from exc

        logger.info(
            f"Loading InsightFace model: {self._cfg.model_name} on CPU."
        )
        self._app = FaceAnalysis(
            name=self._cfg.model_name,
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=(640, 640))
        logger.info(f"InsightFace {self._cfg.model_name} loaded successfully.")

    def generate_embedding(
        self,
        frame: np.ndarray,
        detection: FaceDetection,
    ) -> Optional[EmbeddingResult]:
        """
        Generate a normalized ArcFace embedding for a detected face.

        Args:
            frame: Full BGR frame.
            detection: Face detection result.

        Returns:
            EmbeddingResult with embedding and quality flag, or None on failure.
        """
        if self._app is None:
            raise RuntimeError("Recognizer not loaded. Call load() first.")

        crop = extract_face_crop(
            frame, detection,
            padding_fraction=0.1  # Small padding for recognition
        )

        if crop is None or crop.shape[0] < 14 or crop.shape[1] < 14:
            logger.debug(
                f"Face crop too small or invalid at frame {detection.frame_number}."
            )
            return EmbeddingResult(
                embedding=np.zeros(self._cfg.embedding_dimension, dtype=np.float32),
                face_id=None,
                similarity=0.0,
                is_ambiguous=False,
                crop=crop,
                quality_ok=False,
            )

        # Run InsightFace on the crop, fallback to full frame only with IoU matching
        try:
            faces = self._app.get(crop)
            if not faces and frame is not None and frame.shape[0] >= 64 and frame.shape[1] >= 64:
                full_faces = self._app.get(frame)
                if full_faces:
                    # Select face in full frame that specifically overlaps detection.bbox
                    det_box = detection.bbox
                    best_match = None
                    best_iou = 0.0
                    for f in full_faces:
                        f_box = (int(f.bbox[0]), int(f.bbox[1]), int(f.bbox[2]), int(f.bbox[3]))
                        ix1 = max(det_box[0], f_box[0])
                        iy1 = max(det_box[1], f_box[1])
                        ix2 = min(det_box[2], f_box[2])
                        iy2 = min(det_box[3], f_box[3])
                        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                        area_d = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
                        area_f = (f_box[2] - f_box[0]) * (f_box[3] - f_box[1])
                        union = area_d + area_f - inter
                        iou = inter / union if union > 0 else 0.0
                        if iou > best_iou:
                            best_iou = iou
                            best_match = f
                    if best_match and best_iou >= 0.25:
                        faces = [best_match]
        except Exception as exc:
            logger.error(
                f"InsightFace inference failed at frame {detection.frame_number}: {exc}"
            )
            return None

        if not faces:
            logger.debug(
                f"InsightFace found no embedding at frame {detection.frame_number}."
            )
            return EmbeddingResult(
                embedding=np.zeros(self._cfg.embedding_dimension, dtype=np.float32),
                face_id=None,
                similarity=0.0,
                is_ambiguous=False,
                crop=crop,
                quality_ok=False,
            )

        # Use the highest-confidence face detection within the crop
        face = max(faces, key=lambda f: f.det_score)

        if face.det_score < self._cfg.min_detection_score:
            logger.debug(
                f"Face quality too low (score={face.det_score:.3f}) at frame {detection.frame_number}."
            )
            return EmbeddingResult(
                embedding=np.zeros(self._cfg.embedding_dimension, dtype=np.float32),
                face_id=None,
                similarity=0.0,
                is_ambiguous=False,
                crop=crop,
                quality_ok=False,
            )

        raw_emb = face.embedding
        normalized = normalize_embedding(raw_emb)

        # Wider crop for saving as evidence
        evidence_crop = extract_face_crop(
            frame, detection,
            padding_fraction=0.25
        )

        logger.debug(
            f"Embedding generated (dim={len(normalized)}) at frame {detection.frame_number}."
        )

        return EmbeddingResult(
            embedding=normalized,
            face_id=None,     # Will be set by identity_service
            similarity=0.0,   # Will be set by match()
            is_ambiguous=False,
            crop=evidence_crop,
            quality_ok=True,
        )

    def match(
        self,
        query_embedding: np.ndarray,
        registered_faces: List[RegisteredFace],
    ) -> Tuple[Optional[str], float, bool]:
        """
        Find the closest registered identity for a query embedding.

        Args:
            query_embedding: Normalized float32 query vector.
            registered_faces: List of registered identities with embeddings.

        Returns:
            Tuple (face_id_or_None, best_similarity, is_ambiguous).
            face_id is None if no match exceeds the threshold.
            is_ambiguous is True if top-2 similarities are within ambiguity_margin.
        """
        if not registered_faces:
            return None, 0.0, False

        similarities = [
            (rf.face_id, cosine_similarity(query_embedding, rf.embedding))
            for rf in registered_faces
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)

        best_id, best_sim = similarities[0]
        is_ambiguous = False

        # Only check ambiguity among potential candidates (above threshold)
        if best_sim >= self._cfg.similarity_threshold:
            if len(similarities) >= 2:
                second_sim = similarities[1][1]
                # If top candidate matches threshold and second candidate is very close
                if (best_sim - second_sim) < self._cfg.ambiguity_margin:
                    is_ambiguous = True
                    logger.debug(
                        f"Ambiguous match: top1={best_sim:.4f} top2={second_sim:.4f} "
                        f"margin={self._cfg.ambiguity_margin}"
                    )
            return best_id, best_sim, is_ambiguous
        else:
            return None, best_sim, False

    def unload(self) -> None:
        """Release model resources."""
        self._app = None
        logger.info("FaceRecognizer unloaded.")


# --------- Factory ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def create_recognizer(config) -> FaceRecognizer:
    """Create and load a FaceRecognizer from AppConfig."""
    recognizer = FaceRecognizer(config.recognition)
    recognizer.load()
    return recognizer

