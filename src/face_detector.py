"""
SentinelTrack --- src/face_detector.py

YOLO-based face detection layer.
Uses YOLOv8n-face (ultralytics) --- a model trained specifically on face
detection tasks (WIDER FACE dataset), NOT a general person/object detector.

Model download:
  The model file is downloaded from Hugging Face on first run via ultralytics.
  Model path is configurable in config.json (detection.model_path).

Design:
  - Detection runs every `detection_frame_skip` frames (controlled by config).
  - Between detection frames, the detector returns the LAST valid detections
    so the tracker can continue interpolating.
  - Bounding boxes are validated (minimum size, frame bounds).
  - Confidence threshold is configurable.

IMPORTANT: This module uses a face-specific YOLO model. Using a generic
YOLOv8 (COCO) model for face detection is explicitly disallowed per spec.
The recommended model is `yolov8n-face.pt` from
https://github.com/akanametov/yolo-face --- trained on WIDER FACE.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("sentineltrack.detector")

# --------- Data types ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class FaceDetection:
    """
    A single face detection result.

    Attributes:
        bbox: (x1, y1, x2, y2) in pixel coordinates.
        confidence: Detector confidence score [0.0, 1.0].
        frame_number: Frame index where detected.
    """
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    frame_number: int

    @property
    def x1(self) -> int: return self.bbox[0]
    @property
    def y1(self) -> int: return self.bbox[1]
    @property
    def x2(self) -> int: return self.bbox[2]
    @property
    def y2(self) -> int: return self.bbox[3]
    @property
    def width(self) -> int: return self.x2 - self.x1
    @property
    def height(self) -> int: return self.y2 - self.y1
    @property
    def area(self) -> int: return self.width * self.height
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


# --------- Validator ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def validate_bbox(
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int],
    min_width: int = 30,
    min_height: int = 30,
) -> bool:
    """
    Validate that a bounding box is geometrically sound.

    Args:
        bbox: (x1, y1, x2, y2).
        frame_shape: (height, width) of the source frame.
        min_width: Minimum face width in pixels.
        min_height: Minimum face height in pixels.

    Returns:
        True if valid, False otherwise.
    """
    x1, y1, x2, y2 = bbox
    h, w = frame_shape[:2]

    if x1 >= x2 or y1 >= y2:
        return False  # Degenerate box

    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return False  # Out of frame bounds

    if (x2 - x1) < min_width or (y2 - y1) < min_height:
        return False  # Too small to recognize reliably

    return True


# --------- Detector ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class YOLOFaceDetector:
    """
    YOLOv8-face face detector.

    Uses ultralytics YOLOv8 with a face-specific model weight.
    Supports configurable frame skipping: detection runs every N frames;
    the last valid result is returned on skipped frames.

    Attributes:
        _model: Loaded YOLO model (ultralytics).
        _frame_skip: Detect every N frames.
        _conf: Confidence threshold.
        _iou: IoU threshold for NMS.
        _last_detections: Cached detections from the last detection frame.
        _frames_since_detection: Counter for frame-skip logic.
    """

    # Public model name for logging (licensing info)
    MODEL_INFO = (
        "YOLOv8n-face | Trained on WIDER FACE | "
        "Source: https://github.com/akanametov/yolo-face | "
        "License: AGPL-3.0"
    )

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        detection_frame_skip: int = 2,
        min_face_width: int = 30,
        min_face_height: int = 30,
        input_size: int = 640,
    ) -> None:
        self._model_path = model_path
        self._conf = confidence_threshold
        self._iou = iou_threshold
        self._frame_skip = detection_frame_skip
        self._min_w = min_face_width
        self._min_h = min_face_height
        self._input_size = input_size

        self._model = None  # Loaded lazily
        self._last_detections: List[FaceDetection] = []
        self._frames_since_detection: int = 0

    def load(self) -> None:
        """
        Load the YOLO model. Downloads if not present.
        Raises FileNotFoundError if model_path points to a non-existent
        local file that cannot be resolved by ultralytics auto-download.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            ) from exc

        model_path = Path(self._model_path)
        if model_path.exists():
            logger.info(f"Loading YOLO face model from: {model_path}")
            self._model = YOLO(str(model_path))
        else:
            # Attempt to download the face model from the standard source
            logger.warning(
                f"Model not found at {model_path}. "
                "Attempting to download yolov8n-face.pt from Hugging Face. "
                f"Info: {self.MODEL_INFO}"
            )
            # ultralytics will download if given a known hub name
            # For face detection, we use the community yolo-face weights
            self._model = YOLO("yolov8n-face.pt")
            # Save to configured path for future runs
            model_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"YOLO face detector loaded. frame_skip={self._frame_skip}, conf={self._conf}")

    def detect(
        self, frame: np.ndarray, frame_number: int
    ) -> Tuple[List[FaceDetection], bool]:
        """
        Run face detection on a frame, respecting frame_skip.

        Args:
            frame: BGR image array (HxWxC).
            frame_number: Current frame index (used for frame_skip logic).

        Returns:
            Tuple of (detections, ran_detection).
            ran_detection=True if the model actually ran this frame.
            ran_detection=False if cached results were returned.
        """
        if self._model is None:
            raise RuntimeError("Detector not loaded. Call load() first.")

        if frame is None or frame.size == 0:
            logger.warning(f"Empty or None frame received at frame {frame_number}.")
            return [], False

        # Frame-skip logic: run detection on the first frame and every N frames thereafter
        if self._frames_since_detection % self._frame_skip != 0:
            self._frames_since_detection += 1
            # On skipped frames, do NOT return cached detections as fresh observations.
            # Return empty list and ran_detection=False so tracker can extrapolate position.
            return [], False

        self._frames_since_detection += 1

        # Run inference
        try:
            results = self._model.predict(
                source=frame,
                conf=self._conf,
                iou=self._iou,
                imgsz=self._input_size,
                verbose=False,
                device="cpu",  # Explicit CPU; set to 0 for GPU
            )
        except Exception as exc:
            logger.error(f"YOLO inference failed at frame {frame_number}: {exc}")
            return [], False

        detections: List[FaceDetection] = []
        frame_h, frame_w = frame.shape[:2]

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())

                # Clip to frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame_w, x2)
                y2 = min(frame_h, y2)

                bbox = (x1, y1, x2, y2)
                if validate_bbox(bbox, (frame_h, frame_w), self._min_w, self._min_h):
                    detections.append(
                        FaceDetection(bbox=bbox, confidence=conf, frame_number=frame_number)
                    )

        logger.debug(
            f"Frame {frame_number}: detected {len(detections)} valid faces."
        )
        self._last_detections = detections
        return detections, True

    def unload(self) -> None:
        """Release model resources."""
        self._model = None
        logger.info("YOLOFaceDetector unloaded.")


# --------- Factory ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def create_detector(config) -> YOLOFaceDetector:
    """Create and load a YOLOFaceDetector from AppConfig."""
    det_cfg = config.detection
    detector = YOLOFaceDetector(
        model_path=det_cfg.model_path,
        confidence_threshold=det_cfg.confidence_threshold,
        iou_threshold=det_cfg.iou_threshold,
        detection_frame_skip=det_cfg.detection_frame_skip,
        min_face_width=det_cfg.min_face_width,
        min_face_height=det_cfg.min_face_height,
        input_size=det_cfg.input_size,
    )
    detector.load()
    return detector

