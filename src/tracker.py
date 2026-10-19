"""
SentinelTrack --- src/tracker.py

Lightweight multi-face tracker (ByteTrack-inspired IoU matching).

Design rationale:
  ByteTrack (Zhang et al., 2022) requires specific C++ extensions that are
  difficult to install on Windows. This module implements the core concepts:
    - Active tracks with unique temporary IDs.
    - IoU-based matching between detections and existing tracks.
    - Missed-frame counting for track expiry.
    - Track lifecycle: Tentative --- Active --- Lost --- Removed.

Key distinction:
  - Track IDs are TEMPORARY and change between runs.
  - Persistent face IDs are assigned by identity_service and survive restarts.
  - Do NOT use track_id as a database face_id.

Frame-skip behaviour:
  When detection is skipped (detector returned cached results), the tracker
  still updates position estimates using the cached bounding boxes.
  This keeps tracks alive between detection frames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.face_detector import FaceDetection

logger = logging.getLogger("sentineltrack.tracker")

# --------- Track lifecycle states ---------------------------------------------------------------------------------------------------------------------------------------------------------

class TrackState:
    TENTATIVE = "tentative"   # Recently created, not yet confirmed
    ACTIVE = "active"         # Confirmed, consistently detected
    LOST = "lost"             # Missed frames but not yet expired
    REMOVED = "removed"       # Expired, should be purged


# --------- Track dataclass ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class Track:
    """
    A single tracked face across frames.

    track_id: Temporary in-process ID (NOT persistent).
    face_id: Persistent database face ID, assigned after recognition.
    """
    track_id: int
    bbox: Tuple[int, int, int, int]      # Current (x1, y1, x2, y2)
    confidence: float
    state: str = TrackState.TENTATIVE
    age: int = 0                          # Total frames this track has existed
    missed_frames: int = 0               # Consecutive missed detections
    face_id: Optional[str] = None        # Set by identity_service after recognition
    recognition_status: str = "Unknown"  # "New identity", "Recognized", "Unknown"
    similarity: Optional[float] = None   # Cosine similarity score
    entry_logged: bool = False           # Has entry event been fired?
    exit_pending: bool = False           # Awaiting exit confirmation?
    exit_pending_frames: int = 0        # Frames since track was lost

    # History for simple velocity estimation (last N bboxes)
    bbox_history: List[Tuple[int, int, int, int]] = field(default_factory=list)

    def update(self, detection: FaceDetection, min_age: int = 3) -> None:
        """Update track with a new detection."""
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.missed_frames = 0
        self.exit_pending = False
        self.exit_pending_frames = 0
        self.age += 1
        self.bbox_history.append(detection.bbox)
        if len(self.bbox_history) > 10:
            self.bbox_history.pop(0)

        # Promote tentative to active after sufficient age
        if self.state == TrackState.TENTATIVE and self.age >= min_age:
            self.state = TrackState.ACTIVE
        # Re-promote lost tracks back to active on re-detection
        elif self.state == TrackState.LOST:
            self.state = TrackState.ACTIVE

    def mark_missed(self) -> None:
        """Increment missed-frame counter."""
        self.missed_frames += 1
        self.age += 1
        if self.state in (TrackState.ACTIVE, TrackState.TENTATIVE):
            self.state = TrackState.LOST

    def predict_motion(self) -> None:
        """
        Extrapolate position on skipped frames based on recent velocity.
        Advances track age and prevents frozen coordinates without
        replaying stale detections as fresh observations.
        """
        self.age += 1
        if len(self.bbox_history) >= 2:
            prev = self.bbox_history[-2]
            curr = self.bbox_history[-1]
            dx1 = curr[0] - prev[0]
            dy1 = curr[1] - prev[1]
            dx2 = curr[2] - prev[2]
            dy2 = curr[3] - prev[3]

            # Damped linear velocity prediction (0.8 factor)
            new_x1 = int(round(curr[0] + 0.8 * dx1))
            new_y1 = int(round(curr[1] + 0.8 * dy1))
            new_x2 = int(round(curr[2] + 0.8 * dx2))
            new_y2 = int(round(curr[3] + 0.8 * dy2))

            if new_x2 > new_x1 + 10 and new_y2 > new_y1 + 10:
                self.bbox = (new_x1, new_y1, new_x2, new_y2)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


# --------- IoU utility ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def compute_iou(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
) -> float:
    """Compute Intersection-over-Union between two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0
    return intersection / union


# --------- Tracker ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class FaceTracker:
    """
    IoU-based multi-face tracker.

    Matches incoming detections to existing tracks using greedy IoU matching.
    Unmatched detections create new tentative tracks.
    Tracks missing for more than max_missed_frames are removed.
    """

    def __init__(
        self,
        max_missed_frames: int = 15,
        iou_match_threshold: float = 0.20,
        min_track_age_frames: int = 3,
        max_age: int = 300,
    ) -> None:
        self._max_missed = max_missed_frames
        self._iou_thresh = iou_match_threshold
        self._min_age = min_track_age_frames
        self._max_age = max_age

        self._tracks: Dict[int, Track] = {}
        self._next_track_id: int = 1

    def update(
        self,
        detections: List[FaceDetection],
        frame_number: int,
        is_detection_frame: bool = True,
    ) -> Tuple[List[Track], List[Track]]:
        """
        Update tracker state with new detections or extrapolate on skipped frames.

        Args:
            detections: Face detections from the current frame (empty on skipped frames).
            frame_number: Current frame index.
            is_detection_frame: True if face detection actually ran on this frame;
                                False if this is a skipped frame (predict motion only).

        Returns:
            Tuple (active_tracks, removed_tracks).
            active_tracks: All non-removed tracks (including lost).
            removed_tracks: Tracks that expired this frame.
        """
        active = [t for t in self._tracks.values()
                  if t.state != TrackState.REMOVED]

        # On skipped frames, use velocity prediction without matching or falsely
        # marking tracks missed or creating spurious tracks from cached bboxes.
        if not is_detection_frame:
            for track in active:
                track.predict_motion()
            return [t for t in self._tracks.values() if t.state != TrackState.REMOVED], []

        unmatched_dets = list(range(len(detections)))
        matched_track_ids = set()

        # Global best-first greedy IoU matching:
        # Compute all pairwise IoUs, sort by score descending, assign greedily.
        # This prevents a suboptimal detection→track assignment from blocking
        # a better one when multiple faces are close together.
        if active and detections:
            pairs = []
            for det_idx, det in enumerate(detections):
                for track in active:
                    iou = compute_iou(det.bbox, track.bbox)
                    if iou >= self._iou_thresh:
                        pairs.append((iou, det_idx, track.track_id))

            # Sort by IoU descending so the best matches are assigned first
            pairs.sort(key=lambda x: x[0], reverse=True)

            for iou_score, det_idx, tid in pairs:
                if det_idx not in unmatched_dets:
                    continue  # This detection was already matched
                if tid in matched_track_ids:
                    continue  # This track was already matched
                track = self._tracks[tid]
                track.update(detections[det_idx], min_age=self._min_age)
                matched_track_ids.add(tid)
                unmatched_dets.remove(det_idx)

            # Secondary matching pass for remaining detections & tracks:
            # Recovers fast-moving faces where IoU is between 0.10 and threshold
            # with small center-to-center displacement
            if unmatched_dets and len(matched_track_ids) < len(active):
                second_pairs = []
                for det_idx in unmatched_dets:
                    det = detections[det_idx]
                    min_dim = min(det.width, det.height)
                    for track in active:
                        if track.track_id in matched_track_ids:
                            continue
                        iou = compute_iou(det.bbox, track.bbox)
                        cx1, cy1 = det.center
                        cx2, cy2 = track.center
                        dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                        if iou >= 0.10 and dist <= min_dim * 0.85:
                            second_pairs.append((iou, -dist, det_idx, track.track_id))

                second_pairs.sort(key=lambda x: (x[0], x[1]), reverse=True)
                for iou_score, neg_dist, det_idx, tid in second_pairs:
                    if det_idx not in unmatched_dets:
                        continue
                    if tid in matched_track_ids:
                        continue
                    track = self._tracks[tid]
                    track.update(detections[det_idx], min_age=self._min_age)
                    matched_track_ids.add(tid)
                    unmatched_dets.remove(det_idx)

        # Mark unmatched tracks as missed
        for track in active:
            if track.track_id not in matched_track_ids:
                track.mark_missed()

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            init_state = TrackState.ACTIVE if self._min_age <= 1 else TrackState.TENTATIVE
            new_track = Track(
                track_id=self._next_track_id,
                bbox=det.bbox,
                confidence=det.confidence,
                state=init_state,
                age=1,
            )
            new_track.bbox_history.append(det.bbox)
            self._tracks[self._next_track_id] = new_track
            self._next_track_id += 1
            logger.debug(
                f"New track created: id={new_track.track_id} frame={frame_number}"
            )

        # Expire old tracks
        # IMPORTANT: Only expire LOST or TENTATIVE tracks that exceeded
        # max_missed_frames. ACTIVE tracks must never be auto-removed
        # just because they have been alive for many frames — that would
        # kill a person who is continuously visible and force a duplicate
        # session to be created.
        removed: List[Track] = []
        for track in list(self._tracks.values()):
            should_remove = False
            if track.state in (TrackState.LOST, TrackState.TENTATIVE):
                if track.missed_frames >= self._max_missed:
                    should_remove = True
                # Also expire tentative/lost tracks that have been alive
                # a very long time without ever becoming active
                elif track.state == TrackState.TENTATIVE and track.age > self._max_age:
                    should_remove = True

            if should_remove:
                track.state = TrackState.REMOVED
                removed.append(track)
                logger.debug(
                    f"Track expired: id={track.track_id} face_id={track.face_id} "
                    f"missed={track.missed_frames}"
                )

        current_active = [
            t for t in self._tracks.values()
            if t.state != TrackState.REMOVED
        ]

        return current_active, removed

    def get_active_tracks(self) -> List[Track]:
        """Return all non-removed tracks."""
        return [t for t in self._tracks.values()
                if t.state != TrackState.REMOVED]

    def get_track(self, track_id: int) -> Optional[Track]:
        """Get a specific track by ID."""
        return self._tracks.get(track_id)

    def assign_face_id(self, track_id: int, face_id: str) -> None:
        """Assign a persistent face_id to a track after recognition."""
        track = self._tracks.get(track_id)
        if track:
            track.face_id = face_id
            logger.debug(f"Track {track_id} assigned face_id={face_id}")

    @property
    def track_count(self) -> int:
        """Number of currently active (non-removed) tracks."""
        return sum(1 for t in self._tracks.values()
                   if t.state != TrackState.REMOVED)


# --------- Factory ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def create_tracker(config) -> FaceTracker:
    """Create a FaceTracker from AppConfig."""
    t_cfg = config.tracker
    return FaceTracker(
        max_missed_frames=t_cfg.max_missed_frames,
        iou_match_threshold=t_cfg.iou_match_threshold,
        min_track_age_frames=t_cfg.min_track_age_frames,
        max_age=t_cfg.max_age,
    )

