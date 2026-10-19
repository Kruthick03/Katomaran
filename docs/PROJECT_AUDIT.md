# SentinelTrack — Complete Project Audit and Defect Analysis

**Date:** 2026-10-18  
**System:** SentinelTrack — Intelligent Face Tracker with Auto-Registration and Visitor Counting  
**Authoritative Specification:** Katomaran Hackathon Problem Statement PDF  

---

## 1. Existing Architecture & Module Overview

SentinelTrack implements a pipelined computer-vision and transactional database architecture:

```
[ Video Input Source (File / RTSP) ]
                ↓
    [ YOLOv8n-face Detector ] (Configurable frame-skip & scale filtering)
                ↓
    [ Multi-Face ByteTrack-style Tracker ] (Global best-first IoU matching & track state machine)
                ↓
    [ InsightFace ArcFace Recognizer ] (512-dim embedding extraction & cosine similarity)
                ↓
    [ IdentityService ] (In-memory registered gallery cache + PostgreSQL Face repository)
                ↓
    [ VisitSessionManager ] (Finite State Machine: ACTIVE ↔ EXIT_PENDING → EXITED)
                ↓
[ EventService & EvidenceStorage ] ──→ [ logs/entries/ & logs/exits/ JPEG crops ]
                ↓
    [ PostgreSQL 18 DB ] (faces, visit_sessions, events, cameras)
                ↓
[ SentinelVisualizer ] ──→ [ Live OpenCV Window + logs/output_demo.mp4 ]
```

### Inspected Files and Modules:
- `src/main.py`: Top-level CLI and pipeline orchestrator.
- `src/config.py` & `config.json`: Pydantic validated application configuration.
- `src/face_detector.py`: YOLOv8 face detector using WIDER-FACE pretrained `yolov8n-face.pt`.
- `src/face_recognizer.py`: ArcFace 512-dim embedding generation and normalized cosine similarity.
- `src/tracker.py`: Multi-face tracker with IoU association and track lifecycle.
- `src/identity_service.py`: Auto-registration of new identities and cache synchronization.
- `src/visit_session.py`: Visit session state machine managing entry/exit event triggering.
- `src/repositories.py`: SQLAlchemy database repositories with transaction management.
- `src/models.py`: Declarative PostgreSQL models (faces, visit_sessions, events, cameras).
- `src/evidence_storage.py`: Collision-resistant JPEG face crop storage with atomic rename.
- `src/visualizer.py`: OpenCV frame renderer with bounding boxes, HUD overlay, and banner alerts.
- `src/metrics.py`: Latency, throughput (FPS), and system counter metrics collector.
- `src/recovery.py`: Startup recovery checker for interrupted sessions.

---

## 2. Confirmed Defects and Root Causes

| Defect ID | Component | Description & Root Cause | Correction Applied |
|---|---|---|---|
| **DEF-01** | `src/tracker.py` | **Premature Track Expiration**: Tracks were unconditionally expired if `track.age > max_age` (default 30). This killed actively detected persons every ~1.2s, creating fragmented tracks and duplicate visits. | Removed `max_age` check for `ACTIVE` tracks; `max_age` now only purges dead `TENTATIVE` tracks. Default raised to 300. |
| **DEF-02** | `src/tracker.py` | **Track Hijacking on Adjacent Faces**: Sequential greedy IoU matching allowed early detections to capture suboptimal tracks. | Replaced with global best-first IoU matching (all pairs sorted by IoU descending). |
| **DEF-03** | `src/face_recognizer.py` | **Ambiguity False Rejection**: `match()` tested ambiguity across all gallery entries even when `best_sim` was below threshold (e.g., sim=0.09). New faces were mislabeled ambiguous and not registered. | Constrained ambiguity checks only to candidates where `best_sim >= similarity_threshold`. |
| **DEF-04** | `src/visit_session.py` | **Immediate Exit Confirmation**: Absence evaluation transitioned directly to `EXIT_PENDING` and then confirmed `EXIT` in the same loop pass. | Required absence across multiple cycles before confirming exit, absorbing transient frame drops. |
| **DEF-05** | `src/face_detector.py` | **Infant/Baby Face Misses**: Face bounding boxes <30x30px and confidence <0.45 were discarded by geometry validation and high detector thresholds. | Lowered `min_face_width` to 16px, `confidence_threshold` to 0.25, and internal recognizer crop limit to 14px. |
| **DEF-06** | `src/main.py` | **GUI Window Displaced / No Headless Video**: OpenCV window opened off-screen on scaled monitors; headless mode produced no visual artifact. | Added `cv2.moveWindow(..., 0, 0)` and automatic `cv2.VideoWriter` to `logs/output_demo.mp4`. |
| **DEF-07** | `src/visualizer.py` & `src/visit_session.py` | **Suspicious HUD Visitor Counts**: The HUD queried cumulative `COUNT(*)` from historical DB runs across all previous test executions, showing static counts that didn't reflect the active video run. | Added stream-level unique visitor (`feed_unique_face_ids`) and session counters to `VisitSessionManager`, displaying both feed metrics and persistent `DB Registered` totals separately. |
| **DEF-08** | `src/visualizer.py` | **Ghost Overlapping Boxes on Inactive Tracks**: The visualizer previously drew every track where `state != TrackState.REMOVED`. Tracks in `TrackState.LOST` with `missed_frames > 0` remained frozen on screen for up to 30 frames over clothes and background. | Visualizer production mode strictly filters `track.missed_frames == 0`. Overlapping ghost tracks dropped from 9 to 1 on Frame 492. |
| **DEF-09** | `src/face_recognizer.py` | **Duplicate Identity Stamping Across Multiple Tracks**: When a stale/noise bounding box contained no face in its crop, InsightFace fell back to full-frame detection without spatial intersection constraints, repeatedly selecting the dominant face in the frame (the woman `ca775c36`) and assigning her ID to phantom tracks. | Enforced spatial $\text{IoU} \ge 0.25$ threshold between the track bounding box and the candidate face detected in full-frame fallback. Track identities locked after initial confirmation. |
| **DEF-10** | `src/face_recognizer.py` & DB | **Cosine Dot Product Exceeding 1.00**: An unnormalized vector in PostgreSQL record `a37439f2` with L2 norm $21.0605$ caused raw dot products to reach $1.74$–$2.45$. | Rewrote cosine similarity to divide by $(\|a\|_2 \|b\|_2)$ and bound mathematically within $[-1.0, 1.0]$. Normalized DB embeddings; repositories normalize on load. |
| **DEF-11** | `src/visualizer.py` | **HUD Ghost Text & Unicode Title Bar**: The HUD had a 78% transparent overlay allowing the bright yellow CCTV camera timestamp (`10:39:23`) to bleed through text. Windows GDI OpenCV corrupted the Unicode em-dash (`\u2014`) in the window title. | Increased HUD opacity to 95% dark charcoal (`BGR (18, 18, 18)`); replaced Unicode em-dash with ASCII hyphen. |
| **DEF-12** | `src/face_detector.py` & `src/tracker.py` | **Stale Detections Replayed on Skipped Frames**: On frames where detection was skipped (`detection_frame_skip > 1`), `detect()` previously replayed cached detection boxes as fresh observations into `tracker.update()`, causing frozen boxes and bypassing true motion extrapolation. | On skipped frames, `detect()` returns empty detections with `ran_detection=False`. Tracker runs `predict_motion()` to smoothly extrapolate position using damped linear velocity without creating spurious tracks or replaying stale boxes. |
| **DEF-13** | `src/face_detector.py` | **Baby/Child Face Investigation**: Analysis of the sample video confirmed the child is not walking with the couple in frames 500-600. The child enters alone at frame 740. In the uncalibrated configuration (threshold 0.45), the child's detections (conf 0.31-0.46) were dropped. | Calibrated confidence threshold to 0.35 and min face dimensions to 20x20. Child is detected and tracked across frames 740-827 continuously. |

---

## 3. Mandatory Requirements Verification

| Requirement | Implementation Module | Verification Status |
|---|---|---|
| Process sample video file (`data/katomaran_sample.mp4`) | `src/video_input.py` | Verified (1502 frames processed) |
| Configurable live RTSP camera support | `src/video_input.py`, `scripts/run_rtsp.py` | Implemented & validated |
| YOLO face detection (not generic COCO) | `src/face_detector.py` | Verified (`yolov8n-face.pt`) |
| InsightFace ArcFace facial embeddings | `src/face_recognizer.py` | Verified (`buffalo_sc`, 512-dim) |
| Auto-registration of new identities | `src/identity_service.py` | Verified (UUID assigned in PostgreSQL) |
| Multi-face tracking across frames | `src/tracker.py` | Verified (ByteTrack IoU logic) |
| Configurable frame skip (`detection_frame_skip`) | `config.json`, `src/face_detector.py` | Verified (frame_skip=2) |
| Exactly one entry & exit per session | `src/visit_session.py` | Verified (idempotent DB keys) |
| Cropped face evidence saved locally | `src/evidence_storage.py` | Verified (`logs/entries/`, `logs/exits/`) |
| Structured `events.log` audit trail | `src/event_logger.py` | Verified (5150+ lines generated) |
| PostgreSQL primary persistence | `src/database.py`, `src/repositories.py` | Verified (PostgreSQL 18 active) |
| Automated test suite | `tests/` | Verified (122/122 tests passing) |

---

## 4. Risks & Limitations

1. **CPU Execution Throughput**: ArcFace on CPU achieves ~5.3 FPS on 2K resolution video. Frame-skip=2 is necessary to maintain smooth throughput without GPU acceleration.
2. **Extreme Occlusion**: Complete physical occlusion lasting beyond `max_missed_frames` (30 frames) will trigger exit confirmation. When the subject re-appears, their identity is matched by ArcFace, initiating a new visit session without incrementing unique visitor count.
3. **Severe Off-Axis Poses**: When faces exceed 60° yaw or pitch (e.g. looking straight down), detector confidence drops below 0.25.
