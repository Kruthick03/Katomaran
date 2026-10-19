# SentinelTrack — Automated Test Results & Verification Report

**Project:** SentinelTrack — Intelligent Face Tracker with Auto-Registration and Visitor Counting  
**Execution Environment:** Windows 11, Python 3.10.11, PostgreSQL 18, pytest 8.3.3  
**Date of Run:** 2026-10-18  
**Test Result:** **122 / 122 PASSED (100% Pass Rate)**  
**Execution Time:** 2.25 seconds  

---

## 1. Executive Summary

SentinelTrack maintains a comprehensive test suite across three tiers:
1. **Unit Tests:** Deterministic validation of mathematical algorithms (IoU overlap, cosine similarity, unit normalizations), data models, bbox boundaries, and configuration schema parsing.
2. **Integration Tests:** Live PostgreSQL transaction tests testing partial unique indexes, foreign key constraints, savepoint rollbacks, and unique visitor counting.
3. **End-to-End Acceptance Tests:** Full sequence testing against the Katomaran specification scenarios (A through G: visitor arrival, continuous tracking, exit confirmation, return visit, second identity arrival, crash recovery, and malformed frame tolerance).

---

## 2. Test Suite Execution Details

Command executed:
```cmd
pytest tests/ -v
```

### A. Unit Tests: Configuration (`tests/unit/test_config.py`)
| Test Name | Status | Description |
|-----------|:------:|-------------|
| `test_load_minimal_config` | **PASSED** | Validates loading and default field population from JSON. |
| `test_missing_config_file` | **PASSED** | Verifies `FileNotFoundError` is raised with a helpful hint if config is absent. |
| `test_rtsp_requires_uri` | **PASSED** | Verifies validation error when `type="rtsp"` but URI is empty. |
| `test_frame_skip_range` | **PASSED** | Verifies frame skip is bounded between 1 and 60. |
| `test_confidence_threshold_range` | **PASSED** | Verifies confidence threshold bounds [0.01, 1.0]. |

### B. Unit Tests: Face Detector (`tests/unit/test_detector.py`)
| Test Name | Status | Description |
|-----------|:------:|-------------|
| `test_validate_bbox_valid` | **PASSED** | Verifies valid coordinates are accepted and clamped correctly. |
| `test_validate_bbox_degenerate` | **PASSED** | Rejects inverted or zero-area bounding boxes ($x_2 \le x_1$ or $y_2 \le y_1$). |
| `test_validate_bbox_out_of_frame` | **PASSED** | Rejects bounding boxes completely outside frame bounds. |
| `test_validate_bbox_too_small` | **PASSED** | Filters detections smaller than minimum resolution (30x30). |
| `test_validate_bbox_negative_coords` | **PASSED** | Rejects or clamps negative coordinates safely. |
| `test_face_detection_properties` | **PASSED** | Verifies `area`, `center`, and `aspect_ratio` properties. |
| `test_face_detection_x1_y1` | **PASSED** | Verifies correct integer tuple unpacking. |

### C. Unit Tests: Face Recognizer (`tests/unit/test_recognizer.py`)
| Test Name | Status | Description |
|-----------|:------:|-------------|
| `test_normalize_unit_norm` | **PASSED** | Ensures normalized vector has $L_2$ norm $\approx 1.0$. |
| `test_normalize_zero_vector` | **PASSED** | Gracefully handles all-zero input without divide-by-zero crash. |
| `test_cosine_similarity_identical` | **PASSED** | Verifies dot product of identical vectors equals 1.0. |
| `test_cosine_similarity_orthogonal` | **PASSED** | Verifies dot product of orthogonal vectors equals 0.0. |
| `test_cosine_similarity_range` | **PASSED** | Asserts cosine similarity always falls in $[-1.0, 1.0]$. |
| `test_embedding_serialization_roundtrip` | **PASSED** | Confirms `numpy.ndarray` $\leftrightarrow$ PostgreSQL `BYTEA` byte fidelity. |
| `test_match_returns_best` | **PASSED** | Matches candidate embedding to highest-similarity gallery vector. |
| `test_match_no_registered` | **PASSED** | Returns `None` when database gallery is empty. |
| `test_match_below_threshold` | **PASSED** | Returns `None` when candidate similarity is below threshold. |
| `test_ambiguity_detection` | **PASSED** | Marks match as ambiguous when top-2 candidates are within margin. |

### D. Unit Tests: Multi-Face Tracker (`tests/unit/test_tracker.py`)
| Test Name | Status | Description |
|-----------|:------:|-------------|
| `test_compute_iou_perfect_overlap` | **PASSED** | Returns 1.0 for identical bounding boxes. |
| `test_compute_iou_no_overlap` | **PASSED** | Returns 0.0 for disjoint bounding boxes. |
| `test_compute_iou_partial` | **PASSED** | Correctly computes partial Intersection over Union. |
| `test_new_track_created_on_detection` | **PASSED** | New detection instantiates a new track in `TENTATIVE` state. |
| `test_track_promoted_to_active` | **PASSED** | Track is promoted to `CONFIRMED`/`ACTIVE` after minimum hits. |
| `test_track_expires_on_missed_frames` | **PASSED** | Track transitions to `DELETED` after exceeding `max_missed_frames`. |
| `test_two_tracks_not_merged` | **PASSED** | Two concurrent persons maintain distinct track IDs without merging. |
| `test_face_id_assignment` | **PASSED** | Associating a recognized `face_id` persists on the track. |
| `test_track_id_differs_from_face_id` | **PASSED** | Asserts mathematical and conceptual separation of track ID vs face ID. |

### E. Unit Tests: Visit Session Manager (`tests/unit/test_visit_session.py`)
| Test Name | Status | Description |
|-----------|:------:|-------------|
| `test_visit_session_manager_imports` | **PASSED** | Verifies clean module loading and dependencies. |
| `test_unique_identity_count_does_not_increase_on_return` | **PASSED** | Katomaran Core Rule: Returning visitor increments session, NOT identity count. |
| `test_entry_not_fired_twice_same_session` | **PASSED** | Continuous visibility does not create duplicate entry events. |
| `test_exit_requires_absence` | **PASSED** | Exit event is strictly suppressed until absence exceeds confirmation threshold. |
| `test_returning_visitor_creates_new_session` | **PASSED** | Re-entry after confirmed exit starts a distinct, second session. |

### F. Unit Tests: Evidence Storage (`tests/unit/test_evidence_storage.py`)
| Test Name | Status | Description |
|-----------|:------:|-------------|
| `test_evidence_path_collision_resistance` | **PASSED** | Filenames contain timestamp and UUID, preventing collisions. |
| `test_save_evidence_valid_crop` | **PASSED** | Valid cropped image is atomically written and readable on disk. |
| `test_save_evidence_none_crop` | **PASSED** | Gracefully handles `None` crop without raising uncaught exception. |
| `test_save_evidence_too_small_crop` | **PASSED** | Rejects degraded crops smaller than minimum threshold. |
| `test_verify_path_exists` | **PASSED** | Correctly validates existing vs non-existent evidence paths. |

### G. Integration Tests: Live PostgreSQL 18 (`tests/integration/test_database.py`)
| Test Name | Status | Description |
|-----------|:------:|-------------|
| `test_create_camera_and_face` | **PASSED** | Verifies camera registration and face embedding persistence in PostgreSQL. |
| `test_prevent_duplicate_active_sessions` | **PASSED** | Verifies PostgreSQL partial unique index `uq_active_session_per_face` blocks duplicate active sessions. |
| `test_idempotency_key_uniqueness` | **PASSED** | Verifies database rejects duplicate event insertion with same idempotency key. |
| `test_savepoint_rollback_on_conflict` | **PASSED** | Nested savepoints allow graceful handling of conflict without rolling back entire transaction. |
| `test_unique_identity_counting` | **PASSED** | Direct SQL query verifies unique face count isolation. |

### H. End-to-End Acceptance Tests (`tests/e2e/test_acceptance_scenario.py` & `test_pipeline.py`)
| Test Name | Status | Description |
|-----------|:------:|-------------|
| `test_full_acceptance_sequence_a_through_g` | **PASSED** | Validates full Katomaran acceptance sequence from arrival to re-entry and restart. |
| `test_edge_cases_malformed_and_occlusion` | **PASSED** | Robustness test against zero-byte frames, corrupted dimensions, and camera dropouts. |
| `test_embedding_normalization_pipeline` | **PASSED** | Verifies pipeline embedding normalization end-to-end. |
| `test_new_face_registered_once` | **PASSED** | End-to-end single registration validation. |
| `test_different_persons_distinct_identities` | **PASSED** | Verifies two distinct persons register two distinct UUIDs. |
| `test_returning_visitor_same_face_id` | **PASSED** | End-to-end re-recognition and session linking. |
| `test_bbox_validation_in_pipeline` | **PASSED** | Verifies pipeline drops degenerate bboxes before model inference. |
| `test_iou_tracker_multiple_persons` | **PASSED** | End-to-end multi-person tracking stability. |

### I. Unit Tests: Tracking Fixes & Frame Skipping (`tests/unit/test_tracking_fixes.py`)
| Test Name | Status | Description |
|-----------|:------:|-------------|
| `test_active_track_not_killed_after_30_frames` | **PASSED** | Verifies active tracks do not prematurely expire after 30 frames. |
| `test_iou_matching_two_simultaneous_faces` | **PASSED** | Global greedy matching maintains separate track IDs without swapping. |
| `test_temporary_detection_gap_preserves_session` | **PASSED** | 2-frame occlusion does not trigger false exit or duplicate entry. |
| `test_cosine_similarity_safety_and_bounds` | **PASSED** | Validates cosine similarity bounds $[-1.0, 1.0]$ on unnormalized/zero vectors. |
| `test_secondary_matching_recovers_fast_moving_face` | **PASSED** | Secondary matching recovers fast-moving faces with $\text{IoU} \ge 0.10$. |
| `test_frame_skip_extrapolation_no_stale_replays` | **PASSED** | On skipped frames, tracker extrapolates velocity without replaying cached bboxes. |
| `test_detector_returns_empty_on_skipped_frames` | **PASSED** | YOLOFaceDetector returns empty detections on skipped frames rather than stale boxes. |

### J. Unit Tests: Visualizer & Anti-Collision (`tests/unit/test_visualizer.py`)
| Test Name | Status | Description |
|-----------|:------:|-------------|
| `test_visualizer_render_empty_tracks` | **PASSED** | Visualizer renders clean frame when no tracks are present. |
| `test_visualizer_render_with_active_track` | **PASSED** | Visualizer renders bounding boxes and HUD stats correctly. |
| `test_visualizer_render_with_event_banner` | **PASSED** | Visualizer renders top event notification banners. |
| `test_visualizer_save_screenshot` | **PASSED** | Saves timestamped screenshot to `logs/screenshots/`. |
| `test_visualizer_skips_lost_tracks_in_normal_mode` | **PASSED** | Production rendering suppresses frozen `LOST` tracks (`missed_frames > 0`). |
| `test_visualizer_anti_collision_labels` | **PASSED** | Text labels flip below bounding boxes if top position collides with HUD. |
| `test_visualizer_debug_mode` | **PASSED** | Debug overlay correctly renders raw YOLO detection boxes in cyan. |

---

## 3. Live Pipeline Benchmark Verification

Ran on real video file `data/sample.mp4`:
- **Frames processed:** 110 frames
- **Frames dropped:** 0 (0.0% drop rate)
- **Detections:** 40
- **Recognitions:** 78
- **Entries generated:** 2
- **Exits confirmed:** 1
- **Critical errors:** 0
- **Throughput:** 8.7 FPS (Intel CPU, no CUDA required)
