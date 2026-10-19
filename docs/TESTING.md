# SentinelTrack — Testing Strategy & Verification Report

This document records the automated testing suite, execution commands, verified test results, and documented boundary conditions for SentinelTrack.

---

## 1. Testing Strategy

The test suite is structured into three distinct layers:

1. **Unit Tests (`tests/unit/`)**: Fast, deterministic tests verifying individual module behaviors, input validation, mathematical normalization, cosine similarity calculations, bounding box geometry, and state transitions without external database or hardware dependencies.
2. **End-to-End Pipeline Tests (`tests/e2e/`)**: Simulated full-pipeline tests verifying that detections are converted to normalized embeddings, registered as identities, matched on subsequent frames, and processed through session state transitions.
3. **Database Integration Tests (`tests/integration/`)**: Real PostgreSQL database tests validating migrations, schema constraints, foreign keys, transaction rollbacks, partial unique indexes, and idempotency protection. These tests are automatically skipped if PostgreSQL credentials are not configured in the active environment.

---

## 2. Test Execution Commands

### Running Unit Tests
```powershell
python -m pytest tests/unit -v
```

### Running End-to-End Tests
```powershell
python -m pytest tests/e2e -v
```

### Running All Tests with Coverage Report
```powershell
python -m pytest --cov=src tests/unit tests/e2e -v
```

### Running PostgreSQL Integration Tests (Requires Live DB)
```powershell
# Set credentials in .env, then run:
python -m pytest tests/integration/ -v
```

---

## 3. Verified Test Results

### 3.1 Unit Test Suite (`tests/unit/`)
Executed on Python 3.10.11 (Windows 11). **41 tests executed, 41 passed (100% pass rate).**

| Test Module | Test Name | Status | Verified Functionality |
|---|---|---|---|
| `test_config.py` | `test_load_minimal_config` | **PASSED** | Validates default values and minimal config loading |
| `test_config.py` | `test_missing_config_file` | **PASSED** | Verifies `FileNotFoundError` on non-existent config |
| `test_config.py` | `test_rtsp_requires_uri` | **PASSED** | Enforces `rtsp_uri` requirement when source is RTSP |
| `test_config.py` | `test_frame_skip_range` | **PASSED** | Validates `detection_frame_skip` boundary `[1, 60]` |
| `test_config.py` | `test_confidence_threshold_range` | **PASSED** | Rejects confidence values outside `[0.01, 1.0]` |
| `test_detector.py` | `test_validate_bbox_valid` | **PASSED** | Accepts geometrically sound face bounding boxes |
| `test_detector.py` | `test_validate_bbox_degenerate` | **PASSED** | Rejects inverted boxes (`x1 >= x2` or `y1 >= y2`) |
| `test_detector.py` | `test_validate_bbox_out_of_frame` | **PASSED** | Rejects boxes exceeding frame dimensions |
| `test_detector.py` | `test_validate_bbox_too_small` | **PASSED** | Rejects faces smaller than `30x30` pixels |
| `test_detector.py` | `test_validate_bbox_negative_coords` | **PASSED** | Rejects negative pixel coordinates |
| `test_detector.py` | `test_face_detection_properties` | **PASSED** | Verifies width, height, area, center computations |
| `test_detector.py` | `test_face_detection_x1_y1` | **PASSED** | Verifies coordinate accessors |
| `test_evidence_storage.py` | `test_evidence_path_collision_resistance` | **PASSED** | Ensures distinct frame/time produces unique paths |
| `test_evidence_storage.py` | `test_save_evidence_valid_crop` | **PASSED** | Confirms atomic image write and non-zero file size |
| `test_evidence_storage.py` | `test_save_evidence_none_crop` | **PASSED** | Safely returns `None` on empty or invalid crop input |
| `test_evidence_storage.py` | `test_save_evidence_too_small_crop` | **PASSED** | Rejects micro-crops below minimum threshold |
| `test_evidence_storage.py` | `test_verify_path_exists` | **PASSED** | Verifies filesystem existence check utility |
| `test_recognizer.py` | `test_normalize_unit_norm` | **PASSED** | Confirms L2 vector normalization yields norm $\approx 1.0$ |
| `test_recognizer.py` | `test_normalize_zero_vector` | **PASSED** | Guards against division by zero on blank embeddings |
| `test_recognizer.py` | `test_cosine_similarity_identical` | **PASSED** | Identical unit vectors yield similarity $1.000$ |
| `test_recognizer.py` | `test_cosine_similarity_orthogonal` | **PASSED** | Orthogonal vectors yield similarity $0.000$ |
| `test_recognizer.py` | `test_cosine_similarity_range` | **PASSED** | Cosine similarity is strictly bounded in `[-1.0, 1.0]` |
| `test_recognizer.py` | `test_embedding_serialization_roundtrip` | **PASSED** | Confirms lossless float32 BYTEA serialization |
| `test_recognizer.py` | `test_match_returns_best` | **PASSED** | Matches closest registered identity |
| `test_recognizer.py` | `test_match_no_registered` | **PASSED** | Returns `None` when database has zero registered faces |
| `test_recognizer.py` | `test_match_below_threshold` | **PASSED** | Rejects candidate when similarity < threshold |
| `test_recognizer.py` | `test_ambiguity_detection` | **PASSED** | Flags matches when top-2 candidates are within margin |
| `test_tracker.py` | `test_compute_iou_perfect_overlap` | **PASSED** | Identical bounding boxes produce $\text{IoU} = 1.0$ |
| `test_tracker.py` | `test_compute_iou_no_overlap` | **PASSED** | Disjoint bounding boxes produce $\text{IoU} = 0.0$ |
| `test_tracker.py` | `test_compute_iou_partial` | **PASSED** | Partial overlaps produce fractional IoU |
| `test_tracker.py` | `test_new_track_created_on_detection` | **PASSED** | First detection creates `TENTATIVE` track |
| `test_tracker.py` | `test_track_promoted_to_active` | **PASSED** | Track promoted to `ACTIVE` after $N$ observations |
| `test_tracker.py` | `test_track_expires_on_missed_frames` | **PASSED** | Track expired and removed after `max_missed_frames` |
| `test_tracker.py` | `test_two_tracks_not_merged` | **PASSED** | Spatially separated faces receive distinct track IDs |
| `test_tracker.py` | `test_face_id_assignment` | **PASSED** | Persistent `face_id` attached to temporary `track_id` |
| `test_tracker.py` | `test_track_id_differs_from_face_id` | **PASSED** | Confirms track ID (int) vs face ID (UUID) separation |
| `test_visit_session.py` | `test_visit_session_manager_imports` | **PASSED** | Module imports and initializes cleanly |
| `test_visit_session.py` | `test_unique_identity_count_does_not_increase_on_return` | **PASSED** | Returning visitor does not increment unique count |
| `test_visit_session.py` | `test_entry_not_fired_twice_same_session` | **PASSED** | Second frame of same visit suppresses duplicate entry |
| `test_visit_session.py` | `test_exit_requires_absence` | **PASSED** | Single missing frame does not prematurely fire exit |
| `test_visit_session.py` | `test_returning_visitor_creates_new_session` | **PASSED** | Subsequent visit creates distinct session record |

---

### 3.2 End-to-End Pipeline Suite (`tests/e2e/`)
Executed on Python 3.10.11 (Windows 11). **6 tests executed, 6 passed (100% pass rate).**

| Test Name | Result | Verified End-to-End Invariant |
|---|---|---|
| `test_embedding_normalization_pipeline` | **PASSED** | End-to-end embedding pipeline preserves unit norm |
| `test_new_face_registered_once` | **PASSED** | Face registered once; subsequent frames recognized |
| `test_different_persons_distinct_identities` | **PASSED** | Dissimilar faces assigned separate persistent IDs |
| `test_returning_visitor_same_face_id` | **PASSED** | Person leaves and returns; same UUID reused |
| `test_bbox_validation_in_pipeline` | **PASSED** | Malformed crops filtered before recognition |
| `test_iou_tracker_multiple_persons` | **PASSED** | Concurrent multi-person tracking maintains distinct tracks |

---

## 4. Documented Untested Environments & Limitations

To maintain absolute engineering integrity, the following test limitations are formally documented:

1. **Live Physical RTSP Camera**: Live testing was performed using simulated video streams and synthetic sources. Live streaming on proprietary IP hardware remains subject to physical network deployment.
2. **GPU Acceleration**: All benchmarks and automated tests were executed using the CPU runtime (`onnxruntime` with `CPUExecutionProvider`). CUDA execution paths are implemented but were not benchmarked on physical NVIDIA hardware in this test cycle.
3. **High Dwell Time Tests (>1 Hour)**: Automated test fixtures test session timeouts using simulated frame counters rather than real 1-hour physical delays.
