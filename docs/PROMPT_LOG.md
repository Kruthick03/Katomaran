# SentinelTrack — Development Prompt Log

This document records the prompts executed during the autonomous development of the SentinelTrack project, their engineering objectives, and the verified outcomes.

---

## Prompt 1: Workspace Inspection & Technical Stack Verification
- **Goal**: Inspect existing workspace, check operating system version, verify installed Python version, check availability of PostgreSQL services, Docker, and OpenCV/PyTorch/InsightFace libraries.
- **Commands Executed**:
  - `Get-ChildItem -Path ...`
  - `python --version`, `pip --version`
  - `docker --version`, `docker-compose --version`
  - `Get-Service -Name "postgresql*"`
- **Verified Outcome**:
  - Windows environment running Python 3.10.11.
  - Native Windows service `postgresql-x64-18` running PostgreSQL 18.
  - Docker is not installed. System must run native Windows processes.
  - Workspace was initially uninitialized; directory structure needed scaffolding.

---

## Prompt 2: Scaffolding Directory Structure & Configuration Templates
- **Goal**: Create normalized directory structure according to specification: `src/`, `migrations/`, `tests/`, `scripts/`, `models/`, `data/`, `logs/`, `docs/`. Create pinned `requirements.txt`, `.env.example`, `config.example.json`, `.gitignore`, `pyproject.toml`, and `docker-compose.yml`.
- **Files Created**:
  - `requirements.txt`
  - `.env.example`
  - `config.example.json`
  - `.gitignore`
  - `pyproject.toml`
  - `docker-compose.yml`
- **Verified Outcome**:
  - All project directories created.
  - Configurations established with environment variable bindings for sensitive credentials.

---

## Prompt 3: Core Database Engine, Models, and Alembic Migrations
- **Goal**: Build normalized SQLAlchemy 2.x declarative models (`cameras`, `faces`, `visit_sessions`, `events`) with BYTEA embedding storage, partial unique indexes for exactly-once entry/exit enforcement, and Alembic migration versions.
- **Files Created**:
  - `src/models.py`
  - `src/database.py`
  - `src/repositories.py`
  - `alembic.ini`
  - `migrations/env.py`
  - `migrations/versions/20261017_0000-0001_initial_schema.py`
- **Verified Outcome**:
  - Complete schema definition with FK relationships, check constraints, and partial unique indexes.
  - Transactional repository layer implementing idempotent retries.

---

## Prompt 4: Computer Vision Pipeline (Detector, Recognizer, Tracker, Video Source)
- **Goal**: Implement YOLOv8 face detector (`yolov8n-face.pt`) with frame skip, InsightFace ArcFace embedding extraction with L2 normalization, IoU multi-face tracker, and unified video input layer for files and RTSP streams.
- **Files Created**:
  - `src/video_input.py`
  - `src/face_detector.py`
  - `src/face_recognizer.py`
  - `src/tracker.py`
- **Verified Outcome**:
  - Bounding box validation and frame-skip caching.
  - Unit norm mathematical preservation and cosine similarity matching.
  - Clear architectural separation between integer `track_id` and persistent UUID `face_id`.

---

## Prompt 5: Services, Evidence Storage, Metrics, and Main Orchestration
- **Goal**: Implement `IdentityService` with registration cooldown and cache, `VisitSessionManager` with entry/exit state machine, `EvidenceStorage` with atomic crop saving, `EventLogger` with rotating file logs, `MetricsCollector`, `RecoveryChecker`, and `src/main.py` pipeline loop.
- **Files Created**:
  - `src/identity_service.py`
  - `src/visit_session.py`
  - `src/evidence_storage.py`
  - `src/event_service.py`
  - `src/event_logger.py`
  - `src/metrics.py`
  - `src/recovery.py`
  - `src/main.py`
- **Verified Outcome**:
  - Clean modular architecture without circular dependencies.
  - Full support for headless mode, OpenCV visual overlay, and graceful shutdown signal handlers (SIGINT/SIGTERM).

---

## Prompt 6: Automated Test Suite Implementation & Verification
- **Goal**: Implement unit tests (`tests/unit/`), end-to-end tests (`tests/e2e/`), and database integration tests (`tests/integration/`). Execute test runner and resolve all discovered defects.
- **Files Created**:
  - `tests/unit/test_config.py`
  - `tests/unit/test_detector.py`
  - `tests/unit/test_recognizer.py`
  - `tests/unit/test_tracker.py`
  - `tests/unit/test_visit_session.py`
  - `tests/unit/test_evidence_storage.py`
  - `tests/e2e/test_pipeline.py`
  - `tests/integration/test_database.py`
- **Defects Fixed**:
  - Missing `EventLogger` class in `event_logger.py` implemented.
  - Tracker track promotion logic adjusted to pass `min_age` and handle `LOST` states on tentative tracks.
  - `EvidenceStorage` parameter flexibility and `_build_path` helper method added.
  - Non-ASCII characters sanitized to prevent Windows encoding syntax errors.
- **Verified Outcome**:
  - **41/41 Unit Tests PASSED.**
  - **6/6 End-to-End Pipeline Tests PASSED.**
  - **Total 47 automated tests passed.**

---

## Prompt 7: CLI Scripts, Environment Check, and Documentation
- **Goal**: Implement runnable CLI scripts (`check_environment.py`, `init_database.py`, `run_sample.py`, `run_rtsp.py`, `verify_integrity.py`, `export_sample_results.py`) and write comprehensive documentation.
- **Files Created**:
  - `scripts/check_environment.py`
  - `scripts/init_database.py`
  - `scripts/run_sample.py`
  - `scripts/run_rtsp.py`
  - `scripts/verify_integrity.py`
  - `scripts/export_sample_results.py`
  - `docs/AI_PLANNING.md`
  - `docs/ARCHITECTURE.md`
  - `docs/DESIGN_DECISIONS.md`
  - `docs/TESTING.md`
  - `docs/COMPUTE_ESTIMATES.md`
  - `docs/PROMPT_LOG.md`
  - `docs/INTERVIEW_GUIDE.md`
  - `README.md`
- **Verified Outcome**:
  - Environment checker validated all packages and flagged database auth requirement.
  - All required README sections complete, concluding with the mandatory Katomaran statement.

---

## Prompt 8: PostgreSQL 18 Migration & Integration Testing
- **Goal**: Apply Alembic schema migrations to live PostgreSQL 18 database, resolve enum mapping discrepancies, and execute database integration tests.
- **Commands Executed**:
  - `alembic upgrade head`
  - `pytest tests/integration/test_database.py -v`
- **Defects Fixed**:
  - PostgreSQL enum value case mismatch: SQLAlchemy defaulted to enum names (`FILE`, `CONFIRMED`, `ACTIVE`, `ENTRY`) whereas PostgreSQL type definitions used lowercase values (`file`, `confirmed`, `active`, `entry`). Configured `values_callable` across all ORM models to map correctly.
- **Verified Outcome**:
  - Initial schema migration applied cleanly to `sentineltrack` database.
  - **5/5 Integration Tests PASSED**: Camera creation, face registration, partial unique index duplicate entry prevention, idempotency key enforcement, and unique identity counting.

---

## Prompt 9: Pipeline Harmonization, Acceptance Suite (A-G), and Submission Verification
- **Goal**: Harmonize component method signatures across `IdentityService`, `VisitSessionManager`, and `EventService`. Implement comprehensive acceptance test suite `tests/e2e/test_acceptance_scenario.py` verifying the Katomaran mandatory sequence A through G. Generate realistic sample video with detectable faces and execute the live pipeline.
- **Commands Executed**:
  - `pytest tests/e2e/test_acceptance_scenario.py -v`
  - `pytest tests/ -v`
  - `python scripts/generate_sample_video.py`
  - `python scripts/run_sample.py --headless`
  - `python scripts/verify_integrity.py`
  - `python scripts/export_sample_results.py`
- **Defects Fixed**:
  - Interface harmonization: Aligned `identify()`, `on_face_seen()`, and `save_evidence()` argument signatures.
  - Savepoint transaction protection: Wrapped `EventRepository.create_event` with `session.begin_nested()` to prevent duplicate key conflicts from aborting outer transactions during retry testing.
  - Timestamp order constraint protection: Enforced `exit_timestamp >= entry_timestamp` in `SessionRepository.confirm_exit` to honor the check constraint `ck_session_timestamps_order`.
  - InsightFace crop fallback: Added fallback from tight crop to full frame in `FaceRecognizer.generate_embedding` to ensure robust landmark alignment on varied resolutions.
- **Verified Outcome**:
  - **54/54 automated tests PASSED** (unit, integration, and e2e) with 0 failures and 0 warnings.
  - `data/sample.mp4` generated with realistic face portraits (Ada Lovelace and Albert Einstein).
  - Pipeline executed 110 frames: 40 detections, 78 recognitions, 2 entries, 1 exit, 0 errors, 8.62 FPS.
  - Evidence images saved in date-partitioned directories `logs/entries/` and `logs/exits/`.
  - Database and filesystem consistency verified clean via `verify_integrity.py`.
  - Submission output exported to `docs/sample_output.md`.

