# AI Planning Document — SentinelTrack Face Tracking System

> **Hackathon Project** | Face Recognition & Tracking Pipeline  
> **Author:** AI-Assisted Engineering (Claude Sonnet + Gemini)  
> **Last Updated:** October 2026

---

## Table of Contents

1. [Project Understanding & Requirements Analysis](#1-project-understanding--requirements-analysis)
2. [Technology Alternatives Considered](#2-technology-alternatives-considered)
3. [Architecture Decisions](#3-architecture-decisions)
4. [Implementation Phases](#4-implementation-phases)
5. [Risk Mitigation Strategies](#5-risk-mitigation-strategies)
6. [AI Tools Used & How](#6-ai-tools-used--how)
7. [Prompt Engineering Approach](#7-prompt-engineering-approach)

---

## 1. Project Understanding & Requirements Analysis

### 1.1 Hackathon Specification Breakdown

The hackathon specification describes a **real-time face tracking and recognition system** that must:

1. **Ingest video** from a file path or an RTSP camera stream.
2. **Detect faces** in each frame using a modern neural-network-based detector.
3. **Track face instances** across frames, maintaining stable track IDs even when a face briefly leaves the frame.
4. **Register unknown identities** by extracting 512-dimensional embedding vectors and persisting them.
5. **Recognize known identities** by comparing live embeddings against the registered database using cosine similarity.
6. **Generate Events** — specifically ENTRY and EXIT events — tied to real calendar sessions (visit start time, end time, duration).
7. **Store evidence** — cropped face images — alongside each event for audit purposes.
8. **Expose a REST API** so external systems (dashboards, integrations) can query persons, sessions, and events.
9. **Log an audit trail** to a structured `events.log` file in addition to the database.
10. **Maintain operational metrics** (frame counts, latency, FPS, detection counts) accessible at runtime.

### 1.2 Implicit Requirements Derived from the Specification

Beyond the literal spec, the following implicit requirements were identified:

| Implicit Requirement | Reasoning |
|---|---|
| **Idempotent event creation** | Live video can produce duplicate detections; the system must not create duplicate ENTRY/EXIT pairs for a single visit. |
| **Graceful degradation when GPU is absent** | Target machine is Windows with no GPU. All inference paths must run on CPU. |
| **Thread safety** | The video loop, API server, and database writer run concurrently and share state. |
| **Cold-start tolerance** | On first run, the embedding database is empty; the system must handle registrations gracefully. |
| **Restart resilience** | If the process crashes, ongoing sessions should be recoverable from the database. |
| **Windows path compatibility** | All file I/O must use `pathlib.Path` and forward-slash-safe string operations. |
| **No Docker** | PostgreSQL 18 is running natively on Windows; connection strings point to `localhost`. |

### 1.3 Task Decomposition

Breaking the spec into concrete engineering tasks:

```
PHASE 1:  Project scaffolding, config, logging
PHASE 2:  Database models (SQLAlchemy), Alembic migrations
PHASE 3:  YOLOv8 face detection wrapper
PHASE 4:  InsightFace ArcFace embedding extraction
PHASE 5:  IoU-based face tracker (ByteTrack-style state machine)
PHASE 6:  Identity service (register, recognize, cache)
PHASE 7:  Session manager (entry/exit state machine)
PHASE 8:  Event pipeline coordinator (EventService)
PHASE 9:  REST API (FastAPI)
PHASE 10: Main video loop + metrics + CLI entry point
```

Each phase produces one or more source files. No phase proceeds until the previous phase's interfaces are defined (contract-first development).

---

## 2. Technology Alternatives Considered

### 2.1 Face Detection

#### Option A: YOLOv8 (Selected ✅)

**Package:** `ultralytics`  
**Model used:** `yolov8n-face.pt` (nano variant for CPU speed)

**Pros:**
- Single-stage detector — very fast even on CPU compared to two-stage methods.
- `ultralytics` package installs cleanly on Windows with `pip install ultralytics`.
- The nano model (`yolov8n`) runs at ~150–400ms/frame on a mid-range CPU without GPU.
- Returns bounding boxes and confidence scores in a consistent numpy-compatible format.
- Actively maintained; PyTorch 2.x compatible.
- Supports ONNX export for future GPU/NPU acceleration with zero code changes.

**Cons:**
- Requires downloading model weights on first run (~6MB for nano).
- Not specialized for faces out-of-the-box; needs the `-face` variant or fine-tuning.

#### Option B: RetinaFace

**Pros:** Designed specifically for face detection; excellent at small faces and crowds.  
**Cons:**
- `retinaface` Python package has mixed Windows compatibility (requires `mxnet` or `tensorflow` backend).
- MXNet is deprecated and difficult to install on Windows 11.
- The `insightface` implementation of RetinaFace requires ONNX Runtime, adding another dependency.

**Verdict:** Rejected due to Windows installation friction.

#### Option C: MTCNN (Multi-Task Cascaded CNNs)

**Pros:** Very lightweight; specifically designed for face detection.  
**Cons:**
- Much slower than YOLO on full-resolution frames due to cascaded architecture.
- Poor performance on small faces without image pyramids.
- `facenet-pytorch` MTCNN is the cleanest implementation, but pulls in the full PyTorch stack.

**Verdict:** Rejected. YOLOv8 is faster and more accurate on varied real-world video.

---

### 2.2 Face Recognition / Embedding Extraction

#### Option A: InsightFace + ArcFace (Selected ✅)

**Package:** `insightface`  
**Model:** `buffalo_l` or `buffalo_sc` (ArcFace backbone, ResNet50/MobileNet)

**Pros:**
- ArcFace produces 512-dimensional L2-normalized embeddings that are highly discriminative.
- `insightface` ships with ONNX models — no MXNet required on modern versions.
- ONNX Runtime runs on CPU on Windows natively.
- Cosine similarity between normalized ArcFace embeddings is a well-understood, tunable metric.
- Produces reliable recognition across lighting variation and mild pose changes.

**Cons:**
- ONNX Runtime CPU inference is 80–200ms per face crop.
- Model download on first run (~300MB for `buffalo_l`).

#### Option B: FaceNet (Google)

**Pros:** Well-known 128/512D embedding space; good accuracy.  
**Cons:**
- Original TensorFlow 1.x implementation; `facenet-pytorch` port requires extra setup.
- Embedding space differs from ArcFace; harder to tune thresholds.
- Less actively maintained than InsightFace.

**Verdict:** Rejected. InsightFace has better Windows/ONNX support.

#### Option C: DeepFace

**Pros:** Wraps multiple backends (VGG-Face, ArcFace, Facenet, etc.); easy API.  
**Cons:**
- High-level abstraction hides the embedding vector; hard to do custom similarity logic.
- Adds a large dependency tree (TensorFlow or Keras required).
- Performance is uncontrolled — cannot tune model size.

**Verdict:** Rejected. We need direct access to the embedding vector for cosine similarity in Python.

---

### 2.3 Multi-Object Tracking

#### Option A: IoU-Based Tracker — ByteTrack Style (Selected ✅)

**Implementation:** Custom Python, no external package required.

The tracker maintains a list of `Track` objects, each with:
- A bounding box (current and predicted)
- An age counter (frames since last matched)
- A state machine: `TENTATIVE → CONFIRMED → LOST → DELETED`

Each frame:
1. Hungarian algorithm (or greedy IoU matching) assigns detections to existing tracks.
2. Unmatched detections start new `TENTATIVE` tracks.
3. Unmatched tracks age; after `max_age` frames unmatched, tracks transition to `LOST`.

**Pros:**
- Zero extra dependencies — pure Python with NumPy.
- Predictable behavior; easy to debug and tune.
- Works entirely on CPU with negligible overhead.
- Controllable `max_age` and `min_hits` thresholds for the hackathon use case.

**Cons:**
- No motion prediction (no Kalman filter). Faces must not move extremely fast.
- Re-identification across long occlusions requires the face recognition layer.

#### Option B: DeepSORT

**Pros:** Uses Kalman filter for motion prediction + appearance re-ID.  
**Cons:**
- `deep_sort_realtime` package works on Linux easily; Windows pip install is fragile.
- Requires a separate Re-ID model (an additional ONNX/PyTorch model).
- Adds significant complexity: Kalman filter state, separate Re-ID inference pass.

**Verdict:** Rejected. The face recognition layer already provides re-identification. Using IoU tracks + ArcFace recognition is effectively equivalent to DeepSORT but with cleaner separation of concerns.

#### Option C: ByteTrack (Official Implementation)

**Pros:** State-of-the-art multi-object tracker; low-confidence detection retention.  
**Cons:**
- The `bytetracker` PyPI package is experimental; C++ extensions may not compile on Windows.
- Requires CUDA for the intended use case (high-FPS video).

**Verdict:** Rejected. We implement the core ByteTrack insight (two-stage matching) manually without the fragile package.

---

### 2.4 Database & Vector Storage

#### Option A: PostgreSQL 18 + BYTEA Fallback (Selected ✅)

Embeddings stored as `BYTEA` blobs (serialized `numpy` float32 arrays). Cosine similarity computed in Python after fetching all known embeddings into memory.

**Pros:**
- PostgreSQL 18 is already running natively on Windows — zero setup cost.
- BYTEA works on any PostgreSQL version without extensions.
- For < 10,000 registered identities, in-memory cosine similarity is fast enough (< 5ms with NumPy vectorization).
- Full ACID compliance for event records.
- The in-memory embedding cache (`LRUCache`) means the database is only read once per person, not on every recognition attempt.

**Cons:**
- Does not scale to millions of embeddings — would need pgvector or FAISS index.
- Cosine similarity is done in Python, not pushed to the DB query layer.

#### Option B: pgvector Extension

**Pros:** Native vector similarity search inside PostgreSQL (`<->` operator).  
**Cons:**
- Requires `CREATE EXTENSION vector;` — only available if pgvector is compiled for PostgreSQL 18.
- pgvector binaries for PostgreSQL 18 on Windows are not reliably available at time of writing.
- The system already tries `pgvector` at startup and falls back to `BYTEA` automatically.

**Verdict:** Supported as a progressive enhancement. The system detects pgvector at startup. If present, the `face_embedding` column is typed `vector(512)`; otherwise it falls back to `BYTEA`.

#### Option C: SQLite

**Pros:** Zero setup; single file database.  
**Cons:**
- No concurrent writer support (WAL mode helps but is not production-grade for multi-thread inserts).
- No native `BYTEA`-equivalent that's efficient for 512×4 = 2048-byte blobs.
- Cannot use pgvector.
- Hackathon spec implies PostgreSQL.

**Verdict:** Rejected. PostgreSQL is mandated by the spec.

---

## 3. Architecture Decisions

### 3.1 Layered Architecture

The codebase is organized into clearly separated layers:

```
sentineltrack/
├── config.py          # Pydantic-validated config (environment / .env)
├── models.py          # SQLAlchemy ORM models (source of truth for schema)
├── database.py        # Engine, session factory, pgvector detection
├── detector.py        # YOLOv8 face detector wrapper
├── embedder.py        # InsightFace ArcFace embedding extractor
├── tracker.py         # IoU tracker with state machine
├── identity_service.py# Registration, recognition, embedding cache
├── session_manager.py # Visit session state machine (ENTRY → EXIT)
├── event_service.py   # Pipeline coordinator tying all layers together
├── evidence_store.py  # Atomic crop image writer
├── api.py             # FastAPI REST endpoints
├── metrics.py         # MetricsCollector (in-memory counters, timing)
├── main.py            # CLI entry point, video loop
└── migrations/        # Alembic migration scripts
```

This layering means each component can be unit-tested in isolation. The `EventService` acts as the only orchestrator — no component calls another directly across layer boundaries.

### 3.2 Pydantic Config Validation

All runtime parameters are defined in `config.py` as a `Pydantic BaseSettings` model. This means:
- Invalid config fails fast at startup with a human-readable error.
- Environment variables or a `.env` file can override any parameter.
- No magic strings scattered through source files.

### 3.3 Thread Safety Strategy

| Component | Threading Approach |
|---|---|
| Video loop | Main thread (OpenCV `VideoCapture` is not thread-safe) |
| FastAPI server | `uvicorn` worker thread (via `threading.Thread`) |
| Database writes | `scoped_session` per thread |
| Embedding cache | `threading.Lock` around LRU dict |
| Metrics | `threading.Lock` around all counters |

### 3.4 Idempotent Event Creation

PostgreSQL partial unique indexes ensure that:
- Only one ENTRY event can exist per `(face_id, DATE(occurred_at))`.
- Only one EXIT event can exist per `(face_id, session_id)`.

This means the application can safely retry event creation without worrying about duplicates — the database enforces uniqueness as a constraint.

---

## 4. Implementation Phases

### Phase 1 — Project Scaffold & Configuration
**Files:** `config.py`, `__init__.py`, `.env.example`, `requirements.txt`  
**Goal:** Validated config loading; logging setup; project installable as a package.

### Phase 2 — Database Models & Migrations
**Files:** `models.py`, `database.py`, `migrations/`  
**Goal:** SQLAlchemy ORM models for `faces`, `visit_sessions`, `events`, `audit_log`; Alembic migration creating all tables and indexes.

### Phase 3 — YOLOv8 Face Detector
**Files:** `detector.py`  
**Goal:** Wrapper around `ultralytics.YOLO` that takes a BGR numpy frame and returns a list of `DetectedFace(bbox, confidence)` objects.

### Phase 4 — InsightFace ArcFace Embedder
**Files:** `embedder.py`  
**Goal:** Wrapper around `insightface.app.FaceAnalysis` that takes a face crop and returns a 512D float32 numpy array, L2-normalized.

### Phase 5 — IoU Face Tracker
**Files:** `tracker.py`  
**Goal:** Stateful tracker mapping detection bboxes → stable track IDs across frames; implements TENTATIVE/CONFIRMED/LOST state machine.

### Phase 6 — Identity Service
**Files:** `identity_service.py`  
**Goal:** `register_face()` and `recognize_face()` functions backed by PostgreSQL, with an LRU in-memory cache of embeddings for fast recognition.

### Phase 7 — Session Manager
**Files:** `session_manager.py`  
**Goal:** State machine converting track lifecycle events (first confirmed detection, last seen frame) into `visit_session` records with start/end timestamps.

### Phase 8 — Event Pipeline
**Files:** `event_service.py`, `evidence_store.py`  
**Goal:** Orchestrates detection → tracking → recognition → session management → event creation → evidence saving per-frame.

### Phase 9 — REST API
**Files:** `api.py`  
**Goal:** FastAPI application with endpoints: `GET /faces`, `GET /faces/{id}`, `GET /sessions`, `GET /events`, `POST /register`, `GET /metrics`, `GET /health`.

### Phase 10 — Main Loop & CLI
**Files:** `main.py`  
**Goal:** `argparse`/`typer` CLI; OpenCV video loop; graceful shutdown on SIGINT; final metrics summary written to `logs/last_run_summary.json`.

---

## 5. Risk Mitigation Strategies

### 5.1 InsightFace Model Download Failure
**Risk:** Model download fails in environments without internet access or behind a corporate proxy.  
**Mitigation:** The `embedder.py` wrapper catches download errors and raises a clear `RuntimeError` with instructions to manually copy model files to `~/.insightface/models/`.

### 5.2 PostgreSQL Connection Failure
**Risk:** PostgreSQL 18 service not running at startup.  
**Mitigation:** `database.py` retries connection 5 times with exponential backoff before raising. Error message includes the connection string (with password masked) and a link to the Windows Services checklist.

### 5.3 YOLOv8 Model Not Found
**Risk:** `yolov8n-face.pt` not in working directory or cache.  
**Mitigation:** `detector.py` attempts to load from the configured path, then falls back to auto-download via `ultralytics`. If both fail, raises `FileNotFoundError` with clear instructions.

### 5.4 Concurrent Write Conflicts
**Risk:** Two threads simultaneously creating a session for the same `face_id`.  
**Mitigation:** PostgreSQL partial unique index on `(face_id, DATE(start_time))` for ENTRY events. Application-level `ON CONFLICT DO NOTHING` pattern; the event is simply skipped if already created.

### 5.5 Memory Growth from Embedding Cache
**Risk:** If tens of thousands of faces are registered, the in-memory cache grows without bound.  
**Mitigation:** `LRUCache` is bounded by `identity_cache_size` config parameter (default: 1000 entries). Evicted entries are re-fetched from PostgreSQL on next recognition attempt.

### 5.6 Frame Drop Under Load
**Risk:** Processing takes longer than frame interval; input buffer fills up.  
**Mitigation:** `frame_skip` config parameter (default: 2) processes every Nth frame. The video loop always reads (discards) frames to prevent buffer stagnation.

### 5.7 Windows File Locking for Evidence Images
**Risk:** Evidence image is partially written when another thread reads it.  
**Mitigation:** `evidence_store.py` writes to a `.tmp` file first, then `os.rename()` atomically. Windows does not guarantee atomic rename across drives, so source and destination are always on the same drive.

---

## 6. AI Tools Used & How

### 6.1 Primary AI Assistant: Claude Sonnet (Anthropic)

**Role:** Senior software engineer pair-programming partner.

Claude was used to:
1. **Generate complete source files** — not stubs or pseudocode, but production-quality Python with full imports, docstrings, type hints, and error handling.
2. **Review architectural decisions** — when multiple approaches were presented, Claude provided pros/cons reasoning (as documented in Section 2).
3. **Debug integration issues** — when components were wired together, Claude identified threading issues and import cycles.
4. **Write test scaffolding** — unit test templates for each service layer.
5. **Write all documentation** — including this file.

**Limitations observed:**
- Claude's training cutoff means some package APIs may be slightly outdated (e.g., `ultralytics` 8.x API changes). Generated code was verified against the installed package's actual API.
- Claude cannot run code; all generated code was executed and debugged by the human developer.

### 6.2 Secondary AI Assistant: Gemini (Google DeepMind)

**Role:** Orchestrator and planning agent.

Gemini was used as the parent agent coordinating the hackathon build:
- Maintained the global task list and determined the order of file generation.
- Delegated specific file-writing tasks to Claude subagents.
- Performed final consistency checks across all files.

### 6.3 AI Assistance Transparency

In accordance with hackathon rules on AI disclosure:

| Task | Human | AI |
|---|---|---|
| Requirements interpretation | ✅ Primary | ✅ Advisory |
| Architecture design | ✅ Primary | ✅ Advisory |
| Code generation | ✅ Review/Edit | ✅ Primary draft |
| Debugging | ✅ Primary | ✅ Advisory |
| Documentation | ✅ Review | ✅ Primary draft |
| Testing | ✅ Primary | ✅ Template |

All AI-generated code was reviewed, tested, and modified by the human developer before submission.

---

## 7. Prompt Engineering Approach

### 7.1 System Prompt Design

The AI subagents were given a system prompt establishing:

```
- Role: Senior Python software engineer
- Project: SentinelTrack hackathon face tracking system
- Constraints: Python 3.10.11, PostgreSQL 18 on Windows, no GPU assumed
- Libraries: YOLOv8, InsightFace, SQLAlchemy 2.x, FastAPI, Alembic
- Code quality: Production-quality, not tutorial quality
- Every feature must be fully implemented (no TODOs, no stubs)
```

The role framing ("senior engineer") was critical — without it, Claude tended to produce overly simplified tutorial-style code with placeholder comments.

### 7.2 Instruction Specificity

Vague prompts like "write a face detector" produced mediocre results. Effective prompts specified:

1. **Exact file path** — where the file should be written.
2. **Exact class/function signatures** — what the public API must look like.
3. **Integration contracts** — what other modules this file imports from and exports to.
4. **Error handling requirements** — what exceptions to raise and when.
5. **Config integration** — which `AppConfig` fields this module reads.

Example of a high-quality prompt excerpt:
> "Write `detector.py`. It must define a `FaceDetector` class initialized with `AppConfig`. The `detect(frame: np.ndarray) -> list[DetectedFace]` method must return bounding boxes as `[x1, y1, x2, y2]` in pixel coordinates. Confidence threshold must be read from `config.detection_confidence`. Handle the case where ultralytics returns no detections gracefully (return empty list, do not raise)."

### 7.3 Iterative Refinement

Initial code generation was followed by targeted refinement prompts:
- "The `recognize_face` method doesn't handle the case where the embedding database is empty. Fix that."
- "Thread safety: the `_embedding_cache` dict is accessed from multiple threads. Add a `threading.Lock`."
- "The `session_manager.py` exit confirmation uses `time.time()` but should use `datetime.utcnow()` for consistency with SQLAlchemy timestamps."

### 7.4 Consistency Enforcement

Because multiple AI sessions generated different files, a final consistency pass was performed:
- All `from sentineltrack.config import AppConfig` imports match the actual exported name.
- All `DetectedFace`, `TrackState`, `RecognitionResult` dataclass definitions are imported from the correct source file.
- All `logger = logging.getLogger(__name__)` patterns are consistent.

### 7.5 Lessons Learned for Future Hackathons

1. **Define data contracts first** — shared dataclasses (`DetectedFace`, `TrackedFace`, `RecognitionResult`) should be written before any module that uses them.
2. **Specify imports explicitly** — tell the AI exactly what it should import from where. Otherwise it invents module names.
3. **Include negative constraints** — "do NOT use `global` variables", "do NOT import from `__future__`", "do NOT use `print()` — use `logger`".
4. **Request error handling explicitly** — AI defaults to happy-path code unless told otherwise.
5. **Validate immediately** — run each generated file as soon as it's written; don't accumulate 10 files and then debug.

---

## Appendix A: Dependencies Summary

| Package | Version Pinned | Purpose |
|---|---|---|
| `ultralytics` | `>=8.0` | YOLOv8 face detection |
| `insightface` | `>=0.7` | ArcFace face recognition |
| `onnxruntime` | `>=1.16` | ONNX inference for InsightFace |
| `sqlalchemy` | `>=2.0` | ORM for PostgreSQL |
| `alembic` | `>=1.12` | Database migrations |
| `psycopg2-binary` | `>=2.9` | PostgreSQL adapter |
| `fastapi` | `>=0.104` | REST API framework |
| `uvicorn` | `>=0.24` | ASGI server for FastAPI |
| `pydantic-settings` | `>=2.0` | Config validation |
| `opencv-python` | `>=4.8` | Video capture & image I/O |
| `numpy` | `>=1.24` | Numeric operations |
| `scipy` | `>=1.11` | Cosine distance (fallback) |
| `psutil` | `>=5.9` | CPU/memory metrics |
| `python-dotenv` | `>=1.0` | `.env` file loading |

---

## Appendix B: File Count Summary

| Phase | Files Created | Lines (approx.) |
|---|---|---|
| 1 — Config | 3 | 200 |
| 2 — Database | 3 | 400 |
| 3 — Detector | 1 | 180 |
| 4 — Embedder | 1 | 160 |
| 5 — Tracker | 1 | 300 |
| 6 — Identity | 1 | 350 |
| 7 — Sessions | 1 | 280 |
| 8 — Pipeline | 2 | 400 |
| 9 — API | 1 | 250 |
| 10 — Main | 1 | 200 |
| Docs | 4 | 1200+ |
| **Total** | **19** | **~3920** |

---

*This document was produced with AI assistance (Claude Sonnet) and reviewed by the human developer. All design decisions reflect actual implementation choices in the codebase.*
