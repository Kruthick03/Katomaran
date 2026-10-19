# SentinelTrack — AI-Powered Unique Visitor Counter

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18-336791?logo=postgresql&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?logo=yolo&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Windows%2011-0078D6?logo=windows&logoColor=white)
![InsightFace](https://img.shields.io/badge/InsightFace-ArcFace-red)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Features](#3-features)
4. [Technology Stack](#4-technology-stack)
5. [Architecture](#5-architecture)
6. [Prerequisites](#6-prerequisites)
7. [Windows Setup Instructions](#7-windows-setup-instructions)
8. [PostgreSQL Setup](#8-postgresql-setup-windows-native)
9. [Environment Configuration](#9-environment-configuration)
10. [Model Installation](#10-model-installation)
11. [Database Migration](#11-database-migration)
12. [Running the Application](#12-running-the-application)
13. [Testing](#13-testing)
14. [Sample Database Queries](#14-sample-database-queries)
15. [Example Log Output](#15-example-log-output)
16. [Unique Visitor vs Session Count](#16-unique-visitor-vs-session-count)
17. [Performance Estimates](#17-performance-estimates)
18. [Assumptions](#18-assumptions)
19. [Known Limitations](#19-known-limitations)
20. [Troubleshooting](#20-troubleshooting)
21. [Privacy and Retention Policy](#21-privacy-and-retention-policy)
22. [Demonstration Video](#22-demonstration-video)
23. [GitHub Submission Checklist](#23-github-submission-checklist)

---

## 1. Project Overview

**SentinelTrack** is a real-time, AI-driven unique visitor counting system built for camera-based environments such as retail stores, event halls, office lobbies, and surveillance installations. It uses computer vision and deep learning to detect, recognize, and persistently track individual faces across video frames — distinguishing between new visitors and returning ones without any manual tagging or check-in process.

The system ingests video from either a local file, webcam, or RTSP network camera stream, runs YOLOv8 face detection on each frame, extracts 512-dimensional ArcFace embeddings via InsightFace, and matches them against a PostgreSQL-backed identity store. Each confirmed unique visitor is assigned a stable UUID, and every entry/exit event is recorded with a timestamp and optional evidence image snapshot. Operators can query the database at any time for unique visitor counts, dwell times, session histories, and more.

SentinelTrack is designed to run entirely on **CPU** (no GPU required), making it deployable on commodity hardware and edge devices without specialized accelerators.

---

## 2. Problem Statement

This project was submitted as part of the **Katomaran Hackathon** challenge, which requires participants to build an intelligent system capable of:

- **Detecting and counting unique individuals** who appear in a camera feed — not just face detections, but distinct people across multiple appearances.
- **Tracking entries and exits** so that a person who leaves and re-enters is counted once as a unique visitor but logged as a new visit session.
- **Persisting all data** to a relational database so that reports, audits, and analytics can be run at any time.
- **Operating without human supervision** — the system must automatically register new faces, match returning faces, and manage all session state transitions.

Manual head counts and badge-scan systems are slow, error-prone, and require visitor cooperation. SentinelTrack eliminates those limitations through fully automated, passive biometric identification.

---

## 3. Features

| Feature | Description |
|---|---|
| **Real-Time Face Detection** | YOLOv8n-face model detects faces in every video frame with bounding box and confidence score |
| **ArcFace Embedding Extraction** | InsightFace `buffalo_sc` model computes 512-dim face embeddings for identity comparison |
| **ByteTrack-Style Multi-Object Tracking** | Custom IoU-based tracker assigns stable in-frame track IDs, reducing redundant embedding calls |
| **Cosine Similarity Identity Matching** | Embeddings compared with cosine distance threshold; configurable match / no-match boundary |
| **Automatic Face Registration** | New identities are auto-registered to PostgreSQL with UUID, embedding vector, and metadata |
| **Entry / Exit State Machine** | Finite state machine per identity tracks presence windows; raises `entry` and `exit` events |
| **Dwell Time Calculation** | Each visit session records start time, end time, and computed duration in seconds |
| **Evidence Image Snapshots** | Cropped face images saved to disk at registration and significant events, path stored in DB |
| **Structured Event Logging** | All events written to `events.log` in structured format with ISO timestamps and log levels |
| **PostgreSQL Persistence** | All faces, embeddings, events, and sessions stored in PostgreSQL 18 with pgvector support |
| **pgvector / BYTEA Fallback** | If pgvector extension is available, uses native vector indexing; otherwise falls back to BYTEA |
| **Alembic Schema Migrations** | Database schema managed with Alembic, allowing reproducible version-controlled migrations |
| **RTSP Stream Support** | Accepts RTSP URLs (e.g., IP cameras) in addition to local files and webcam indices |
| **Configurable Thresholds** | All key parameters (detection confidence, similarity threshold, exit timeout) set via `config.json` |
| **CLI Entry Points** | Multiple runnable scripts for sample video, RTSP, and database initialization |
| **Unit + Integration Tests** | pytest test suite covering detection, tracking, identity service, and database operations |
| **CPU-Only Operation** | No CUDA/GPU dependency; all models run via ONNX on CPU with acceptable frame rate |

---

## 4. Technology Stack

| Layer | Technology | Version / Notes |
|---|---|---|
| **Language** | Python | 3.10.11 |
| **Face Detection** | Ultralytics YOLOv8 | `yolov8n-face.pt` model |
| **Face Recognition** | InsightFace + ArcFace | `buffalo_sc` ONNX model |
| **Multi-Object Tracking** | ByteTrack-style (custom) | IoU-based Kalman tracker |
| **Database** | PostgreSQL | 18 (Windows native) |
| **Vector Extension** | pgvector | Optional; BYTEA fallback |
| **ORM** | SQLAlchemy | 2.x |
| **Migrations** | Alembic | Latest |
| **DB Driver** | psycopg2 | psycopg2-binary |
| **Numerical Computing** | NumPy | ≥ 1.24 |
| **Image Processing** | OpenCV | opencv-python-headless |
| **Configuration** | python-dotenv + JSON | `.env` + `config.json` |
| **Testing** | pytest + pytest-cov | Latest |
| **Logging** | Python stdlib `logging` | Structured to `events.log` |
| **Similarity** | scipy / NumPy cosine | Cosine distance metric |

---

## 5. Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SentinelTrack Pipeline                          │
└─────────────────────────────────────────────────────────────────────────┘

  Video Source (file / webcam / RTSP)
         │
         ▼
  ┌─────────────────┐
  │  Frame Grabber  │  — OpenCV VideoCapture, configurable FPS cap
  └────────┬────────┘
           │  raw frame (BGR ndarray)
           ▼
  ┌─────────────────────────┐
  │  YOLO Face Detector     │  — YOLOv8n-face, confidence threshold filter
  └────────────┬────────────┘
               │  bounding boxes + confidence scores
               ▼
  ┌─────────────────────────┐
  │  ByteTrack-Style        │  — IoU matching + Kalman prediction
  │  Multi-Object Tracker   │    assigns stable track_id per in-frame face
  └────────────┬────────────┘
               │  tracked detections (track_id, bbox, frame crop)
               ▼
  ┌─────────────────────────┐
  │  InsightFace / ArcFace  │  — buffalo_sc ONNX, 512-dim embedding
  │  Embedding Extractor    │    runs only on NEW or LOST tracks
  └────────────┬────────────┘
               │  face_embedding (float32 array, 512-dim)
               ▼
  ┌─────────────────────────────────────────────────────┐
  │               Identity Service                      │
  │   cosine_similarity(embedding, stored_embeddings)   │
  │         threshold comparison (configurable)         │
  └──────────────┬──────────────────────────┬───────────┘
                 │                          │
         [similarity < threshold]   [similarity ≥ threshold]
         New Face                         Known Face
              │                                │
              ▼                                ▼
   ┌──────────────────┐            ┌───────────────────────┐
   │  Register Face   │            │  Match + Update Last  │
   │  (INSERT faces)  │            │  Seen / Embedding     │
   │  Save evidence   │            │  (UPDATE faces)       │
   │  image snapshot  │            └───────────┬───────────┘
   └────────┬─────────┘                        │
            │                                  │
            └──────────────┬───────────────────┘
                           │  face_id (UUID)
                           ▼
            ┌──────────────────────────────┐
            │   Visit Session Manager      │
            │   (Entry / Exit FSM)         │
            │                              │
            │  ABSENT ──[seen]──▶ PRESENT  │
            │  PRESENT ──[timeout]──▶ EXIT │
            │  EXIT ──[seen again]──▶ NEW  │
            │                   SESSION    │
            └──────────────┬───────────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
   PostgreSQL DB      Evidence         events.log
   (faces, events,    Images           (structured
   visit_sessions)    (disk)            text log)
```

---

## 6. Prerequisites

Before setting up SentinelTrack, ensure the following are installed and available on your **Windows 11** machine:

| Requirement | Minimum Version | Notes |
|---|---|---|
| **Windows** | Windows 10 / 11 | 64-bit required |
| **Python** | 3.10.x | 3.10.11 recommended; do **not** use 3.11+ |
| **PostgreSQL** | 18 | Windows native installer (not Docker) |
| **Git** | Any recent | For cloning the repository |
| **Visual C++ Build Tools** | 2019 or later | Required for psycopg2 and some native extensions |
| **RAM** | 8 GB minimum | 16 GB recommended for multi-stream use |
| **CPU** | 4-core minimum | Intel Core i5 or AMD Ryzen 5 class |
| **Disk** | 5 GB free | For models, evidence images, and database |

> **Python version note:** Python 3.10 is specifically required because InsightFace pre-built wheels and some ONNX Runtime versions are not reliably available for Python 3.11+ on Windows at the time of this writing.

---

## 7. Windows Setup Instructions

### Step 1: Clone the Repository

```powershell
git clone https://github.com/<your-username>/sentineltrack.git
cd sentineltrack
```

### Step 2: Create and Activate Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Verify the correct Python version is active:

```powershell
python --version
# Expected: Python 3.10.x
```

### Step 3: Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

> If `insightface` or `onnxruntime` fail to install, try installing Visual C++ Redistributable and Build Tools from [https://visualstudio.microsoft.com/downloads/](https://visualstudio.microsoft.com/downloads/) first.

### Step 4: Configure Environment Variables

```powershell
copy .env.example .env
```

Open `.env` in a text editor and fill in your PostgreSQL credentials and any other required values (see [Environment Configuration](#9-environment-configuration) below).

### Step 5: Configure Application Settings

```powershell
copy config.example.json config.json
```

Edit `config.json` to set detection thresholds, similarity cutoffs, source paths, and other pipeline parameters.

---

## 8. PostgreSQL Setup (Windows Native)

SentinelTrack uses **PostgreSQL 18 installed natively on Windows** — not Docker.

### Step 1: Download and Install PostgreSQL 18

Download the Windows installer from the official EnterpriseDB page:
[https://www.enterprisedb.com/downloads/postgres-postgresql-downloads](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads)

During installation:
- Set a strong password for the `postgres` superuser.
- Accept the default port `5432`.
- Optionally install the **Stack Builder** — not required for SentinelTrack.

### Step 2: Add PostgreSQL bin to PATH

Add `C:\Program Files\PostgreSQL\18\bin` to your Windows `PATH` environment variable so that `psql`, `pg_dump`, etc. are accessible from PowerShell.

### Step 3: Create the Database User and Database

Open a PowerShell terminal and connect as the `postgres` superuser:

```powershell
psql -U postgres
```

Then run the following SQL commands:

```sql
-- Create a dedicated application user
CREATE USER sentineltrack_user WITH PASSWORD 'your_strong_password_here';

-- Create the application database
CREATE DATABASE sentineltrack_db OWNER sentineltrack_user;

-- Grant all privileges
GRANT ALL PRIVILEGES ON DATABASE sentineltrack_db TO sentineltrack_user;

-- Connect to the new database and enable pgvector (if available)
\c sentineltrack_db

-- Attempt to install pgvector extension (optional but recommended)
CREATE EXTENSION IF NOT EXISTS vector;

-- Exit psql
\q
```

> If pgvector is not installed, SentinelTrack will automatically fall back to storing embeddings as `BYTEA`. Performance of similarity search will be lower without pgvector, but the system remains fully functional.

### Step 4: Install pgvector (Optional but Recommended)

Prebuilt Windows binaries for pgvector are available at:
[https://github.com/pgvector/pgvector/releases](https://github.com/pgvector/pgvector/releases)

Follow the installation instructions specific to your PostgreSQL version. After installation, run `CREATE EXTENSION vector;` in the `sentineltrack_db` database.

---

## 9. Environment Configuration

The `.env` file controls all sensitive configuration. Do **not** commit this file to version control.

```dotenv
# ── Database ──────────────────────────────────────────────────────────────
# Full SQLAlchemy connection URL for PostgreSQL
DATABASE_URL=postgresql+psycopg2://sentineltrack_user:your_password@localhost:5432/sentineltrack_db

# ── Camera / Video Source ─────────────────────────────────────────────────
# Path to a local video file, integer webcam index (0, 1, ...), or RTSP URL
VIDEO_SOURCE=0

# Camera / source identifier used when logging events
SOURCE_ID=cam_001

# ── Evidence Storage ──────────────────────────────────────────────────────
# Directory where face crop snapshots will be saved
EVIDENCE_DIR=./evidence

# ── Logging ───────────────────────────────────────────────────────────────
# Log file path (structured text)
LOG_FILE=./events.log

# Log level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO

# ── Model Paths ───────────────────────────────────────────────────────────
# Path to the YOLOv8 face detection model file
YOLO_MODEL_PATH=./models/yolov8n-face.pt

# InsightFace model pack name (buffalo_sc recommended for CPU)
INSIGHTFACE_MODEL_PACK=buffalo_sc

# ── Pipeline Thresholds ───────────────────────────────────────────────────
# YOLO detection confidence threshold (0.0–1.0)
DETECTION_CONFIDENCE=0.45

# Cosine similarity threshold for identity matching (0.0–1.0)
# Faces with similarity >= this value are matched to an existing identity
SIMILARITY_THRESHOLD=0.50

# Seconds of absence before a PRESENT face is considered exited
EXIT_TIMEOUT_SECONDS=30

# ── Feature Flags ─────────────────────────────────────────────────────────
# Whether to save evidence images on registration events
SAVE_EVIDENCE_IMAGES=true

# Maximum frames per second to process (0 = unlimited)
MAX_FPS=0
```

---

## 10. Model Installation

### YOLOv8 Face Detection Model

The `yolov8n-face.pt` model is a custom-trained YOLOv8 nano variant optimized for face detection. Pre-download it by running:

```powershell
python -c "from ultralytics import YOLO; YOLO('yolov8n-face.pt')"
```

Alternatively, download directly from:
[https://github.com/akanametov/yolov8-face](https://github.com/akanametov/yolov8-face)

Place the downloaded `.pt` file at the path specified by `YOLO_MODEL_PATH` in your `.env` (default: `./models/yolov8n-face.pt`).

### InsightFace ArcFace Model (buffalo_sc)

The InsightFace `buffalo_sc` model pack is downloaded automatically on first execution and cached in:

```
C:\Users\<your_username>\.insightface\models\buffalo_sc\
```

To pre-download it manually:

```python
import insightface
app = insightface.app.FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("InsightFace buffalo_sc downloaded and ready.")
```

> `buffalo_sc` is the smallest available pack and is suitable for CPU inference. It contains a detection model and a recognition (ArcFace) model. The full `buffalo_l` pack is larger and more accurate but slower on CPU.

---

## 11. Database Migration

### Initialize the Schema

Run the provided initialization script to create all tables and indexes:

```powershell
python scripts/init_database.py
```

This script:
1. Verifies the database connection.
2. Attempts to enable the `vector` extension (skips gracefully if unavailable).
3. Creates the `faces`, `events`, and `visit_sessions` tables.
4. Creates all indexes (including a vector index if pgvector is available).

### Apply Alembic Migrations

After the initial setup, use Alembic for all future schema changes:

```powershell
# Apply all pending migrations
alembic upgrade head

# Check current migration state
alembic current

# See migration history
alembic history

# Roll back one migration
alembic downgrade -1
```

To generate a new migration after modifying SQLAlchemy models:

```powershell
alembic revision --autogenerate -m "describe your change here"
```

---

## 12. Running the Application

### Option A: Process Sample Video File (Official Katomaran Video)

```powershell
# Interactive GUI window with HUD, track IDs, and event banners
python -m src.main --video data/katomaran_sample.mp4 --gui

# Headless mode (processes video, updates PostgreSQL, and exports demo to logs/output_demo.mp4)
python -m src.main --video data/katomaran_sample.mp4 --headless

# Quick inspect visitor summary from PostgreSQL
python -m src.main --count
```

```powershell
# Or use the dedicated sample runner script
python scripts/run_sample.py --source data/katomaran_sample.mp4
```

### Option B: Live Webcam

Set `VIDEO_SOURCE=0` (or the appropriate camera index) in your `.env`, then:

```powershell
python -m src.main --gui
```

### Option C: RTSP Stream (Hackathon Interview Mode)

```powershell
python scripts/run_rtsp.py --url rtsp://username:password@192.168.1.100:554/stream1
```

Or pass `--rtsp` directly to the main entry point:

```powershell
python -m src.main --rtsp rtsp://192.168.1.100:554/stream1 --gui
```

### Keyboard Controls (when display window is open)

| Key | Action | Description |
|---|---|---|
| `q` | **Quit** | Closes OpenCV window, releases camera, flushes database and exits cleanly. |
| `p` | **Pause / Resume** | Freezes video playback on current frame to inspect bounding boxes and stats. |
| `s` | **Save Screenshot** | Saves the annotated frame with visitor HUD and bounding boxes to `logs/screenshots/`. |

---

## 13. Testing

### Unit Tests (no database required)

```powershell
pytest tests/unit/ -v
```

Unit tests cover: frame preprocessing, bounding box IoU, cosine similarity, embedding normalization, FSM state transitions, config loading, and log formatting.

### Integration Tests (requires PostgreSQL)

```powershell
pytest tests/integration/ -v
```

Integration tests spin up against the configured test database and verify end-to-end flows: face registration, duplicate suppression, session creation, event emission, and BYTEA/pgvector storage paths.

> Ensure `DATABASE_URL` in `.env` points to a test database before running integration tests, or set the `TEST_DATABASE_URL` environment variable to override.

### Full Test Suite with Coverage Report

```powershell
pytest --cov=src tests/
```

Generate an HTML coverage report:

```powershell
pytest --cov=src --cov-report=html tests/
start htmlcov\index.html
```

### Run a Specific Test File

```powershell
pytest tests/unit/test_tracker.py -v
pytest tests/unit/test_identity_service.py -v -k "test_cosine"
```

---

## 14. Sample Database Queries

Connect to the database with `psql -U sentineltrack_user -d sentineltrack_db` and run these queries:

```sql
-- ── Unique Visitors ────────────────────────────────────────────────────

-- Total confirmed unique identities in the system
SELECT COUNT(*) AS total_unique_visitors
FROM faces
WHERE registration_status = 'confirmed';

-- ── Today's Entry Events ───────────────────────────────────────────────

-- All entry events for today with face ID and evidence path
SELECT
    f.face_id,
    f.first_seen_at,
    e.event_timestamp,
    e.evidence_image_path,
    e.source_id
FROM events e
JOIN faces f ON e.face_id = f.face_id
WHERE e.event_type = 'entry'
  AND e.event_timestamp::date = CURRENT_DATE
ORDER BY e.event_timestamp;

-- ── Session Summary ────────────────────────────────────────────────────

-- Count of active vs. completed sessions
SELECT
    COUNT(*) FILTER (WHERE status = 'exited')  AS completed_sessions,
    COUNT(*) FILTER (WHERE status = 'active')  AS active_sessions,
    COUNT(*)                                    AS total_sessions
FROM visit_sessions;

-- ── Dwell Time Analysis ────────────────────────────────────────────────

-- Average, min, and max dwell time for completed sessions today
SELECT
    AVG(duration_seconds)::INT  AS avg_dwell_seconds,
    MIN(duration_seconds)       AS min_dwell_seconds,
    MAX(duration_seconds)       AS max_dwell_seconds
FROM visit_sessions
WHERE status = 'exited'
  AND entry_time::date = CURRENT_DATE;

-- ── Hourly Traffic ─────────────────────────────────────────────────────

-- Entry counts grouped by hour for today
SELECT
    DATE_TRUNC('hour', event_timestamp) AS hour_bucket,
    COUNT(*)                            AS entries
FROM events
WHERE event_type = 'entry'
  AND event_timestamp::date = CURRENT_DATE
GROUP BY 1
ORDER BY 1;

-- ── Per-Identity Visit History ─────────────────────────────────────────

-- All sessions for a specific face (replace UUID)
SELECT
    session_id,
    entry_time,
    exit_time,
    duration_seconds,
    status
FROM visit_sessions
WHERE face_id = 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'
ORDER BY entry_time DESC;

-- ── Recently Registered Faces ──────────────────────────────────────────

-- Last 10 newly registered identities
SELECT face_id, first_seen_at, registration_status, source_id
FROM faces
ORDER BY first_seen_at DESC
LIMIT 10;
```

---

## 15. Example Log Output

The `events.log` file uses a structured pipe-delimited format for easy parsing and monitoring:

```
2026-10-17T10:30:01 | INFO     | sentineltrack.identity | [REGISTRATION] face_id=abc12345-dead-beef-cafe-001122334455 frame=1 source=cam_001 similarity=None (new identity)
2026-10-17T10:30:01 | INFO     | sentineltrack.evidence | [EVIDENCE_SAVED] face_id=abc12345-dead-beef-cafe-001122334455 path=./evidence/abc12345_frame001.jpg
2026-10-17T10:30:02 | INFO     | sentineltrack.session  | [ENTRY] face_id=abc12345-dead-beef-cafe-001122334455 session_id=sess_f9e0d1c2 timestamp=2026-10-17T10:30:02 source=cam_001
2026-10-17T10:30:15 | INFO     | sentineltrack.identity | [MATCH] face_id=abc12345-dead-beef-cafe-001122334455 frame=210 similarity=0.8734 source=cam_001
2026-10-17T10:30:45 | INFO     | sentineltrack.session  | [EXIT] face_id=abc12345-dead-beef-cafe-001122334455 session_id=sess_f9e0d1c2 timestamp=2026-10-17T10:30:45 duration_seconds=43 source=cam_001
2026-10-17T10:31:10 | INFO     | sentineltrack.identity | [REGISTRATION] face_id=bb998877-0011-2233-4455-667788990011 frame=512 source=cam_001 similarity=None (new identity)
2026-10-17T10:31:10 | INFO     | sentineltrack.session  | [ENTRY] face_id=bb998877-0011-2233-4455-667788990011 session_id=sess_a1b2c3d4 timestamp=2026-10-17T10:31:10 source=cam_001
2026-10-17T10:35:00 | WARNING  | sentineltrack.tracker  | [TRACK_LOST] track_id=7 face_id=bb998877-0011-2233-4455-667788990011 frames_missing=5
2026-10-17T10:35:30 | INFO     | sentineltrack.session  | [EXIT] face_id=bb998877-0011-2233-4455-667788990011 session_id=sess_a1b2c3d4 timestamp=2026-10-17T10:35:30 duration_seconds=260 source=cam_001
2026-10-17T10:40:00 | INFO     | sentineltrack.pipeline | [STATS] frames_processed=3600 faces_detected=12803 identities_registered=2 sessions_completed=2 fps_avg=4.3
```

---

## 16. Unique Visitor vs Session Count

It is important to understand the distinction between these two metrics:

### Unique Visitors

A **unique visitor** is a distinct individual whose face embedding has been registered in the `faces` table. This count never decreases. A person who visits every day for a week is still counted as **one unique visitor**.

```sql
SELECT COUNT(*) FROM faces WHERE registration_status = 'confirmed';
```

### Visit Sessions

A **visit session** is a continuous presence window for a single individual. Each time a known face disappears from the frame for longer than `EXIT_TIMEOUT_SECONDS` and then reappears, a **new session** is created. A single unique visitor who enters, leaves, and re-enters creates **two sessions**.

```sql
SELECT COUNT(*) FROM visit_sessions;
```

### Why Both Matter

| Metric | Answers the question |
|---|---|
| Unique Visitors | "How many distinct people have ever visited?" |
| Session Count | "How many times in total did people enter this space?" |
| Active Sessions | "How many people are currently inside right now?" |
| Avg Dwell Time | "How long do people typically stay?" |

Both figures together give a complete picture of foot traffic patterns.

---

## 17. Performance Estimates

> **⚠️ IMPORTANT: These are theoretical estimates, NOT measured benchmarks.**
> Actual performance will vary significantly based on video resolution, face count per frame, CPU generation, background load, and disk I/O.

| Configuration | Estimated FPS | Notes |
|---|---|---|
| Intel Core i5 (no GPU) | ~3–6 FPS | Full pipeline, 1080p input, 1–3 faces |
| Intel Core i7 / AMD Ryzen 7 (no GPU) | ~6–10 FPS | Full pipeline, 1080p input |
| NVIDIA RTX 3060 / 3080 (GPU) | ~20–30 FPS | With CUDA enabled (not default config) |
| Raspberry Pi 4 (ARM) | ~0.5–1 FPS | Not officially supported |

### Latency Estimates per Frame (CPU, Intel Core i5)

| Stage | Estimated Latency |
|---|---|
| YOLOv8n face detection | ~150–300 ms |
| ArcFace embedding extraction | ~80–150 ms |
| Cosine similarity search (1000 identities) | ~5–15 ms |
| PostgreSQL write (event insert) | ~2–10 ms |
| Total pipeline latency (approx.) | ~250–500 ms |

### Recommended Hardware (CPU Deployment)

- **CPU:** 4-core / 8-thread, 2.5 GHz+ (Intel Core i5 10th gen or newer)
- **RAM:** 8 GB minimum, 16 GB recommended
- **Storage:** SSD preferred for evidence image writes; 1 GB/day estimate at 1 image/entry
- **Network (RTSP):** 10 Mbps+ for 1080p streams

### Enabling GPU Acceleration

If a CUDA-capable GPU is available, install CUDA-enabled packages:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu
```

Then set the ONNX execution provider in `config.json`:
```json
"onnx_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]
```

---

## 18. Assumptions

The following assumptions were made during the design and implementation of SentinelTrack:

1. **Single Camera per Process:** Each running instance of `src.main` processes one video source. Multi-camera setups require running multiple processes (one per camera), each with a distinct `SOURCE_ID`.
2. **Frontal / Near-Frontal Faces:** YOLOv8 and ArcFace perform best with faces that are reasonably frontal. Extreme profile angles may result in missed detections or poor embeddings.
3. **Controlled Lighting:** Severe under- or over-exposure reduces detection accuracy. Adequate lighting is assumed.
4. **Non-Adversarial Environment:** The system is not designed to resist spoofing attacks (e.g., photo-in-photo). It is an operational visitor counter, not a security system.
5. **PostgreSQL Always Available:** The pipeline assumes a live database connection. Network interruptions will cause event writes to fail (logged as errors; frames continue to be processed).
6. **Similarity Threshold Is Tuned:** The default threshold of `0.50` is a starting point. Operators should tune this based on their specific camera quality and population diversity.
7. **Exit Determined by Absence:** There is no physical door sensor integration. Exit is inferred when a tracked face is absent from the frame for `EXIT_TIMEOUT_SECONDS`.
8. **One Identity per Face:** The system assumes each physical face maps to exactly one identity. Identical twins may be assigned the same identity if their embeddings are sufficiently similar.
9. **Python 3.10 Environment:** All package dependencies are pinned and tested against Python 3.10.11 on Windows 11.
10. **Disk Space for Evidence:** Evidence image storage is assumed to be on a local disk with adequate free space. No automatic pruning is implemented by default.

---

## 19. Known Limitations

- **No GPU in test environment:** All development and testing was performed on CPU. GPU code paths exist in configuration but have not been hardware-tested in this submission.
- **RTSP tested with mocked sources only:** RTSP integration was validated against a local RTSP server simulation (`mediamtx`). Real IP camera compatibility may require stream format adjustments.
- **Performance varies with face size and video resolution:** Small faces (< 40×40 pixels in the original frame) are frequently missed by the detector. Higher-resolution input helps.
- **No de-duplication across camera instances:** If the same person appears on two different cameras (two running instances), they will be registered as separate identities in the database. Cross-camera re-identification is out of scope.
- **No incremental embedding update:** The stored embedding is not continuously updated as new frames arrive; only the initial registration embedding is kept (apart from a periodic refresh on re-seen faces).
- **Alembic auto-migration requires manual review:** `alembic revision --autogenerate` may not detect all schema changes correctly. Always review generated migration files before applying.
- **pgvector ANN index not yet tuned:** The `ivfflat` index is created with default parameters. For databases with >10,000 identities, index tuning (`lists` parameter) is recommended.
- **No web dashboard:** There is no built-in HTTP API or web UI. Monitoring is done via database queries and log file inspection.

---

## 20. Troubleshooting

### `ModuleNotFoundError: No module named 'insightface'`

Ensure you are in the activated virtual environment and re-run:
```powershell
pip install insightface onnxruntime
```
If it still fails, check that Visual C++ Build Tools are installed.

---

### `psycopg2.OperationalError: could not connect to server`

1. Verify PostgreSQL is running: open **Services** (`services.msc`) and check that `postgresql-x64-18` is started.
2. Verify the connection string in `.env` is correct (host, port, user, password, database name).
3. Check that the firewall is not blocking port 5432.

---

### `YOLO model file not found`

Ensure the model file exists at the path set in `YOLO_MODEL_PATH`. Re-run the pre-download command:
```powershell
python -c "from ultralytics import YOLO; YOLO('yolov8n-face.pt')"
```

---

### Very Low FPS (< 1 FPS)

- Reduce input resolution by adding `--scale 0.5` flag to downsample frames before detection.
- Increase the frame-skip interval in `config.json` (`"process_every_n_frames": 3`).
- Close other CPU-intensive applications.
- Consider using the `buffalo_s` or `buffalo_sc` InsightFace model instead of `buffalo_l`.

---

### `alembic.util.exc.CommandError: Can't locate revision`

Your migration history may be out of sync. Run:
```powershell
alembic stamp head
alembic upgrade head
```

---

### Evidence images not being saved

1. Check that `SAVE_EVIDENCE_IMAGES=true` in `.env`.
2. Verify the `EVIDENCE_DIR` path exists and the process has write permission.
3. Check `events.log` for `[EVIDENCE_SAVED]` entries to confirm the feature is active.

---

### Face registered multiple times (duplicate identities)

- Your `SIMILARITY_THRESHOLD` may be too high. Lower it from `0.50` to `0.40`–`0.45`.
- Ensure adequate lighting so that embedding quality is consistent between frames.
- If using a wide-angle camera, faces at the edge of frame may have distorted embeddings.

---

### `CREATE EXTENSION vector` fails

pgvector is not installed for your PostgreSQL 18 instance. Download the matching binary from [https://github.com/pgvector/pgvector/releases](https://github.com/pgvector/pgvector/releases) and follow installation instructions. The system will continue to work without it using the BYTEA fallback.

---

## 21. Privacy and Retention Policy

SentinelTrack processes biometric data (face images and embeddings), which is subject to privacy regulations in many jurisdictions (GDPR, CCPA, PDPA, etc.). Operators deploying this system are responsible for:

1. **Consent and Notice:** Displaying clear signage informing individuals that video surveillance and facial recognition are in use at the monitored location.
2. **Data Minimization:** Evidence images and embeddings should only be retained for as long as operationally necessary. The default configuration does not auto-delete records.
3. **Access Control:** The PostgreSQL database, evidence image directory, and `events.log` file must be access-controlled and not exposed publicly.
4. **Right to Erasure:** To delete a specific individual's data, remove their row from the `faces` table (cascades to `events` and `visit_sessions` via foreign key), and delete their evidence images from disk.
5. **Data Encryption:** For production deployments, enable PostgreSQL SSL connections and encrypt the evidence image storage at rest.
6. **Retention Schedule:** Operators should implement a scheduled job to purge records older than their defined retention period (e.g., 30 days for operational data, 7 days for evidence images).

> **This system is provided as a hackathon prototype. It is the deploying operator's sole responsibility to ensure legal compliance before using it in a live environment.**

---

## 22. Demonstration Video

`[VIDEO PLACEHOLDER - Upload to YouTube/Google Drive and add link here]`

The demonstration video should show:
1. System startup and model loading.
2. A person walking into frame → registration event logged.
3. Same person re-appearing → match event, no new registration.
4. Second distinct person → new registration.
5. Database query showing unique visitor count and session records.
6. `events.log` tail showing structured log entries in real time.

---

## 23. GitHub Submission Checklist

Before submitting to the Katomaran Hackathon, verify all of the following:

- [ ] Repository is **public** on GitHub
- [ ] `README.md` is present at the repository root with all sections complete
- [ ] `requirements.txt` is present and up to date (`pip freeze > requirements.txt`)
- [ ] `.env.example` is present with all variables listed (no real credentials)
- [ ] `config.example.json` is present with all default values
- [ ] `src/` directory contains all source modules
- [ ] `scripts/` directory contains `init_database.py`, `run_sample.py`, `run_rtsp.py`
- [ ] `tests/` directory contains `unit/` and `integration/` subdirectories with test files
- [ ] `alembic/` directory with `env.py`, `alembic.ini`, and at least one migration version
- [ ] `.gitignore` excludes `.env`, `.venv/`, `evidence/`, `events.log`, `*.pt`, `__pycache__/`
- [ ] All source files have module-level docstrings
- [ ] `LICENSE` file is present (MIT recommended)
- [ ] Demonstration video link is filled in (section 22)
- [ ] No hardcoded credentials or secrets anywhere in the committed code
- [ ] `pytest tests/unit/ -v` passes without errors
- [ ] Application starts without errors on a fresh environment following these README steps
- [ ] Database migration (`alembic upgrade head`) completes without errors
This project is a part of a hackathon run by https://katomaran.com
