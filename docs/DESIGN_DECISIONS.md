# SentinelTrack — Architectural & Design Decisions

This document details the major technical decisions, trade-offs, and design rationales made during the development of SentinelTrack for the Katomaran Hackathon.

---

## 1. Relational Database: PostgreSQL 18 Over SQLite / Document Stores

### Decision
PostgreSQL was selected as the sole, required database engine. SQLite and MongoDB were explicitly avoided for production and pipeline state tracking.

### Rationale
- **Concurrency & Transaction Isolation**: Video processing pipelines frequently issue concurrent identity updates, session state changes, and event audit insertions. SQLite's database-level write lock causes contention and `database is locked` exceptions under multithreaded or multi-camera loads.
- **Advanced Indexing & Integrity Constraints**: PostgreSQL supports partial unique indexes (`CREATE UNIQUE INDEX ... WHERE event_type = 'entry'`), which mathematically enforce that no logical visit session can have more than one entry event or exit event. SQLite partial indexes have limited constraint semantics and lack native date-time timezone operations.
- **Enterprise Vector Search Readiness**: PostgreSQL natively supports the `pgvector` extension for Hierarchical Navigable Small World (HNSW) and Inverted File (IVF) vector index searches.
- **Source of Truth**: Keeping face metadata, session lifecycles, and event history in a single relational schema guarantees ACID transactional consistency.

---

## 2. Embedding Storage: BYTEA with Python Cosine Similarity (pgvector Fallback)

### Decision
Face embeddings (512-dimensional `float32` arrays) are stored in PostgreSQL as `BYTEA` (binary blobs) with an in-memory Python cosine-similarity cache. `pgvector` is supported when the extension is installed in the database cluster.

### Rationale
- **Zero-Dependency Portability**: In many Windows environments, compiling or installing the C-based `pgvector` PostgreSQL extension requires specific Visual Studio build tools. Storing embeddings as raw IEEE 754 serialized float32 bytes (`ndarray.tobytes()`) guarantees 100% platform portability.
- **In-Memory Cache Latency**: For visitor tracking, queries are executed every few frames. Querying PostgreSQL over a network socket for every detected face adds 2-10ms of network latency. Loading registered embeddings into memory on startup and keeping the cache updated allows similarity search in $< 0.1\text{ ms}$ using vectorized NumPy dot products:
  $$\text{sim}(\mathbf{q}, \mathbf{d}) = \mathbf{q} \cdot \mathbf{d} \quad (\text{since } \|\mathbf{q}\| = \|\mathbf{d}\| = 1)$$
- **Exact Vector Reconstruction**: Storing float32 bytes avoids floating-point precision loss associated with text-based representation or lossy quantization.

---

## 3. Face Detection: YOLOv8 Face (WIDER FACE) vs Generic Object Detectors

### Decision
We use a face-specialized YOLO model (`yolov8n-face.pt` trained on WIDER FACE) rather than a standard COCO person detector or heavy models like MTCNN/RetinaFace-ResNet50.

### Rationale
- **Domain Specialization**: Standard COCO YOLO detectors detect generic "person" bounding boxes (whole bodies). Cropping whole bodies causes ArcFace recognition models to fail or degrade drastically. A dedicated face detector outputs tight bounding boxes around the forehead, cheeks, and chin.
- **Inference Speed on CPU**: `yolov8n-face` is a nano model (~6MB) optimized for high throughput. On standard x86 CPU hardware, it executes in ~80-180ms per frame, whereas heavy models (e.g. RetinaFace ResNet50) take 500-1200ms per frame.
- **Configurable Frame Skipping**: To further conserve CPU cycles, `detection_frame_skip` allows YOLO to run every $N$ frames (default 2), while the lightweight tracker interpolates positions on skipped frames.

---

## 4. Multi-Object Tracking: IoU Tracker vs DeepSORT / ByteTrack C++ Extensions

### Decision
A clean, pure-Python IoU-based Kalman/velocity tracker was implemented, adhering to ByteTrack track-lifecycle principles (Tentative $\rightarrow$ Active $\rightarrow$ Lost $\rightarrow$ Removed).

### Rationale
- **Windows Binary Compatibility**: C++ tracker libraries (such as `lap`, `cython-bbox`, or `motpy`) frequently fail to build on Windows systems lacking MSVC C++ 14.0 build tools.
- **Domain Fit for Fixed Cameras**: For doorway, hallway, and entrance cameras, camera motion is zero and visitor speed is moderate. High-confidence face detections have high spatial overlap between successive frames (IoU > 0.3).
- **Separation of Track ID vs Face ID**:
  - `track_id` is an integer assigned by the tracker to maintain temporal continuity across frames.
  - `face_id` is a stable UUID assigned by `IdentityService` after ArcFace embedding extraction.
  - This separation ensures tracker restarts or brief re-acquisitions never overwrite persistent identity records.

---

## 5. Visit Session State Machine & Hysteresis

### Decision
Session management is structured as a finite state machine:
$$\text{NOT\_PRESENT} \longrightarrow \text{ACTIVE} \longleftrightarrow \text{EXIT\_PENDING} \longrightarrow \text{EXITED}$$

### Rationale
- **Elimination of Flapping**: In real-world surveillance, people frequently turn their heads, look down at phones, or are momentarily occluded by other people. If absence in a single frame triggered an exit, one continuous visit would generate dozens of false entries and exits.
- **Configurable Confirmation Windows**:
  - `exit_confirmation_frames`: Number of frames a track must be missing before state transitions to `EXIT_PENDING` (default 20 frames).
  - If the person reappears during this grace period, the session reverts to `ACTIVE` without a state change event.
  - If the absence persists, the session is marked `EXITED`, and dwell time is computed.
- **Returning Visitor Logic**: When a known identity returns after a confirmed exit, a **new visit session** is created, but the **unique identity count remains constant**.

---

## 6. Atomic Evidence Storage with Filesystem-Database Consistency

### Decision
Evidence images (cropped face snapshots) are written using a two-stage atomic write pattern:
1. Write to a temporary file in the target directory (`.tmp`).
2. Atomically rename/replace (`os.replace`) to the final structured path:
   `logs/entries/YYYY-MM-DD/entry_<face8>_<sess8>_<frame>_<ts>.jpg`
3. Verify file existence and non-zero size before persisting the file path in PostgreSQL.

### Rationale
- **Protection Against Partial Writes**: Sudden process termination or disk exhaustion will never leave corrupted, zero-byte image files at the final path.
- **Audit Traceability**: File paths encode event type, face UUID prefix, session UUID prefix, frame number, and microsecond timestamp, guaranteeing zero filename collisions.
- **Reconciliation Tooling**: The included `scripts/verify_integrity.py` scans database records against filesystem paths, flagging any discrepancies for operator resolution.

---

## 7. Registration De-duplication & Ambiguity Margin

### Decision
To prevent a single visitor from being registered as multiple distinct identities due to lighting or angle variations across consecutive frames:
1. **Track-Level Cooldown**: A 2-second cooldown window prevents the same visual track from attempting multiple registrations.
2. **Ambiguity Margin**: If a query embedding matches two registered candidates within `ambiguity_margin` ($\Delta < 0.08$ similarity), the match is flagged as ambiguous and deferred rather than making a false positive association.
3. **Conservative Quality Gate**: Detections with bounding box dimensions smaller than 30x30 pixels or detector confidence below 0.45 are rejected from registration.
