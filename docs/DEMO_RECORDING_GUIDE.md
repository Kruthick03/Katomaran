# SentinelTrack — Demonstration Recording & Submission Guide

This guide provides step-by-step instructions for recording a 3–5 minute demonstration video for the Katomaran Hackathon review team.

---

## 1. Demo Preparation Checklist

Before starting your screen recording:
- [ ] Ensure PostgreSQL is running (`localhost:5432`).
- [ ] Verify database credentials in [`.env`](file:///C:/Users/kruth/OneDrive/Documents/FACETRACKER_KATOMARAN/sentineltrack/.env).
- [ ] Open two windows side-by-side on your screen:
  1. **Left Half:** Command Prompt (`cmd.exe`) in `C:\Users\kruth\OneDrive\Documents\FACETRACKER_KATOMARAN\sentineltrack`.
  2. **Right Half:** pgAdmin 4 showing the `sentineltrack` database tables (`faces`, `visit_sessions`, `events`).
- [ ] Tool to record: Windows Game Bar (`Win + Alt + R`), OBS Studio, or Loom.

---

## 2. Recommended 4-Minute Recording Script

### Scene 1: Introduction (0:00 – 0:45)
- **What to say:**
  > "Hello, my name is [Your Name], and this is SentinelTrack: an intelligent face tracker with auto-registration and visitor counting, built for the Katomaran Hackathon. SentinelTrack processes video streams, detects and tracks human faces in real time, registers new identities into PostgreSQL using ArcFace embeddings, and logs exactly-once entry and exit events with cropped evidence images."
- **What to show:** Briefly show the repository structure in VS Code or File Explorer (`src/`, `config.json`, `README.md`).

### Scene 2: Running Automated Tests (0:45 – 1:30)
- **Command to run:**
  ```cmd
  pytest tests/ -v
  ```
- **What to say:**
  > "First, we run our automated test suite with pytest. All 108 tests pass in under 2 seconds, covering unit tests for YOLO bounding box validation, ArcFace cosine similarity, IoU tracking, PostgreSQL constraints, and end-to-end acceptance sequences."

### Scene 3: Live Video Pipeline with OpenCV GUI (1:30 – 2:45)
- **Command to run:**
  ```cmd
  python -m src.main --video data/sample.mp4
  ```
- **What to say & show:**
  > "Now, we run the live pipeline on our sample video. A desktop OpenCV window appears showing the annotated video stream in real-time. Notice the bounding boxes color-coded with Track IDs and persistent Face IDs. When a person first enters, SentinelTrack automatically computes a 512-dimensional ArcFace embedding and registers them in PostgreSQL. It saves a timestamped cropped evidence image to `logs/entries/`. When they leave the frame, after the absence confirmation threshold, an exit event is logged."

### Scene 4: Database Inspection & Visitor Counts (2:45 – 3:30)
- **Command to run in CMD:**
  ```cmd
  python -m src.main --count
  python scripts/verify_integrity.py
  ```
- **What to show in pgAdmin 4:**
  - In pgAdmin 4, show the query result of `SELECT * FROM visit_sessions;` and `SELECT * FROM faces;`.
- **What to say:**
  > "Notice that when a returning visitor reappears, SentinelTrack recognizes them using cosine similarity against the database embeddings. Their visit session count increments, but the unique registered visitor count remains strictly unchanged. Our integrity checker verifies zero orphaned records and zero missing evidence files."

### Scene 5: Conclusion & Hackathon Acknowledgment (3:30 – 4:00)
- **What to say:**
  > "SentinelTrack is designed for high reliability with database transactions, atomic file writes, and restart recovery. Thank you for watching!"
- **What to show:** Show the final line of `README.md`:
  `This project is a part of a hackathon run by https://katomaran.com`

---

## 3. Upload & Submission Instructions

1. **Upload Video:**
   - Upload your recorded `.mp4` file to YouTube (set visibility to **Unlisted**) or Google Drive (set access to **Anyone with the link can view**).
2. **Update README.md:**
   - Open [`README.md`](file:///C:/Users/kruth/OneDrive/Documents/FACETRACKER_KATOMARAN/sentineltrack/README.md) and replace the placeholder `[VIDEO PLACEHOLDER]` with your link.
3. **Submit:**
   - Commit and push your code to your GitHub repository.
   - Submit your repository link before **Monday, October 19, 2026, 12:00 PM**.
