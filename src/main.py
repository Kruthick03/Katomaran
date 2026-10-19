"""
SentinelTrack --- src/main.py

Application entry point and top-level pipeline orchestrator.

Pipeline flow (per frame):
  1. Read frame from VideoSource.
  2. Run YOLOFaceDetector (every N frames, respecting frame_skip).
  3. Update FaceTracker with new detections.
  4. For each active track (age >= min_track_age):
     a. Generate embedding via FaceRecognizer.
     b. Identify/register via IdentityService.
     c. Create/update visit session via VisitSessionManager.
     d. Save evidence image if entry event was just fired.
  5. Check for exits: tracks that disappeared trigger exit confirmation.
  6. Annotate and display frame (if not headless).
  7. Collect metrics. Report periodically.
  8. On shutdown: flush pending events, close DB, release video.

Error handling:
  - Frame read failures: log and continue.
  - DB failures: log, record in metrics, continue (never crash pipeline).
  - Shutdown signals (Ctrl+C, SIGTERM): graceful shutdown.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Ensure project root is on sys.path when running as script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.config import load_config, set_config, AppConfig
from src.event_logger import setup_logging, get_logger, log_startup, log_shutdown
from src.database import init_db, get_session, check_connection, close_db
from src.video_input import create_video_source
from src.face_detector import create_detector, FaceDetection
from src.face_recognizer import create_recognizer
from src.tracker import create_tracker, TrackState
from src.identity_service import IdentityService
from src.visit_session import VisitSessionManager
from src.evidence_storage import EvidenceStorage
from src.event_service import EventService
from src.metrics import MetricsCollector
from src.recovery import RecoveryChecker
from src.repositories import (
    CameraRepository,
    FaceRepository,
    SessionRepository,
    EventRepository,
)
from src.visualizer import SentinelVisualizer

logger = get_logger("main")

# --------- Shutdown flag ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
_shutdown_requested = False


def _handle_signal(signum, frame):
    """Graceful shutdown on SIGINT or SIGTERM."""
    global _shutdown_requested
    logger.info(f"Shutdown signal received ({signum}). Finishing current frame...")
    _shutdown_requested = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# --------- Annotation ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def annotate_frame(
    frame: np.ndarray,
    tracks,
    unique_count: int = 0,
    session_count: int = 0,
    fps: float = 0.0,
) -> np.ndarray:
    """
    Draw bounding boxes, face IDs and stats overlay on the frame.
    Returns the annotated frame (does not modify in place).
    """
    vis = SentinelVisualizer()
    return vis.render(
        frame=frame,
        tracks=tracks,
        unique_count=unique_count,
        session_count=session_count,
        active_sessions=len([t for t in tracks if getattr(t, 'state', None) != TrackState.REMOVED]),
        total_entries=0,
        total_exits=0,
        fps=fps,
        frame_number=0,
    )


# --------- Main pipeline ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def run(
    config_path: str = "config.json",
    video: Optional[str] = None,
    rtsp: Optional[str] = None,
    headless: Optional[bool] = None,
    output_video: Optional[str] = None,
    max_frames: Optional[int] = None,
) -> dict:
    """
    Main pipeline. Reads config, initializes all components, runs
    the frame loop, and returns a run summary dictionary.

    Args:
        config_path: Path to config.json.
        video: Optional path to override video file input.
        rtsp: Optional RTSP URI to override video stream input.
        headless: Optional boolean to override headless display mode.

    Returns:
        Run summary dict (frames, identities, sessions, errors, duration_seconds).
    """
    # ------ Load configuration ------------------------------------------------------------------------------------------------------------------------------------------------------------
    config = load_config(config_path)

    if video:
        config.source.type = "file"
        config.source.video_path = video
    if rtsp:
        config.source.type = "rtsp"
        config.source.rtsp_uri = rtsp
    if headless is not None:
        config.headless = headless

    set_config(config)

    # ------ Setup logging ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    setup_logging(
        level=config.logging.level,
        log_file=config.logging.log_file,
        max_bytes=config.logging.max_bytes,
        backup_count=config.logging.backup_count,
        console_output=config.logging.console_output,
    )
    log_startup(config_path, config.source.type)

    # ------ Database ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    init_db(config)
    if not check_connection():
        logger.error("Cannot connect to PostgreSQL. Exiting.")
        sys.exit(1)

    # Register camera record
    with get_session() as db:
        CameraRepository.get_or_create(
            db,
            source_id=config.source.source_id,
            source_type=config.source.type,
            source_name=config.source.source_name,
        )

    # ------ Recovery: mark stale sessions from previous run ---------------------------------------------------------------------
    recovery = RecoveryChecker()
    with get_session() as db:
        recovery.recover_incomplete_sessions(db, config.source.source_id)

    # ------ Initialize pipeline components ------------------------------------------------------------------------------------------------------------------------
    video_source = create_video_source(config)
    detector = create_detector(config)
    recognizer = create_recognizer(config)
    tracker = create_tracker(config)
    metrics = MetricsCollector()

    evidence_storage = EvidenceStorage(config.evidence)
    identity_service = IdentityService(config, recognizer)
    session_manager = VisitSessionManager(config.session)
    event_service = EventService(
        config=config,
        recognizer=recognizer,
        identity_service=identity_service,
        session_manager=session_manager,
        evidence_storage=evidence_storage,
        metrics=metrics,
    )

    visualizer = SentinelVisualizer()
    is_paused = False
    debug_mode = False

    # Load registered faces into memory
    with get_session() as db:
        identity_service.reload_from_database(db)
        unique_count = FaceRepository.count_unique_identities(db)
        session_count = SessionRepository.count_sessions(db)

    logger.info(
        f"Startup state: unique_identities={unique_count} sessions={session_count}"
    )

    # ------ Open video source ---------------------------------------------------------------------------------------------------------------------------------------------------------------
    try:
        video_source.open()
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error(f"Cannot open video source: {exc}")
        sys.exit(1)

    # ------ Frame loop ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    start_time = time.time()
    frame_count = 0
    fps_estimate = 0.0
    fps_counter = 0
    fps_timer = time.time()

    logger.info("Pipeline started. Press Ctrl+C to stop.")

    if not config.headless:
        cv2.namedWindow(visualizer.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(visualizer.window_name, 1280, 720)
        cv2.moveWindow(visualizer.window_name, 0, 0)

    # Output video writer — saves a rendered demo video (useful for
    # hackathon submissions and headless runs where cv2.imshow is unavailable)
    output_video_path = Path(output_video) if output_video else Path("logs") / "annotated_demo.mp4"
    output_video_path.parent.mkdir(exist_ok=True)
    video_writer = None
    try:
        # Get source video properties for the writer
        src_fps = getattr(video_source, '_fps', 25.0) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Writer resolution will be set on the first frame
    except Exception:
        src_fps = 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    try:
        while not _shutdown_requested:
            t_frame_start = time.perf_counter()
            success, frame, frame_number = video_source.read_frame()

            if not success or frame is None:
                logger.info("Video source ended or frame unavailable.")
                break

            frame_count += 1
            if max_frames and frame_count > max_frames:
                logger.info(f"Reached max frames limit ({max_frames}). Stopping.")
                break

            # ------ Detection (respects frame_skip internally) ------------------------------------------------------------
            t_det_start = time.perf_counter()
            detections, ran_detection = detector.detect(frame, frame_number)
            det_time = time.perf_counter() - t_det_start
            if ran_detection:
                metrics.record_detection_time(det_time)
                metrics.increment_detections(len(detections))

            # ------ Tracking ------------------------------------------------------------------------------------------------------------------------------------------------------------------
            active_tracks, removed_tracks = tracker.update(
                detections=detections,
                frame_number=frame_number,
                is_detection_frame=ran_detection,
            )

            # ------ Identity and session processing ---------------------------------------------------------------------------------------------
            now = datetime.now(timezone.utc)
            active_face_ids = set()

            with get_session() as db:
                t_rec_start = time.perf_counter()

                for track in active_tracks:
                    if track.state == TrackState.REMOVED:
                        continue
                    # Never run recognition on missed/lost tracks using old stale coordinates
                    if track.missed_frames > 0:
                        continue
                    # On skipped frames, if track already has a confirmed face_id, keep it active
                    # without re-running CPU-heavy ArcFace embedding inference on predicted boxes
                    if not ran_detection and track.face_id:
                        active_face_ids.add(track.face_id)
                        continue
                    try:
                        processed = event_service.process_tracked_face(
                            db_session=db,
                            frame=frame,
                            track=track,
                            frame_number=frame_number,
                            timestamp=now,
                        )
                        if track.face_id:
                            active_face_ids.add(track.face_id)
                    except Exception as exc:
                        logger.error(
                            f"Error processing track {track.track_id}: {exc}",
                            exc_info=True
                        )
                        metrics.increment_errors()

                # Handle exits for disappeared tracks
                event_service.process_session_exits(
                    db_session=db,
                    active_face_ids=active_face_ids,
                    frame_number=frame_number,
                    timestamp=now,
                )

                rec_time = time.perf_counter() - t_rec_start
                metrics.record_recognition_time(rec_time)

                # Refresh counts periodically
                if frame_count % config.metrics.report_interval_frames == 0:
                    unique_count = FaceRepository.count_unique_identities(db)
                    session_count = SessionRepository.count_sessions(db)

            # ------ Metrics ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
            total_time = time.perf_counter() - t_frame_start
            metrics.record_frame_time(total_time)
            metrics.increment_frames()

            # FPS calculation
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps_estimate = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()

            # Periodic metrics report
            if frame_count % config.metrics.report_interval_frames == 0:
                metrics.report_to_log(logger)
                if config.metrics.log_memory:
                    metrics.get_resource_usage()

            # ------ Visualization (always render for video writer) ------------------------------------
            active_sessions_count = session_manager.active_sessions_count
            feed_unique_count = session_manager.feed_unique_visitors_count
            feed_session_count = session_manager.feed_total_sessions_count
            metric_summary = metrics.get_summary()
            total_entries = metric_summary.get("entries", 0)
            total_exits = metric_summary.get("exits", 0)
            recent_events = session_manager.get_recent_events()
            source_label = config.source.source_name or config.source.video_path

            display_frame = visualizer.render(
                frame=frame,
                tracks=active_tracks,
                unique_count=feed_unique_count,
                session_count=feed_session_count,
                active_sessions=active_sessions_count,
                total_entries=total_entries,
                total_exits=total_exits,
                fps=fps_estimate,
                frame_number=frame_count,
                recent_events=recent_events,
                source_label=source_label,
                is_paused=is_paused,
                db_registered_count=unique_count,
                raw_detections=detections,
                debug_mode=debug_mode,
                frame_skip=config.detection.detection_frame_skip,
            )

            # Write every frame to the output demo video
            if video_writer is None:
                h_out, w_out = display_frame.shape[:2]
                video_writer = cv2.VideoWriter(
                    str(output_video_path), fourcc, src_fps, (w_out, h_out)
                )
            video_writer.write(display_frame)

            # ------ Display (if not headless) -----------------------------------------------
            if not config.headless:
                cv2.imshow(visualizer.window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    logger.info("User pressed 'q'. Stopping.")
                    break
                elif key == ord("d"):
                    debug_mode = not debug_mode
                    logger.info(f"Debug mode {'enabled' if debug_mode else 'disabled'}.")
                elif key == ord("p"):
                    is_paused = not is_paused
                    logger.info(f"Playback {'paused' if is_paused else 'resumed'}.")
                    while is_paused:
                        display_frame_paused = visualizer.render(
                            frame=frame,
                            tracks=active_tracks,
                            unique_count=feed_unique_count,
                            session_count=feed_session_count,
                            active_sessions=active_sessions_count,
                            total_entries=total_entries,
                            total_exits=total_exits,
                            fps=fps_estimate,
                            frame_number=frame_count,
                            recent_events=recent_events,
                            source_label=source_label,
                            is_paused=True,
                            db_registered_count=unique_count,
                            raw_detections=detections,
                            debug_mode=debug_mode,
                        )
                        cv2.imshow(visualizer.window_name, display_frame_paused)
                        pause_key = cv2.waitKey(50) & 0xFF
                        if pause_key == ord("p"):
                            is_paused = False
                            logger.info("Playback resumed.")
                            break
                        elif pause_key == ord("d"):
                            debug_mode = not debug_mode
                            logger.info(f"Debug mode {'enabled' if debug_mode else 'disabled'}.")
                        elif pause_key == ord("q"):
                            logger.info("User pressed 'q' while paused. Stopping.")
                            key = ord("q")
                            break
                        elif pause_key == ord("s"):
                            shot_path = visualizer.save_screenshot(display_frame_paused)
                            logger.info(f"Screenshot saved while paused: {shot_path}")
                    if key == ord("q"):
                        break
                elif key == ord("s"):
                    shot_path = visualizer.save_screenshot(display_frame)
                    logger.info(f"Screenshot saved to: {shot_path}")

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received.")
    except Exception as exc:
        logger.error(f"Unhandled exception in main loop: {exc}", exc_info=True)
    finally:
        # ------ Graceful shutdown ---------------------------------------------------------------------------------------------------------------------------------------------------
        logger.info("Shutting down pipeline...")

        # Final recovery check
        with get_session() as db:
            unique_count = FaceRepository.count_unique_identities(db)
            session_count = SessionRepository.count_sessions(db)

        if video_writer is not None:
            video_writer.release()
            logger.info(f"Rendered demo video saved to: {output_video_path.resolve()}")

        video_source.release()
        detector.unload()
        recognizer.unload()

        if not config.headless:
            cv2.destroyAllWindows()

        close_db()

        duration = time.time() - start_time
        log_shutdown(frame_count, unique_count, session_count)

        summary = metrics.get_summary()
        summary.update({
            "unique_identities": unique_count,
            "total_sessions": session_count,
            "duration_seconds": round(duration, 2),
        })

        # Save run summary to JSON
        summary_path = Path("logs") / "last_run_summary.json"
        summary_path.parent.mkdir(exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Run summary saved to {summary_path}")

    return summary


# --------- Entry point ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SentinelTrack - Intelligent Face Tracker & Visitor Counter")
    parser.add_argument("config_pos", nargs="?", default=None, help="Optional positional config path (e.g. config.json)")
    parser.add_argument("--config", default="config.json", help="Path to config.json (default: config.json)")
    parser.add_argument("--video", default=None, help="Override video file path (e.g. data/sample.mp4)")
    parser.add_argument("--rtsp", default=None, help="Override RTSP stream URI")
    parser.add_argument("--output", default="logs/annotated_demo.mp4", help="Path to output annotated video")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--headless", action="store_true", default=None, help="Run without display window")
    parser.add_argument("--gui", action="store_true", help="Force display window")
    parser.add_argument("--count", action="store_true", help="Inspect and display unique visitor and session counts from database")
    cli_args = parser.parse_args()

    active_config = cli_args.config_pos if cli_args.config_pos else cli_args.config

    if cli_args.count:
        config = load_config(active_config)
        init_db(config)
        with get_session() as db:
            from src.models import VisitSession, SessionStatus
            from sqlalchemy import select, func
            unique_count = FaceRepository.count_unique_identities(db)
            session_count = SessionRepository.count_sessions(db)
            active_sessions = db.scalar(
                select(func.count(VisitSession.session_id)).where(VisitSession.status == SessionStatus.ACTIVE)
            ) or 0
            print("\n================ SentinelTrack Visitor Summary ================")
            print(f"  Unique Registered Visitors : {unique_count}")
            print(f"  Total Visit Sessions       : {session_count}")
            print(f"  Currently Active Sessions  : {active_sessions}")
            print("===============================================================\n")
        close_db()
        sys.exit(0)

    headless_flag = True if cli_args.headless else (False if cli_args.gui else None)

    result = run(
        config_path=active_config,
        video=cli_args.video,
        rtsp=cli_args.rtsp,
        headless=headless_flag,
        output_video=cli_args.output,
        max_frames=cli_args.max_frames,
    )
    print("\n=== Run Summary ===")
    print(json.dumps(result, indent=2))

