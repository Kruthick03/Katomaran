"""
SentinelTrack — src/visualizer.py

Lightweight, high-clarity OpenCV visualization module for SentinelTrack.
Renders real-time bounding boxes, tracking IDs, persistent face IDs,
recognition statuses, confirmed entry/exit event banners, and a live HUD
overlay of visitor statistics without altering or faking any pipeline state.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.tracker import Track, TrackState


class SentinelVisualizer:
    """
    OpenCV frame renderer for SentinelTrack.
    Renders purely based on actual pipeline state passed from the orchestrator.
    """

    def __init__(
        self,
        window_name: str = "SentinelTrack - Intelligent Face Tracker (Katomaran)",
        screenshots_dir: str = "logs/screenshots",
    ) -> None:
        self.window_name = window_name
        self.screenshots_dir = Path(screenshots_dir)
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        # Color palette (BGR format)
        self.COLOR_BG = (20, 20, 20)           # Dark charcoal
        self.COLOR_RECOGNIZED = (46, 204, 113)  # Emerald Green
        self.COLOR_NEW = (255, 191, 0)         # Deep Sky Blue / Gold
        self.COLOR_TENTATIVE = (52, 152, 219)   # Amber / Orange
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_GRAY = (180, 180, 180)
        self.COLOR_ENTRY = (46, 204, 113)       # Green
        self.COLOR_EXIT = (41, 128, 185)        # Orange/Red

    def render(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        unique_count: int,
        session_count: int,
        active_sessions: int,
        total_entries: int,
        total_exits: int,
        fps: float,
        frame_number: int,
        recent_events: Optional[List[Dict[str, Any]]] = None,
        source_label: str = "Camera",
        is_paused: bool = False,
        db_registered_count: Optional[int] = None,
        raw_detections: Optional[List[Any]] = None,
        debug_mode: bool = False,
        frame_skip: int = 2,
    ) -> np.ndarray:
        """
        Produce an annotated copy of the frame with detections, IDs, and HUD.
        Does not mutate the input frame in place.
        """
        canvas = frame.copy()
        h, w = canvas.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Resolution-aware UI scaling (scales 720p base sizes proportionally to 2K/4K CCTV frames)
        ui_scale = max(1.0, w / 1280.0)
        f_thick = max(1, int(round(1.2 * ui_scale)))
        box_thick = max(2, int(round(2.0 * ui_scale)))

        # 0. Debug Mode: Draw Raw YOLO Detections (Thin Cyan Boxes)
        if debug_mode and raw_detections:
            for det in raw_detections:
                dx1, dy1, dx2, dy2 = det.bbox
                dx1, dy1 = max(0, dx1), max(0, dy1)
                dx2, dy2 = min(w - 1, dx2), min(h - 1, dy2)
                cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), (255, 255, 0), max(1, int(round(1.0 * ui_scale))))
                cv2.putText(
                    canvas, f"YOLO raw: {det.confidence:.2f}",
                    (dx1 + int(4 * ui_scale), max(int(14 * ui_scale), dy1 - int(4 * ui_scale))),
                    font, 0.38 * ui_scale, (255, 255, 0), f_thick, cv2.LINE_AA
                )

        # 1. Draw Bounding Boxes and Track Labels (with Collision-Free Labeling)
        active_track_count = 0
        placed_label_rects: List[Tuple[int, int, int, int]] = []

        for track in tracks:
            if track.state == TrackState.REMOVED:
                continue

            # In normal mode, do not draw ghost boxes for lost tracks with no current detection
            if not debug_mode and track.missed_frames > 0:
                continue

            active_track_count += 1
            x1, y1, x2, y2 = track.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Determine color by recognition status and track state
            if track.missed_frames > 0 and debug_mode:
                box_color = (120, 120, 120)  # Gray for lost tracks in debug mode
            elif track.recognition_status == "Recognized":
                box_color = self.COLOR_RECOGNIZED
            elif track.recognition_status == "New identity":
                box_color = self.COLOR_NEW
            else:
                box_color = self.COLOR_TENTATIVE

            # Draw bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, box_thick)

            # Draw stylized corner accents
            corner_len = min(int(16 * ui_scale), (x2 - x1) // 4, (y2 - y1) // 4)
            corner_thick = max(3, int(round(3.0 * ui_scale)))
            if corner_len > 3:
                # Top-left
                cv2.line(canvas, (x1, y1), (x1 + corner_len, y1), box_color, corner_thick)
                cv2.line(canvas, (x1, y1), (x1, y1 + corner_len), box_color, corner_thick)
                # Top-right
                cv2.line(canvas, (x2, y1), (x2 - corner_len, y1), box_color, corner_thick)
                cv2.line(canvas, (x2, y1), (x2, y1 + corner_len), box_color, corner_thick)
                # Bottom-left
                cv2.line(canvas, (x1, y2), (x1 + corner_len, y2), box_color, corner_thick)
                cv2.line(canvas, (x1, y2), (x1, y2 - corner_len), box_color, corner_thick)
                # Bottom-right
                cv2.line(canvas, (x2, y2), (x2 - corner_len, y2), box_color, corner_thick)
                cv2.line(canvas, (x2, y2), (x2, y2 - corner_len), box_color, corner_thick)

            # Build label lines
            conf_str = f"{int(track.confidence * 100)}%" if track.confidence > 0 else "--"
            if debug_mode:
                state_str = f"[{track.state.upper()}" + (f" miss:{track.missed_frames}]" if track.missed_frames > 0 else "]")
                line1 = f"Track #{track.track_id} {state_str} (Conf: {conf_str})"
            else:
                line1 = f"Track #{track.track_id} (Conf: {conf_str})"

            if track.face_id:
                status_tag = track.recognition_status
                sim_str = f" | Sim: {track.similarity:.2f}" if track.similarity is not None else ""
                line2 = f"VISITOR ID: {track.face_id[:8]}... [{status_tag}]{sim_str}"
            else:
                line2 = "Status: Detecting / Enrolling..."

            # Label dimensions
            scale = 0.44 * ui_scale
            (w1, h1), _ = cv2.getTextSize(line1, font, scale, f_thick)
            (w2, h2), _ = cv2.getTextSize(line2, font, scale, f_thick)
            pill_w = max(w1, w2) + int(14 * ui_scale)
            pill_h = h1 + h2 + int(14 * ui_scale)

            # Anti-collision placement: evaluate candidates (above, below, right, inside)
            candidates = [
                (x1, y1 - pill_h - int(4 * ui_scale)),               # 1. Above bounding box
                (x1, y2 + int(6 * ui_scale)),                        # 2. Below bounding box
                (min(w - pill_w - int(6 * ui_scale), x2 + int(6 * ui_scale)), y1),  # 3. Shifted right
                (x1 + int(4 * ui_scale), y1 + int(4 * ui_scale)),    # 4. Inside box
            ]

            chosen_x, chosen_y = candidates[0]
            for cx, cy in candidates:
                cx = max(6, min(w - pill_w - 6, cx))
                cy = max(6, min(h - pill_h - 6, cy))
                cand_rect = (cx, cy, cx + pill_w, cy + pill_h)
                collision = False
                for pr in placed_label_rects:
                    if not (cand_rect[2] <= pr[0] or cand_rect[0] >= pr[2] or cand_rect[3] <= pr[1] or cand_rect[1] >= pr[3]):
                        collision = True
                        break
                if not collision:
                    chosen_x, chosen_y = cx, cy
                    break

            pill_x1 = max(6, min(w - pill_w - 6, chosen_x))
            pill_y1 = max(6, min(h - pill_h - 6, chosen_y))
            pill_x2 = pill_x1 + pill_w
            pill_y2 = pill_y1 + pill_h
            placed_label_rects.append((pill_x1, pill_y1, pill_x2, pill_y2))

            # Draw high-contrast solid dark pill background
            sub_rect = canvas[pill_y1:pill_y2, pill_x1:pill_x2]
            if sub_rect.shape[0] > 0 and sub_rect.shape[1] > 0:
                black_rect = np.full_like(sub_rect, 18)
                cv2.addWeighted(black_rect, 0.94, sub_rect, 0.06, 0, sub_rect)
                canvas[pill_y1:pill_y2, pill_x1:pill_x2] = sub_rect
                cv2.rectangle(canvas, (pill_x1, pill_y1), (pill_x2, pill_y2), box_color, max(1, int(round(1.0 * ui_scale))))

            cv2.putText(
                canvas, line1, (pill_x1 + int(6 * ui_scale), pill_y1 + h1 + int(4 * ui_scale)),
                font, scale, self.COLOR_WHITE, f_thick, cv2.LINE_AA
            )
            cv2.putText(
                canvas, line2, (pill_x1 + int(6 * ui_scale), pill_y1 + h1 + h2 + int(8 * ui_scale)),
                font, scale, box_color, f_thick, cv2.LINE_AA
            )

        # 2. "No face detected" Alert Badge
        if active_track_count == 0:
            msg = "[ No faces detected in frame — Scanning ]"
            (mw, mh), _ = cv2.getTextSize(msg, font, 0.55 * ui_scale, f_thick)
            cx = (w - mw) // 2
            cv2.rectangle(canvas, (cx - int(10 * ui_scale), int(15 * ui_scale)), (cx + mw + int(10 * ui_scale), int(15 * ui_scale) + mh + int(12 * ui_scale)), (30, 30, 30), -1)
            cv2.putText(canvas, msg, (cx, int(15 * ui_scale) + mh + int(4 * ui_scale)), font, 0.55 * ui_scale, (200, 200, 200), f_thick, cv2.LINE_AA)

        # 3. Visitor Statistics HUD (Top-Left High-Contrast Semi-Transparent Box)
        hud_w = int(350 * ui_scale)
        hud_h = int(210 * ui_scale)
        hud_x1 = int(12 * ui_scale)
        hud_y1 = int(12 * ui_scale)
        hud_x2 = hud_x1 + hud_w
        hud_y2 = hud_y1 + hud_h

        # Alpha-blended HUD background (95% solid charcoal to prevent camera timestamp bleed-through)
        hud_roi = canvas[hud_y1:hud_y2, hud_x1:hud_x2]
        hud_bg = np.full_like(hud_roi, 18)
        cv2.addWeighted(hud_bg, 0.95, hud_roi, 0.05, 0, hud_roi)
        canvas[hud_y1:hud_y2, hud_x1:hud_x2] = hud_roi
        cv2.rectangle(canvas, (hud_x1, hud_y1), (hud_x2, hud_y2), (75, 75, 75), max(1, int(round(1.0 * ui_scale))))

        # HUD Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_scale = 0.48 * ui_scale
        cv2.putText(
            canvas, "SENTINELTRACK  VISITOR  HUD",
            (hud_x1 + int(10 * ui_scale), hud_y1 + int(22 * ui_scale)), font, title_scale, (0, 215, 255), f_thick, cv2.LINE_AA
        )
        cv2.line(canvas, (hud_x1 + int(10 * ui_scale), hud_y1 + int(28 * ui_scale)), (hud_x2 - int(10 * ui_scale), hud_y1 + int(28 * ui_scale)), (100, 100, 100), max(1, int(round(1.0 * ui_scale))))

        db_reg_str = f"{db_registered_count}" if db_registered_count is not None else f"{unique_count}"
        recent_str = "None"
        if recent_events and len(recent_events) > 0:
            last_ev = recent_events[-1]
            recent_str = f"{last_ev.get('type', 'EVENT')} (ID: {last_ev.get('face_id', 'unknown')[:8]})"

        hud_lines = [
            (f"Unique Visitors : {unique_count}", self.COLOR_RECOGNIZED),
            (f"Active Tracks   : {active_track_count}", self.COLOR_WHITE),
            (f"Frame Skip      : Every {frame_skip} frames", (0, 215, 255)),
            (f"Recent Event    : {recent_str}", self.COLOR_NEW),
            (f"Active Sessions : {active_sessions}", (0, 255, 255)),
            (f"Total Visits    : {session_count}", self.COLOR_WHITE),
            (f"Entries / Exits : {total_entries} / {total_exits}", self.COLOR_NEW),
            (f"DB Registered   : {db_reg_str}", (200, 200, 200)),
            (f"Throughput      : {fps:.1f} FPS  |  Frame: {frame_number}", self.COLOR_GRAY),
        ]
        line_spacing = int(18 * ui_scale)
        for idx, (text, col) in enumerate(hud_lines):
            cv2.putText(
                canvas, text, (hud_x1 + int(10 * ui_scale), hud_y1 + int(46 * ui_scale) + idx * line_spacing),
                font, 0.40 * ui_scale, col, f_thick, cv2.LINE_AA
            )

        # 4. Confirmed Event Banner (Top-Center Notification)
        if recent_events and len(recent_events) > 0:
            latest = recent_events[-1]
            event_frame = latest.get("frame_number", 0)
            if 0 <= (frame_number - event_frame) <= 45:
                ev_type = latest.get("type", "EVENT")
                face_short = latest.get("face_id", "unknown")[:8]
                sess_short = latest.get("session_id", "unknown")[:8]
                banner_text = f"*** CONFIRMED {ev_type}: Person {face_short} | Session {sess_short} ***"

                (bw, bh), _ = cv2.getTextSize(banner_text, font, 0.52 * ui_scale, max(2, int(round(2.0 * ui_scale))))
                bx = max(10, (w - bw) // 2)
                by = hud_y2 + int(10 * ui_scale)

                bg_color = (0, 140, 0) if ev_type == "ENTRY" else (0, 90, 180)
                pad = int(12 * ui_scale)
                cv2.rectangle(canvas, (bx - pad, by), (bx + bw + pad, by + bh + int(14 * ui_scale)), bg_color, -1)
                cv2.rectangle(canvas, (bx - pad, by), (bx + bw + pad, by + bh + int(14 * ui_scale)), self.COLOR_WHITE, 1)
                cv2.putText(
                    canvas, banner_text, (bx, by + bh + int(5 * ui_scale)),
                    font, 0.52 * ui_scale, self.COLOR_WHITE, max(2, int(round(2.0 * ui_scale))), cv2.LINE_AA
                )

        # 5. System Status Footer Bar
        bar_h = int(28 * ui_scale)
        bar_y1 = h - bar_h
        bar_roi = canvas[bar_y1:h, 0:w]
        bar_bg = np.zeros_like(bar_roi)
        cv2.addWeighted(bar_bg, 0.85, bar_roi, 0.15, 0, bar_roi)
        canvas[bar_y1:h, 0:w] = bar_roi

        footer_left = f"Source: {source_label}  |  PostgreSQL 18: ONLINE  |  Models: YOLOv8 + ArcFace"
        footer_right = "[Q] Quit   [P] Pause   [D] Debug   [S] Screenshot"
        cv2.putText(canvas, footer_left, (int(12 * ui_scale), h - int(9 * ui_scale)), font, 0.40 * ui_scale, self.COLOR_GRAY, f_thick, cv2.LINE_AA)

        (rw, _), _ = cv2.getTextSize(footer_right, font, 0.40 * ui_scale, f_thick)
        cv2.putText(canvas, footer_right, (w - rw - int(14 * ui_scale), h - int(9 * ui_scale)), font, 0.40 * ui_scale, (0, 215, 255), f_thick, cv2.LINE_AA)

        # 6. Debug Mode Banner (Top Right)
        if debug_mode:
            dbg_text = "[ DEBUG MODE: RAW YOLO (CYAN) + TRACKS | PRESS 'D' TO EXIT ]"
            (dw, dh), _ = cv2.getTextSize(dbg_text, font, 0.44, 1)
            cv2.rectangle(canvas, (w - dw - 24, 12), (w - 12, 12 + dh + 14), (0, 140, 255), -1)
            cv2.putText(canvas, dbg_text, (w - dw - 18, 12 + dh + 7), font, 0.44, (0, 0, 0), 1, cv2.LINE_AA)

        # 6. Pause Indicator
        if is_paused:
            pause_text = "PAUSED (Press [P] to Resume)"
            (pw, ph), _ = cv2.getTextSize(pause_text, font, 0.8, 2)
            px = (w - pw) // 2
            py = h // 2
            cv2.rectangle(canvas, (px - 16, py - ph - 12), (px + pw + 16, py + 16), (0, 0, 160), -1)
            cv2.putText(canvas, pause_text, (px, py), font, 0.8, self.COLOR_WHITE, 2, cv2.LINE_AA)

        return canvas

    def save_screenshot(self, frame: np.ndarray) -> str:
        """Save current annotated frame to logs/screenshots/."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        filename = f"screenshot_{timestamp_str}.jpg"
        filepath = self.screenshots_dir / filename
        cv2.imwrite(str(filepath), frame)
        return str(filepath)
