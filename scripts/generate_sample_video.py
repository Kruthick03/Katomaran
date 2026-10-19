"""
SentinelTrack — scripts/generate_sample_video.py

Generates a realistic multi-person video (data/sample.mp4) demonstrating:
  - Sequence 1 (frames 0-25): Person A enters and remains visible.
  - Sequence 2 (frames 26-55): Person A exits (empty background for 30 frames > exit_confirmation_frames).
  - Sequence 3 (frames 56-80): Person A returns (demonstrating re-identification without duplicate registration).
  - Sequence 4 (frames 81-110): Person B enters (demonstrating registration of a second distinct identity).

Uses real face photographs (Ada Lovelace and Albert Einstein) resized onto an office-style background.
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def create_office_background(width: int = 640, height: int = 480) -> np.ndarray:
    """Generate a clean synthetic office lobby background."""
    bg = np.full((height, width, 3), 235, dtype=np.uint8)
    # Floor
    cv2.rectangle(bg, (0, 360), (width, height), (180, 175, 170), -1)
    # Baseboard
    cv2.line(bg, (0, 360), (width, 360), (120, 115, 110), 3)
    # Door frame on the wall
    cv2.rectangle(bg, (220, 80), (420, 360), (200, 195, 190), -1)
    cv2.rectangle(bg, (220, 80), (420, 360), (140, 135, 130), 2)
    # Sign above door
    cv2.putText(bg, "ENTRANCE", (260, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 85, 80), 2)
    return bg


def overlay_face(
    canvas: np.ndarray,
    face_img: np.ndarray,
    center_x: int,
    center_y: int,
) -> None:
    """Blends a face image onto the canvas at (center_x, center_y)."""
    h, w = face_img.shape[:2]
    x1 = max(0, center_x - w // 2)
    y1 = max(0, center_y - h // 2)
    x2 = min(canvas.shape[1], x1 + w)
    y2 = min(canvas.shape[0], y1 + h)

    face_crop = face_img[0 : (y2 - y1), 0 : (x2 - x1)]
    canvas[y1:y2, x1:x2] = face_crop


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    out_video_path = data_dir / "sample.mp4"

    # Load source portraits
    img_a = cv2.imread(str(data_dir / "person_a.jpg"))
    img_b = cv2.imread(str(data_dir / "person_b.jpg"))

    if img_a is None or img_b is None:
        raise FileNotFoundError("Missing person_a.jpg or person_b.jpg in data/")

    # Resize portraits to face region ~180x220
    face_a = cv2.resize(img_a[550:1250, 1000:1550], (180, 220))
    face_b = cv2.resize(img_b[900:2600, 1000:2400], (180, 220))

    fps = 10.0
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    base_bg = create_office_background(width, height)
    total_frames = 110

    print(f"Generating {total_frames} frames to {out_video_path}...")

    for frame_idx in range(total_frames):
        frame = base_bg.copy()

        # Sequence 1 (0 to 25): Person A enters and stays
        if 0 <= frame_idx <= 25:
            # Person A moves towards center
            x = int(150 + (frame_idx / 25.0) * 170)  # 150 -> 320
            overlay_face(frame, face_a, x, 220)

        # Sequence 2 (26 to 55): Person A exits (empty background for 30 frames)
        elif 26 <= frame_idx <= 55:
            pass  # Empty background

        # Sequence 3 (56 to 80): Person A returns
        elif 56 <= frame_idx <= 80:
            progress = (frame_idx - 56) / 24.0
            x = int(180 + progress * 140)  # 180 -> 320
            overlay_face(frame, face_a, x, 220)

        # Sequence 4 (81 to 110): Person B enters from right
        elif 81 <= frame_idx <= 110:
            progress = (frame_idx - 81) / 29.0
            x = int(480 - progress * 160)  # 480 -> 320
            overlay_face(frame, face_b, x, 220)

        out.write(frame)

    out.release()
    print(f"Sample video generated successfully at {out_video_path} ({out_video_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
