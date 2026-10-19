"""
SentinelTrack — scripts/run_rtsp.py

Run SentinelTrack with an RTSP camera stream.
RTSP credentials must be in RTSP_USERNAME / RTSP_PASSWORD env vars.

Usage:
    # Set credentials in .env, then:
    python scripts/run_rtsp.py --rtsp rtsp://192.168.1.100:554/stream1
    python scripts/run_rtsp.py  # Uses rtsp_uri from config.json
    python scripts/run_rtsp.py --headless
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Run SentinelTrack on RTSP stream.")
    parser.add_argument("--rtsp", default=None, help="RTSP URI (override config)")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Patch config for RTSP mode
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = json.load(f)

    cfg["source"]["type"] = "rtsp"
    if args.rtsp:
        # Only set URI if provided — never log it
        cfg["source"]["rtsp_uri"] = args.rtsp
        print("RTSP URI configured (credentials hidden)")

    if args.headless:
        cfg["headless"] = True

    if not cfg["source"].get("rtsp_uri"):
        print("ERROR: No RTSP URI configured. Set in config.json or use --rtsp flag.")
        sys.exit(1)

    # Write temp config
    patched_path = Path("config_rtsp_run.json")
    with open(patched_path, "w") as f:
        json.dump(cfg, f, indent=2)

    try:
        from src.main import run
        summary = run(str(patched_path))
        print("\n=== RTSP Run Complete ===")
        print(json.dumps(summary, indent=2))
    finally:
        if patched_path.exists():
            patched_path.unlink()


if __name__ == "__main__":
    main()
