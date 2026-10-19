"""
SentinelTrack — scripts/run_sample.py

Run SentinelTrack on a sample video file.
Processes the video, writes DB records, evidence images, and events.log.

Usage:
    python scripts/run_sample.py
    python scripts/run_sample.py --config config.json --video data/sample.mp4
    python scripts/run_sample.py --headless
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Run SentinelTrack on a video file.")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--video", default=None, help="Override video path")
    parser.add_argument("--headless", action="store_true", help="No display window")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Patch config for sample mode
    import os
    if args.video:
        # Temporarily write patched config
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            cfg["source"]["type"] = "file"
            cfg["source"]["video_path"] = args.video
            if args.headless:
                cfg["headless"] = True
            patched_path = Path("config_sample_run.json")
            with open(patched_path, "w") as f:
                json.dump(cfg, f, indent=2)
            effective_config = str(patched_path)
        else:
            effective_config = args.config
    else:
        effective_config = args.config

    from src.main import run
    summary = run(effective_config)

    print("\n=== Sample Run Complete ===")
    print(json.dumps(summary, indent=2))

    # Cleanup temp config
    if args.video and Path("config_sample_run.json").exists():
        Path("config_sample_run.json").unlink()


if __name__ == "__main__":
    main()
