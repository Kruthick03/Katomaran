"""
SentinelTrack — scripts/check_environment.py

Pre-flight environment checker. Verifies all required dependencies,
model availability, database connectivity and configuration.
Run this before starting the application.

Usage:
    python scripts/check_environment.py
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _check(label: str, ok: bool, detail: str = "") -> bool:
    status = "  [OK]   " if ok else "  [FAIL] "
    msg = f"{status}{label}"
    if detail:
        msg += f": {detail}"
    print(msg)
    return ok


def check_python_version():
    major, minor = sys.version_info[:2]
    ok = (major == 3 and minor >= 10)
    return _check(
        "Python version",
        ok,
        f"{sys.version} (requires >=3.10)"
    )


def check_packages():
    required = [
        ("ultralytics", "ultralytics"),
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("insightface", "insightface"),
        ("onnxruntime", "onnxruntime"),
        ("sqlalchemy", "sqlalchemy"),
        ("alembic", "alembic"),
        ("psycopg2", "psycopg2-binary"),
        ("pydantic", "pydantic"),
        ("pydantic_settings", "pydantic-settings"),
        ("dotenv", "python-dotenv"),
        ("scipy", "scipy"),
        ("psutil", "psutil"),
        ("pytest", "pytest"),
    ]
    all_ok = True
    for import_name, pkg_name in required:
        try:
            importlib.import_module(import_name)
            _check(f"Package: {pkg_name}", True)
        except ImportError:
            _check(f"Package: {pkg_name}", False, f"run: pip install {pkg_name}")
            all_ok = False
    return all_ok


def check_config():
    config_path = Path("config.json")
    exists = config_path.exists()
    if not exists:
        _check("config.json", False, "Copy config.example.json to config.json")
        return False
    try:
        from src.config import load_config
        cfg = load_config("config.json")
        _check("config.json", True, f"source_type={cfg.source.type}")
        return True
    except Exception as exc:
        _check("config.json", False, str(exc))
        return False


def check_env_vars():
    from dotenv import load_dotenv
    load_dotenv()

    required_vars = [
        "SENTINEL_DB_HOST",
        "SENTINEL_DB_PORT",
        "SENTINEL_DB_NAME",
        "SENTINEL_DB_USER",
        "SENTINEL_DB_PASSWORD",
    ]
    all_ok = True
    for var in required_vars:
        val = os.getenv(var)
        ok = val is not None and len(val) > 0
        _check(f"Env: {var}", ok, "(set)" if ok else "missing — check .env file")
        if not ok:
            all_ok = False
    return all_ok


def check_database():
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from src.config import load_config
        from src.database import init_db, check_connection
        cfg = load_config("config.json")
        init_db(cfg)
        ok = check_connection(retries=2, delay=1.0)
        return _check("PostgreSQL connection", ok)
    except Exception as exc:
        return _check("PostgreSQL connection", False, str(exc))


def check_directories():
    dirs = [
        "logs/entries", "logs/exits", "models", "data",
        "migrations/versions", "tests"
    ]
    all_ok = True
    for d in dirs:
        exists = Path(d).exists()
        _check(f"Directory: {d}", exists)
        if not exists:
            all_ok = False
    return all_ok


def check_model_files():
    from pathlib import Path
    model_path = Path("models/yolov8n-face.pt")
    exists = model_path.exists()
    _check(
        "YOLO face model",
        exists,
        str(model_path) if exists else "Will be downloaded on first run from ultralytics hub"
    )
    # InsightFace downloads automatically to ~/.insightface/
    insight_path = Path.home() / ".insightface" / "models" / "buffalo_sc"
    _check(
        "InsightFace buffalo_sc",
        insight_path.exists(),
        str(insight_path) if insight_path.exists() else "Will be downloaded on first run"
    )
    return True  # Not a hard failure


def main():
    print("=" * 60)
    print("SentinelTrack — Environment Check")
    print("=" * 60)

    results = []

    print("\n[1] Python Environment")
    results.append(check_python_version())
    results.append(check_packages())

    print("\n[2] Configuration")
    results.append(check_config())
    results.append(check_env_vars())

    print("\n[3] Database")
    results.append(check_database())

    print("\n[4] Filesystem")
    results.append(check_directories())
    check_model_files()

    print("\n" + "=" * 60)
    failed = results.count(False)
    if failed == 0:
        print("All checks PASSED. Ready to run SentinelTrack.")
    else:
        print(f"{failed} check(s) FAILED. Resolve issues above before running.")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
