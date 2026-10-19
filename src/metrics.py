"""
metrics.py
==========
Performance and operational metrics collection for the SentinelTrack pipeline.

``MetricsCollector`` gathers fine-grained latency measurements and event
counters as the pipeline runs, then exposes them via several export surfaces:
dict summary, structured log report, and JSON file.

Design decisions
----------------
- **No external metrics server dependency** --- the class is self-contained and
  works without Prometheus, StatsD, or any network sink.  It records data
  in-process and lets the caller decide how to forward it.
- **Thread safety** --- counters use plain ``int`` addition which is atomic in
  CPython, and timing lists are appended to (also GIL-protected in CPython).
  If you need true multi-threaded safety with a non-CPython interpreter,
  wrap mutating methods with a ``threading.Lock``.
- **psutil isolation** --- all ``psutil`` calls are wrapped in try/except so the
  metrics system never crashes the pipeline even if the package is missing or
  the OS refuses access to process stats.
- **Latency storage** --- only the *last* ``_MAX_TIMING_SAMPLES`` samples are
  retained for each timing category to avoid unbounded memory growth.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Optional

logger = logging.getLogger(__name__)

# Maximum number of raw latency samples kept in memory per category.
# Older samples are silently discarded via the deque's maxlen mechanism.
_MAX_TIMING_SAMPLES: int = 10_000


class MetricsCollector:
    """Collects, aggregates, and exports SentinelTrack pipeline metrics.

    All public methods are exception-safe: any failure is silently swallowed
    (with a DEBUG log) so metrics collection never interrupts the pipeline.

    Usage
    -----
    Create one instance per pipeline run and pass it to every component that
    needs to record metrics::

        metrics = MetricsCollector()

        # In the main loop:
        start = metrics.record_frame_start()
        # ... process frame ...
        metrics.record_frame_end(start)

        # Periodically:
        metrics.report_to_log(logger)
        metrics.export_json("/tmp/sentineltrack_metrics.json")
    """

    def __init__(self) -> None:
        # ------------------------------------------------------------------
        # Event counters
        # ------------------------------------------------------------------
        self.frames_processed: int = 0
        """Total frames that completed the full pipeline without being dropped."""

        self.frames_dropped: int = 0
        """Frames skipped due to queue overflow, decode failure, or similar."""

        self.detections_total: int = 0
        """Cumulative face detections across all processed frames."""

        self.registrations: int = 0
        """New face identities registered (first-ever appearance)."""

        self.recognitions: int = 0
        """Face recognition attempts (embedding --- identity lookup)."""

        self.entries: int = 0
        """Session entry events persisted to the database."""

        self.exits: int = 0
        """Session exit events persisted to the database."""

        self.errors: int = 0
        """Recoverable errors logged during pipeline execution."""

        self.reconnects: int = 0
        """Camera/stream reconnection attempts."""

        # ------------------------------------------------------------------
        # Latency sample buffers (bounded deques)
        # ------------------------------------------------------------------
        self.detection_times: Deque[float] = deque(maxlen=_MAX_TIMING_SAMPLES)
        """Per-frame detection latency in milliseconds."""

        self.recognition_times: Deque[float] = deque(maxlen=_MAX_TIMING_SAMPLES)
        """Per-recognition embedding + identity-lookup latency in milliseconds."""

        self.db_write_times: Deque[float] = deque(maxlen=_MAX_TIMING_SAMPLES)
        """Per-event database write latency in milliseconds."""

        self.total_frame_times: Deque[float] = deque(maxlen=_MAX_TIMING_SAMPLES)
        """End-to-end per-frame processing time in milliseconds."""

        # ------------------------------------------------------------------
        # Internal bookkeeping
        # ------------------------------------------------------------------
        self._start_wall: float = time.perf_counter()
        """Epoch (perf_counter) at which this collector was instantiated."""

        self._start_datetime: datetime = datetime.now(tz=timezone.utc)
        """UTC wall-clock time at instantiation (for reporting purposes)."""

        logger.debug("MetricsCollector initialised at %s", self._start_datetime.isoformat())

    # ------------------------------------------------------------------
    # Frame lifecycle helpers
    # ------------------------------------------------------------------

    def record_frame_start(self) -> float:
        """Return the current ``perf_counter`` value as a frame start marker.

        Returns
        -------
        float
            Opaque timestamp to pass back into ``record_frame_end``.
        """
        return time.perf_counter()

    def record_frame_end(self, start: float) -> None:
        """Record total frame processing time.

        Parameters
        ----------
        start:
            Value previously returned by ``record_frame_start``.
        """
        try:
            elapsed_ms = (time.perf_counter() - start) * 1_000.0
            self.total_frame_times.append(elapsed_ms)
        except Exception:  # noqa: BLE001
            pass  # Metrics must never crash the pipeline

    # ------------------------------------------------------------------
    # Stage-specific latency recorders
    # ------------------------------------------------------------------

    def record_detection_time(self, elapsed_ms: float) -> None:
        """Record a single face-detection stage latency.

        Parameters
        ----------
        elapsed_ms:
            Elapsed time in milliseconds for the detection call.
        """
        try:
            self.detection_times.append(float(elapsed_ms))
        except Exception:  # noqa: BLE001
            pass

    def record_recognition_time(self, elapsed_ms: float) -> None:
        """Record a single recognition-stage latency (embedding + lookup).

        Parameters
        ----------
        elapsed_ms:
            Elapsed time in milliseconds for the recognition call.
        """
        try:
            self.recognition_times.append(float(elapsed_ms))
        except Exception:  # noqa: BLE001
            pass

    def record_db_write_time(self, elapsed_ms: float) -> None:
        """Record a single database write latency.

        Parameters
        ----------
        elapsed_ms:
            Elapsed time in milliseconds for the DB write.
        """
        try:
            self.db_write_times.append(float(elapsed_ms))
        except Exception:  # noqa: BLE001
            pass

    def record_frame_time(self, elapsed: float) -> None:
        """Record total frame processing time in seconds or milliseconds."""
        try:
            ms = elapsed * 1000.0 if elapsed < 10.0 else elapsed
            self.total_frame_times.append(float(ms))
        except Exception:
            pass

    def increment_frames(self, count: int = 1) -> None:
        """Alias for increment_frames_processed."""
        self.frames_processed += count

    def increment_frames_processed(self, count: int = 1) -> None:
        """Increment the ``frames_processed`` counter."""
        self.frames_processed += count

    def increment_frames_dropped(self, count: int = 1) -> None:
        """Increment the ``frames_dropped`` counter."""
        self.frames_dropped += count

    def increment_detections(self, count: int = 1) -> None:
        """Increment the ``detections_total`` counter."""
        self.detections_total += count

    def increment_registrations(self, count: int = 1) -> None:
        """Increment the ``registrations`` counter."""
        self.registrations += count

    def increment_recognitions(self, count: int = 1) -> None:
        """Increment the ``recognitions`` counter."""
        self.recognitions += count

    def increment_entries(self, count: int = 1) -> None:
        """Increment the ``entries`` counter."""
        self.entries += count

    def increment_exits(self, count: int = 1) -> None:
        """Increment the ``exits`` counter."""
        self.exits += count

    def increment_errors(self, count: int = 1) -> None:
        """Increment the ``errors`` counter."""
        self.errors += count

    def increment_reconnects(self, count: int = 1) -> None:
        """Increment the ``reconnects`` counter."""
        self.reconnects += count

    # ------------------------------------------------------------------
    # Aggregation helpers (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _stats(samples: Deque[float]) -> Dict[str, Optional[float]]:
        """Compute mean and max for a deque of float samples.

        Returns
        -------
        dict
            ``{"mean_ms": float | None, "max_ms": float | None, "count": int}``
        """
        lst = list(samples)
        if not lst:
            return {"mean_ms": None, "max_ms": None, "count": 0}
        return {
            "mean_ms": round(sum(lst) / len(lst), 3),
            "max_ms": round(max(lst), 3),
            "count": len(lst),
        }

    def _elapsed_seconds(self) -> float:
        """Return total elapsed seconds since this collector was created."""
        return time.perf_counter() - self._start_wall

    # ------------------------------------------------------------------
    # Public aggregation and export
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Build a comprehensive summary dictionary.

        Includes all counters, computed latency statistics (mean/max), an
        estimated FPS value, and runtime duration.

        Returns
        -------
        dict
            Flat-ish dictionary suitable for JSON serialisation or logging.
        """
        elapsed = self._elapsed_seconds()
        fps = (self.frames_processed / elapsed) if elapsed > 0 else 0.0

        return {
            # Meta
            "collected_at": datetime.now(tz=timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "started_at": self._start_datetime.isoformat(),

            # Counters
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "detections_total": self.detections_total,
            "registrations": self.registrations,
            "recognitions": self.recognitions,
            "entries": self.entries,
            "exits": self.exits,
            "errors": self.errors,
            "reconnects": self.reconnects,

            # Derived
            "fps_estimate": round(fps, 2),
            "drop_rate_pct": (
                round(100.0 * self.frames_dropped / max(1, self.frames_processed + self.frames_dropped), 2)
            ),

            # Latency stats
            "detection_latency": self._stats(self.detection_times),
            "recognition_latency": self._stats(self.recognition_times),
            "db_write_latency": self._stats(self.db_write_times),
            "total_frame_latency": self._stats(self.total_frame_times),
        }

    def report_to_log(self, report_logger: Optional[logging.Logger] = None) -> None:
        """Emit a formatted performance report to the given logger.

        Parameters
        ----------
        report_logger:
            Logger to write to.  Falls back to the module-level logger when
            ``None``.
        """
        log = report_logger or logger
        try:
            s = self.get_summary()
            det = s["detection_latency"]
            rec = s["recognition_latency"]
            db_ = s["db_write_latency"]
            frm = s["total_frame_latency"]

            report_lines = [
                "--------------------------------- SentinelTrack Performance Report ---------------------------------",
                f"  Runtime          : {s['elapsed_seconds']:.1f}s",
                f"  FPS (estimate)   : {s['fps_estimate']:.1f}",
                f"  Frames processed : {s['frames_processed']}",
                f"  Frames dropped   : {s['frames_dropped']} ({s['drop_rate_pct']}%)",
                f"  Detections       : {s['detections_total']}",
                f"  Recognitions     : {s['recognitions']}",
                f"  Registrations    : {s['registrations']}",
                f"  Entries / Exits  : {s['entries']} / {s['exits']}",
                f"  Errors           : {s['errors']}",
                f"  Reconnects       : {s['reconnects']}",
                "  --------- Latency (mean / max ms) ---------------------------------------------------------------------------------",
                f"  Detection        : {det['mean_ms']} / {det['max_ms']}  (n={det['count']})",
                f"  Recognition      : {rec['mean_ms']} / {rec['max_ms']}  (n={rec['count']})",
                f"  DB write         : {db_['mean_ms']} / {db_['max_ms']}  (n={db_['count']})",
                f"  Total frame      : {frm['mean_ms']} / {frm['max_ms']}  (n={frm['count']})",
                "---------------------------------------------------------------------------------------------------------------------------------------------------------------------------",
            ]
            for line in report_lines:
                log.info(line)
        except Exception as err:  # noqa: BLE001
            log.debug("MetricsCollector.report_to_log failed: %s", err)

    def get_resource_usage(self) -> Dict[str, Any]:
        """Query current process resource usage via ``psutil``.

        Returns a dictionary with at least the keys ``cpu_percent``,
        ``memory_mb``, and ``available_memory_mb``.  If ``psutil`` is not
        installed or the query fails, the values are ``None`` and the failure
        is logged at DEBUG level (never raised).

        Returns
        -------
        dict
            Resource snapshot; values are ``None`` on failure.
        """
        result: Dict[str, Any] = {
            "cpu_percent": None,
            "memory_mb": None,
            "available_memory_mb": None,
            "psutil_available": False,
        }
        try:
            import psutil  # noqa: PLC0415 (local import to keep module importable without psutil)

            result["psutil_available"] = True
            proc = psutil.Process()

            # CPU percent since last call (interval=None is non-blocking)
            result["cpu_percent"] = proc.cpu_percent(interval=None)

            # RSS (resident set size) in megabytes
            mem_info = proc.memory_info()
            result["memory_mb"] = round(mem_info.rss / (1024 * 1024), 1)

            # System-wide available memory
            vm = psutil.virtual_memory()
            result["available_memory_mb"] = round(vm.available / (1024 * 1024), 1)

            logger.debug(
                "Resource usage --- CPU: %.1f%%  RSS: %.1f MB  Available: %.1f MB",
                result["cpu_percent"],
                result["memory_mb"],
                result["available_memory_mb"],
            )
        except ImportError:
            logger.debug("psutil not installed --- resource metrics unavailable.")
        except Exception as err:  # noqa: BLE001
            logger.debug("psutil query failed: %s", err)

        return result

    def export_json(self, path: str) -> bool:
        """Write the current metrics summary to a JSON file.

        The file is written atomically by first writing to a ``.tmp`` sibling
        and then renaming, so a crash mid-write doesn't corrupt the previous
        good snapshot.

        Parameters
        ----------
        path:
            Destination file path.  Parent directories must already exist.

        Returns
        -------
        bool
            ``True`` on success, ``False`` on any failure.
        """
        try:
            summary = self.get_summary()
            # Also attach a resource snapshot to the JSON export
            summary["resource_usage"] = self.get_resource_usage()

            dest = Path(path)
            tmp_path = dest.with_suffix(".tmp")

            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2, default=str)

            tmp_path.replace(dest)
            logger.debug("Metrics exported to %s", dest)
            return True

        except Exception as err:  # noqa: BLE001
            logger.warning("MetricsCollector.export_json(%s) failed: %s", path, err)
            return False

    # ------------------------------------------------------------------
    # Context-manager support --- convenient for scoped timing
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MetricsCollector("
            f"frames={self.frames_processed}, "
            f"errors={self.errors}, "
            f"fps---{round(self.frames_processed / max(1e-6, self._elapsed_seconds()), 1)}"
            f")"
        )

