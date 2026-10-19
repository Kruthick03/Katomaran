"""
SentinelTrack — scripts/dashboard_server.py

Lightweight Python HTTP dashboard server for SentinelTrack.
Uses only the Python standard library (http.server) and existing SQLAlchemy/psycopg2 stack.
Zero additional external framework dependencies required.

Features:
  - Serves live HTML dashboard on http://localhost:8000
  - JSON REST endpoints:
      /api/stats    -> Visitor, session, entry/exit counts
      /api/events   -> Recent 50 audit events with metadata
      /api/visitors -> Registered face identities with visit histories
  - Static evidence asset server:
      /evidence/... -> Serves JPG facial crop snapshots from logs/entries/ and logs/exits/
      /video        -> Streams logs/annotated_demo.mp4

Usage:
  python scripts/dashboard_server.py
  python scripts/dashboard_server.py --port 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import os
import sys
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, unquote

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.config import load_config
from src.database import init_db, get_session
from src.models import Event, EventType, Face, SessionStatus, VisitSession
from src.repositories import FaceRepository, SessionRepository
from sqlalchemy import select, func, desc

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("dashboard_server")


def get_dashboard_html() -> str:
    """Returns self-contained modern HTML/Tailwind dashboard."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SentinelTrack — Visitor Intelligence Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { background-color: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .card { background-color: #161b22; border: 1px solid #30363d; }
    .badge-entry { background-color: rgba(46, 160, 67, 0.2); color: #3fb950; border: 1px solid rgba(46, 160, 67, 0.4); }
    .badge-exit { background-color: rgba(248, 81, 73, 0.2); color: #f85149; border: 1px solid rgba(248, 81, 73, 0.4); }
    .badge-reg { background-color: rgba(56, 139, 253, 0.2); color: #58a6ff; border: 1px solid rgba(56, 139, 253, 0.4); }
  </style>
</head>
<body class="min-h-screen p-6">
  <header class="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between pb-6 mb-6 border-b border-[#30363d] gap-4">
    <div>
      <div class="flex items-center gap-3">
        <div class="w-3.5 h-3.5 rounded-full bg-emerald-500 animate-pulse"></div>
        <h1 class="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
          <span>SentinelTrack</span>
          <span class="text-xs uppercase px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800">Live Dashboard</span>
        </h1>
      </div>
      <p class="text-sm text-gray-400 mt-1">Intelligent Face Tracker &bull; Auto-Registration &bull; Unique Visitor Counter</p>
    </div>
    <div class="flex items-center gap-4">
      <span id="last-updated" class="text-xs text-gray-500">Live Updating...</span>
      <button onclick="refreshData()" class="px-3 py-1.5 text-xs font-semibold rounded-md bg-[#21262d] hover:bg-[#30363d] text-gray-200 border border-[#30363d] transition">
        Refresh Now
      </button>
    </div>
  </header>

  <main class="max-w-7xl mx-auto space-y-6">
    <!-- KPI Summary Grid -->
    <section class="grid grid-cols-2 md:grid-cols-5 gap-4">
      <div class="card p-4 rounded-xl">
        <p class="text-xs font-medium text-gray-400 uppercase tracking-wider">Unique Visitors</p>
        <p id="stat-unique" class="text-3xl font-black text-emerald-400 mt-2">--</p>
        <p class="text-[11px] text-gray-500 mt-1">Biometrically distinct identities</p>
      </div>
      <div class="card p-4 rounded-xl">
        <p class="text-xs font-medium text-gray-400 uppercase tracking-wider">Active Sessions</p>
        <p id="stat-active" class="text-3xl font-black text-amber-400 mt-2">--</p>
        <p class="text-[11px] text-gray-500 mt-1">Currently inside premises</p>
      </div>
      <div class="card p-4 rounded-xl">
        <p class="text-xs font-medium text-gray-400 uppercase tracking-wider">Total Visits</p>
        <p id="stat-sessions" class="text-3xl font-black text-sky-400 mt-2">--</p>
        <p class="text-[11px] text-gray-500 mt-1">Total completed visit cycles</p>
      </div>
      <div class="card p-4 rounded-xl">
        <p class="text-xs font-medium text-gray-400 uppercase tracking-wider">Total Entries</p>
        <p id="stat-entries" class="text-3xl font-black text-green-300 mt-2">--</p>
        <p class="text-[11px] text-gray-500 mt-1">Confirmed entry transitions</p>
      </div>
      <div class="card p-4 rounded-xl">
        <p class="text-xs font-medium text-gray-400 uppercase tracking-wider">Total Exits</p>
        <p id="stat-exits" class="text-3xl font-black text-rose-400 mt-2">--</p>
        <p class="text-[11px] text-gray-500 mt-1">Confirmed exit transitions</p>
      </div>
    </section>

    <!-- Main Content Split -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Live Audit Feed -->
      <section class="lg:col-span-2 card rounded-xl p-5">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-base font-semibold text-white flex items-center gap-2">
            <span>Verified Entry / Exit Event Stream</span>
            <span id="events-count" class="text-xs font-normal text-gray-400">Loading...</span>
          </h2>
          <span class="text-xs text-gray-500">PostgreSQL Audit Trail</span>
        </div>
        <div class="overflow-x-auto max-h-[580px] overflow-y-auto pr-2">
          <table class="w-full text-left text-sm">
            <thead class="text-xs uppercase bg-[#0d1117] text-gray-400 sticky top-0 border-b border-[#30363d]">
              <tr>
                <th class="py-2.5 px-3">Face Crop</th>
                <th class="py-2.5 px-3">Event</th>
                <th class="py-2.5 px-3">Visitor ID</th>
                <th class="py-2.5 px-3">Session</th>
                <th class="py-2.5 px-3">Timestamp (UTC)</th>
              </tr>
            </thead>
            <tbody id="events-tbody" class="divide-y divide-[#21262d]">
              <tr><td colspan="5" class="py-8 text-center text-gray-500">Fetching events from database...</td></tr>
            </tbody>
          </table>
        </div>
      </section>

      <!-- Registered Visitors Gallery -->
      <section class="card rounded-xl p-5 flex flex-col">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-base font-semibold text-white">Registered Identities</h2>
          <span class="text-xs text-emerald-400 font-medium">ArcFace 512-D</span>
        </div>
        <p class="text-xs text-gray-400 mb-4">Enrolled biometric gallery with PostgreSQL persistence.</p>
        <div id="visitors-gallery" class="space-y-3 overflow-y-auto max-h-[520px] pr-2">
          <p class="text-xs text-gray-500 text-center py-6">Loading visitor gallery...</p>
        </div>
      </section>
    </div>
  </main>

  <script>
    async function refreshData() {
      try {
        const [statsRes, eventsRes, visitorsRes] = await Promise.all([
          fetch('/api/stats'),
          fetch('/api/events'),
          fetch('/api/visitors')
        ]);
        const stats = await statsRes.json();
        const events = await eventsRes.json();
        const visitors = await visitorsRes.json();

        // Update KPIs
        document.getElementById('stat-unique').textContent = stats.unique_visitors;
        document.getElementById('stat-active').textContent = stats.active_sessions;
        document.getElementById('stat-sessions').textContent = stats.total_sessions;
        document.getElementById('stat-entries').textContent = stats.total_entries;
        document.getElementById('stat-exits').textContent = stats.total_exits;
        document.getElementById('last-updated').textContent = 'Updated ' + new Date().toLocaleTimeString();
        document.getElementById('events-count').textContent = `(${events.length} shown)`;

        // Update Events Table
        const tbody = document.getElementById('events-tbody');
        if (events.length === 0) {
          tbody.innerHTML = '<tr><td colspan="5" class="py-6 text-center text-gray-500">No events logged yet.</td></tr>';
        } else {
          tbody.innerHTML = events.map(ev => {
            const badgeClass = ev.event_type === 'entry' ? 'badge-entry' : 'badge-exit';
            const imgTag = ev.image_url 
              ? `<img src="${ev.image_url}" class="w-11 h-11 object-cover rounded border border-[#30363d] bg-black">`
              : `<div class="w-11 h-11 rounded bg-[#21262d] border border-[#30363d] flex items-center justify-center text-[10px] text-gray-500">N/A</div>`;
            return `
              <tr class="hover:bg-[#1f242c] transition">
                <td class="py-2 px-3">${imgTag}</td>
                <td class="py-2 px-3"><span class="px-2 py-0.5 rounded text-xs font-semibold uppercase ${badgeClass}">${ev.event_type}</span></td>
                <td class="py-2 px-3 font-mono text-xs text-gray-300" title="${ev.face_id}">${ev.face_id ? ev.face_id.slice(0, 8) + '...' : 'Unknown'}</td>
                <td class="py-2 px-3 font-mono text-xs text-gray-500" title="${ev.session_id}">${ev.session_id ? ev.session_id.slice(0, 8) : '--'}</td>
                <td class="py-2 px-3 text-xs text-gray-400 whitespace-nowrap">${ev.timestamp ? ev.timestamp.replace('T', ' ').slice(0, 19) : '--'}</td>
              </tr>
            `;
          }).join('');
        }

        // Update Visitors Gallery
        const vGallery = document.getElementById('visitors-gallery');
        if (visitors.length === 0) {
          vGallery.innerHTML = '<p class="text-xs text-gray-500 text-center py-6">No registered identities yet.</p>';
        } else {
          vGallery.innerHTML = visitors.map(v => `
            <div class="flex items-center gap-3 p-2.5 rounded-lg bg-[#0d1117] border border-[#21262d]">
              ${v.crop_url 
                ? `<img src="${v.crop_url}" class="w-12 h-12 rounded object-cover border border-[#30363d] bg-black">` 
                : `<div class="w-12 h-12 rounded bg-[#21262d] flex items-center justify-center text-xs text-gray-500 font-mono">ID</div>`}
              <div class="min-w-0 flex-1">
                <p class="font-mono text-xs font-semibold text-white truncate" title="${v.face_id}">ID: ${v.face_id.slice(0, 8)}...</p>
                <p class="text-[11px] text-gray-400">Total Visits: <span class="text-emerald-400 font-bold">${v.visit_count}</span></p>
                <p class="text-[10px] text-gray-500">Last Seen: ${v.last_seen ? v.last_seen.replace('T', ' ').slice(11, 19) : '--'}</p>
              </div>
            </div>
          `).join('');
        }

      } catch (err) {
        console.error("Dashboard refresh error:", err);
      }
    }

    refreshData();
    setInterval(refreshData, 3000); // 3-second auto-poll
  </script>
</body>
</html>"""


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for stats, events, face crops, and dashboard view."""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path in ("", "/", "/index.html"):
            self._serve_html(get_dashboard_html())
        elif path == "/api/stats":
            self._serve_stats()
        elif path == "/api/events":
            self._serve_events()
        elif path == "/api/visitors":
            self._serve_visitors()
        elif path.startswith("/evidence/"):
            self._serve_evidence(path[len("/evidence/"):])
        elif path == "/video":
            self._serve_video()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def _serve_html(self, content: str):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content.encode("utf-8"))

    def _serve_json(self, data: object):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _serve_stats(self):
        try:
            with get_session() as db:
                unique_faces = FaceRepository.count_unique_identities(db)
                total_sessions = SessionRepository.count_sessions(db)
                active_sessions = db.scalar(
                    select(func.count(VisitSession.session_id)).where(VisitSession.status == SessionStatus.ACTIVE)
                ) or 0
                total_entries = db.scalar(
                    select(func.count(Event.event_id)).where(Event.event_type == EventType.ENTRY)
                ) or 0
                total_exits = db.scalar(
                    select(func.count(Event.event_id)).where(Event.event_type == EventType.EXIT)
                ) or 0

            self._serve_json({
                "unique_visitors": unique_faces,
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "total_entries": total_entries,
                "total_exits": total_exits,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            logger.error(f"Error fetching stats: {exc}")
            self._serve_json({"error": str(exc)})

    def _serve_events(self):
        try:
            with get_session() as db:
                stmt = select(Event).order_by(desc(Event.event_timestamp)).limit(50)
                events = db.scalars(stmt).all()

                data = []
                for ev in events:
                    img_url = None
                    if ev.evidence_image_path:
                        p = Path(ev.evidence_image_path).resolve()
                        if p.exists():
                            try:
                                rel = p.relative_to(_ROOT).as_posix()
                                img_url = f"/evidence/{rel}"
                            except ValueError:
                                img_url = None

                    data.append({
                        "event_id": str(ev.event_id),
                        "event_type": ev.event_type.value if hasattr(ev.event_type, "value") else str(ev.event_type),
                        "face_id": str(ev.face_id) if ev.face_id else None,
                        "session_id": str(ev.session_id) if ev.session_id else None,
                        "timestamp": ev.event_timestamp.isoformat() if ev.event_timestamp else None,
                        "image_url": img_url,
                    })
            self._serve_json(data)
        except Exception as exc:
            logger.error(f"Error fetching events: {exc}", exc_info=True)
            self._serve_json([])

    def _serve_visitors(self):
        try:
            with get_session() as db:
                stmt = select(Face).order_by(desc(Face.last_seen_at)).limit(30)
                faces = db.scalars(stmt).all()

                data = []
                for f in faces:
                    # Count visits
                    visit_count = db.scalar(
                        select(func.count(VisitSession.session_id)).where(VisitSession.face_id == f.face_id)
                    ) or 1

                    # Look up latest crop from events
                    ev_stmt = select(Event.evidence_image_path).where(
                        Event.face_id == f.face_id,
                        Event.evidence_image_path.isnot(None),
                    ).order_by(desc(Event.event_timestamp)).limit(1)
                    crop_path = db.scalar(ev_stmt)

                    crop_url = None
                    if crop_path:
                        p = Path(crop_path).resolve()
                        if p.exists():
                            try:
                                rel = p.relative_to(_ROOT).as_posix()
                                crop_url = f"/evidence/{rel}"
                            except ValueError:
                                crop_url = None

                    data.append({
                        "face_id": str(f.face_id),
                        "first_seen": f.first_seen_at.isoformat() if f.first_seen_at else None,
                        "last_seen": f.last_seen_at.isoformat() if f.last_seen_at else None,
                        "visit_count": visit_count,
                        "crop_url": crop_url,
                    })
            self._serve_json(data)
        except Exception as exc:
            logger.error(f"Error fetching visitors: {exc}", exc_info=True)
            self._serve_json([])

    def _serve_evidence(self, file_path_str: str):
        target = (_ROOT / file_path_str).resolve()
        # Security: verify it remains within workspace
        try:
            target.relative_to(_ROOT)
        except ValueError:
            self.send_response(403)
            self.end_headers()
            return

        if not target.exists() or not target.is_file():
            self.send_response(404)
            self.end_headers()
            return

        mime, _ = mimetypes.guess_type(str(target))
        self.send_response(200)
        self.send_header("Content-Type", mime or "image/jpeg")
        self.send_header("Content-Length", str(target.stat().st_size))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        with open(target, "rb") as f:
            self.wfile.write(f.read())

    def _serve_video(self):
        video_path = Path("logs/annotated_demo.mp4")
        if not video_path.exists():
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Annotated video not generated yet.")
            return

        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(video_path.stat().st_size))
        self.end_headers()
        with open(video_path, "rb") as f:
            self.wfile.write(f.read())


def run_server(port: int = 8000):
    cfg = load_config("config.json")
    init_db(cfg)

    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, DashboardRequestHandler)
    logger.info(f"SentinelTrack Dashboard running at http://127.0.0.1:{port}/")
    print(f"\n================ SentinelTrack Web Dashboard ================")
    print(f"  Live UI & API: http://127.0.0.1:{port}/")
    print(f"  Stats API:     http://127.0.0.1:{port}/api/stats")
    print(f"  Events API:    http://127.0.0.1:{port}/api/events")
    print(f"=============================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Dashboard server shutting down.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentinelTrack Live Web Dashboard")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()
    run_server(port=args.port)
