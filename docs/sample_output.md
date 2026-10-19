# SentinelTrack — Sample Output Report

Generated: 2026-10-17T05:25:59.613923+00:00

## Database Statistics

| Metric | Count |
|--------|-------|
| Unique Identities | 1 |
| Total Sessions | 7 |
| Entry Events | 6 |
| Exit Events | 6 |
| Registration Events | 0 |

## Evidence Images

- Entry images saved: 6
- Exit images saved: 3

Sample entry images:
  - `logs\entries\2026-10-17\entry_a37439f2_06169f98_0000004_051437684528.jpg`
  - `logs\entries\2026-10-17\entry_a37439f2_161a522e_0000060_052022824945.jpg`
  - `logs\entries\2026-10-17\entry_a37439f2_4fa4b38f_0000060_051442334042.jpg`

## Last Run Summary

```json
{
  "collected_at": "2026-10-17T05:20:28.907939+00:00",
  "elapsed_seconds": 12.99,
  "started_at": "2026-10-17T05:20:15.913679+00:00",
  "frames_processed": 110,
  "frames_dropped": 0,
  "detections_total": 40,
  "registrations": 0,
  "recognitions": 78,
  "entries": 2,
  "exits": 1,
  "errors": 0,
  "reconnects": 0,
  "fps_estimate": 8.47,
  "drop_rate_pct": 0.0,
  "detection_latency": {
    "mean_ms": 0.105,
    "max_ms": 2.464,
    "count": 55
  },
  "recognition_latency": {
    "mean_ms": 0.053,
    "max_ms": 0.167,
    "count": 110
  },
  "db_write_latency": {
    "mean_ms": null,
    "max_ms": null,
    "count": 0
  },
  "total_frame_latency": {
    "mean_ms": 107.624,
    "max_ms": 2465.257,
    "count": 110
  },
  "unique_identities": 1,
  "total_sessions": 7,
  "duration_seconds": 12.98
}
```

## Recent events.log (last 20 lines)

```
2026-10-17T10:50:27 | INFO     | sentineltrack.main | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
2026-10-17T10:50:27 | INFO     | sentineltrack.identity | [RECOGNITION] face_id=a37439f2-c7f3-4e3d-99f1-fa4900f084e2 similarity=1.1348 frame=101
2026-10-17T10:50:27 | INFO     | sentineltrack.identity | [RECOGNITION] face_id=a37439f2-c7f3-4e3d-99f1-fa4900f084e2 similarity=0.7714 frame=102
2026-10-17T10:50:28 | INFO     | sentineltrack.identity | [RECOGNITION] face_id=a37439f2-c7f3-4e3d-99f1-fa4900f084e2 similarity=1.0035 frame=103
2026-10-17T10:50:28 | INFO     | sentineltrack.identity | [RECOGNITION] face_id=a37439f2-c7f3-4e3d-99f1-fa4900f084e2 similarity=0.9289 frame=104
2026-10-17T10:50:28 | INFO     | sentineltrack.identity | [RECOGNITION] face_id=a37439f2-c7f3-4e3d-99f1-fa4900f084e2 similarity=1.0957 frame=105
2026-10-17T10:50:28 | INFO     | sentineltrack.identity | [RECOGNITION] face_id=a37439f2-c7f3-4e3d-99f1-fa4900f084e2 similarity=1.2154 frame=106
2026-10-17T10:50:28 | INFO     | sentineltrack.identity | [RECOGNITION] face_id=a37439f2-c7f3-4e3d-99f1-fa4900f084e2 similarity=1.0412 frame=107
2026-10-17T10:50:28 | INFO     | sentineltrack.identity | [RECOGNITION] face_id=a37439f2-c7f3-4e3d-99f1-fa4900f084e2 similarity=0.9210 frame=108
2026-10-17T10:50:28 | INFO     | sentineltrack.identity | [RECOGNITION] face_id=a37439f2-c7f3-4e3d-99f1-fa4900f084e2 similarity=0.9940 frame=109
2026-10-17T10:50:28 | INFO     | sentineltrack.identity | [RECOGNITION] face_id=a37439f2-c7f3-4e3d-99f1-fa4900f084e2 similarity=1.0301 frame=110
2026-10-17T10:50:28 | INFO     | sentineltrack.video_input | End of video file reached at frame 110.
2026-10-17T10:50:28 | INFO     | sentineltrack.main | Video source ended or frame unavailable.
2026-10-17T10:50:28 | INFO     | sentineltrack.main | Shutting down pipeline...
2026-10-17T10:50:28 | INFO     | sentineltrack.video_input | FileVideoSource released.
2026-10-17T10:50:28 | INFO     | sentineltrack.detector | YOLOFaceDetector unloaded.
2026-10-17T10:50:28 | INFO     | sentineltrack.recognizer | FaceRecognizer unloaded.
2026-10-17T10:50:28 | INFO     | sentineltrack.database | Database connection pool closed.
2026-10-17T10:50:28 | INFO     | sentineltrack | [SHUTDOWN] frames_processed=110 unique_identities=1 sessions=7
2026-10-17T10:50:28 | INFO     | sentineltrack.main | Run summary saved to logs\last_run_summary.json
```