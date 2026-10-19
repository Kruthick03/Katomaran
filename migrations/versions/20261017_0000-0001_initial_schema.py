"""Initial schema — SentinelTrack

Creates all tables:
  - cameras (video sources)
  - faces (persistent identities)
  - visit_sessions (per-visit records)
  - events (audit log with exactly-once enforcement)

Revision ID: 0001_initial_schema
Revises: (none)
Create Date: 2026-10-17
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Enumerations ──────────────────────────────────────────────────────────
    sa.Enum(
        "file", "rtsp", name="source_type_enum"
    ).create(op.get_bind(), checkfirst=True)

    sa.Enum(
        "pending", "confirmed", "flagged", name="registration_status_enum"
    ).create(op.get_bind(), checkfirst=True)

    sa.Enum(
        "active", "exit_pending", "exited", name="session_status_enum"
    ).create(op.get_bind(), checkfirst=True)

    sa.Enum(
        "entry", "exit", "registration", "recognition", "ambiguous", "error",
        name="event_type_enum"
    ).create(op.get_bind(), checkfirst=True)

    # ── cameras ───────────────────────────────────────────────────────────────
    op.create_table(
        "cameras",
        sa.Column("source_id", sa.String(64), primary_key=True),
        sa.Column(
            "source_type",
            postgresql.ENUM("file", "rtsp", name="source_type_enum", create_type=False),
            nullable=False,
        ),
        sa.Column("source_name", sa.String(256), nullable=False),
        sa.Column("config_meta", postgresql.JSONB(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # ── faces ─────────────────────────────────────────────────────────────────
    op.create_table(
        "faces",
        sa.Column("face_id", sa.String(36), primary_key=True),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("embedding", sa.LargeBinary(), nullable=True),
        sa.Column("embedding_dimension", sa.Integer(), nullable=True),
        sa.Column("recognition_model", sa.String(128), nullable=False),
        sa.Column(
            "registration_status",
            postgresql.ENUM("pending", "confirmed", "flagged", name="registration_status_enum", create_type=False),
            nullable=False,
            server_default="confirmed",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_faces_last_seen_at", "faces", ["last_seen_at"])
    op.create_index("ix_faces_first_seen_at", "faces", ["first_seen_at"])

    # ── visit_sessions ────────────────────────────────────────────────────────
    op.create_table(
        "visit_sessions",
        sa.Column("session_id", sa.String(36), primary_key=True),
        sa.Column(
            "face_id",
            sa.String(36),
            sa.ForeignKey("faces.face_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "source_id",
            sa.String(64),
            sa.ForeignKey("cameras.source_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("entry_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("exit_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "status",
            postgresql.ENUM("active", "exit_pending", "exited", name="session_status_enum", create_type=False),
            nullable=False,
            server_default="active",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "(exit_timestamp IS NULL) OR (entry_timestamp IS NOT NULL AND exit_timestamp >= entry_timestamp)",
            name="ck_session_timestamps_order",
        ),
    )
    op.create_index("ix_visit_sessions_face_id", "visit_sessions", ["face_id"])
    op.create_index("ix_visit_sessions_source_id", "visit_sessions", ["source_id"])
    op.create_index("ix_visit_sessions_status", "visit_sessions", ["status"])
    op.create_index(
        "ix_visit_sessions_entry_timestamp", "visit_sessions", ["entry_timestamp"]
    )

    # ── events ────────────────────────────────────────────────────────────────
    op.create_table(
        "events",
        sa.Column("event_id", sa.String(36), primary_key=True),
        sa.Column(
            "face_id",
            sa.String(36),
            sa.ForeignKey("faces.face_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "session_id",
            sa.String(36),
            sa.ForeignKey("visit_sessions.session_id", ondelete="RESTRICT"),
            nullable=True,
        ),
        sa.Column(
            "source_id",
            sa.String(64),
            sa.ForeignKey("cameras.source_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "event_type",
            postgresql.ENUM(
                "entry", "exit", "registration", "recognition", "ambiguous", "error",
                name="event_type_enum",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("event_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("evidence_image_path", sa.Text(), nullable=True),
        sa.Column("frame_number", sa.Integer(), nullable=True),
        sa.Column("idempotency_key", sa.String(64), nullable=False, unique=True),
        sa.Column("meta", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_events_face_id", "events", ["face_id"])
    op.create_index("ix_events_session_id", "events", ["session_id"])
    op.create_index("ix_events_event_type", "events", ["event_type"])
    op.create_index("ix_events_event_timestamp", "events", ["event_timestamp"])

    # Partial unique indexes for exactly-once entry/exit per session
    op.execute(
        "CREATE UNIQUE INDEX uq_events_entry_per_session "
        "ON events (session_id) WHERE event_type = 'entry'"
    )
    op.execute(
        "CREATE UNIQUE INDEX uq_events_exit_per_session "
        "ON events (session_id) WHERE event_type = 'exit'"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_events_exit_per_session")
    op.execute("DROP INDEX IF EXISTS uq_events_entry_per_session")
    op.drop_table("events")
    op.drop_table("visit_sessions")
    op.drop_table("faces")
    op.drop_table("cameras")

    # Drop enums
    for enum_name in [
        "event_type_enum", "session_status_enum",
        "registration_status_enum", "source_type_enum"
    ]:
        op.execute(f"DROP TYPE IF EXISTS {enum_name}")
