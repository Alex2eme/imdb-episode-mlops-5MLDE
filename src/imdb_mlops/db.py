"""Persistance applicative (base PostgreSQL `app`) : traces lues par Grafana."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    select,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from imdb_mlops.config import get_settings

metadata = MetaData()

data_snapshots = Table(
    "data_snapshots",
    metadata,
    # SHA du commit source ; suffixé « -fault-<type> » pour les lots de démonstration.
    Column("sha", String(80), primary_key=True),
    Column("source_repo", String(200), nullable=False),
    Column("commit_date", DateTime(timezone=True)),
    Column("built_at", DateTime(timezone=True)),
    Column("ingested_at", DateTime(timezone=True), nullable=False),
    Column("n_rows", Integer, nullable=False),
    Column("n_series", Integer, nullable=False),
    Column("n_orphan_series", Integer),
    Column("parquet_path", String(500)),
    Column("parquet_sha256", String(64)),
    Column("quality_status", String(16)),  # passed | failed
    Column("n_checks", Integer),
    Column("n_failed", Integer),
    Column("n_warnings", Integer),
    Column("quality_report", JSON),
    Column("validated_at", DateTime(timezone=True)),
)

alerts = Table(
    "alerts",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("severity", String(16), nullable=False),  # info | warning | critical
    Column("source", String(100), nullable=False),
    Column("title", String(300), nullable=False),
    Column("message", Text),
    Column("details", JSON),
    Column("delivered", Boolean, default=False),
)

pipeline_steps = Table(
    "pipeline_steps",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), index=True),
    Column("flow", String(100), nullable=False),
    Column("step", String(100), nullable=False),
    Column("status", String(16), nullable=False),  # success | failed
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("duration_s", Float),
    Column("details", JSON),
)


model_events = Table(
    "model_events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("event", String(32), nullable=False),  # registered|promoted|rejected|rollback
    Column("model_name", String(200), nullable=False),
    Column("version", String(16), nullable=False),
    Column("run_id", String(64)),
    Column("data_sha", String(80)),
    Column("rmse", Float),
    Column("champion_rmse", Float),
    Column("baseline_rmse", Float),
    Column("reason", Text),
)


# --- Traces d'utilisation de l'API ---------------------------------------------------

predictions = Table(
    "predictions",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("request_id", String(64)),
    Column("endpoint", String(16), nullable=False),  # single | batch
    Column("series_id", String(16), nullable=False, index=True),
    Column("season", Integer, nullable=False),
    Column("episode", Integer, nullable=False),
    Column("predicted_rating", Float),
    Column("model_version", String(16)),
    Column("model_run_id", String(64)),
    Column("data_sha", String(80)),
    Column("latency_ms", Float),
    Column("cached", Boolean),
    Column("status", String(16), nullable=False),  # ok | error
    Column("error", String(64)),
)

feedback = Table(
    "feedback",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("prediction_id", String(64), nullable=False, index=True),
    Column("rating", Float, nullable=False),
    Column("comment", Text),
)

api_requests = Table(
    "api_requests",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("request_id", String(64)),
    Column("method", String(8), nullable=False),
    Column("path", String(200), nullable=False),
    Column("status_code", Integer, nullable=False),
    Column("duration_ms", Float, nullable=False),
    Column("error", String(64)),
    Column("model_version", String(16)),
    Column("worker_pid", Integer),
)


# Comparaison prédiction → note réelle publiée dans un snapshot ultérieur (J+1).
prediction_outcomes = Table(
    "prediction_outcomes",
    metadata,
    Column("prediction_id", String(36), primary_key=True),
    Column("predicted_at", DateTime(timezone=True), nullable=False),
    Column("evaluated_at", DateTime(timezone=True), nullable=False, index=True),
    Column("series_id", String(16), nullable=False),
    Column("season", Integer, nullable=False),
    Column("episode", Integer, nullable=False),
    Column("predicted_rating", Float, nullable=False),
    Column("actual_rating", Float, nullable=False),
    Column("actual_votes", Integer),
    Column("error", Float, nullable=False),  # prédit - réel
    Column("abs_error", Float, nullable=False),
    # Épisode absent du snapshot ayant servi à la prédiction : vraie prédiction « à l'aveugle ».
    Column("is_new_episode", Boolean, nullable=False),
    Column("model_version", String(16)),
    Column("prediction_data_sha", String(80)),
    Column("actual_data_sha", String(80)),
)


@lru_cache
def get_engine() -> Engine:
    return create_engine(
        get_settings().database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 5},
    )


def init_db() -> None:
    metadata.create_all(get_engine())


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def upsert_snapshot(manifest: dict, report: dict | None = None) -> None:
    row = {
        **{
            k: manifest[k]
            for k in (
                "sha",
                "source_repo",
                "n_rows",
                "n_series",
                "n_orphan_series",
                "parquet_path",
                "parquet_sha256",
            )
        },
        "commit_date": _parse_dt(manifest.get("commit_date")),
        "built_at": _parse_dt(manifest.get("built_at")),
        "ingested_at": _parse_dt(manifest["ingested_at"]),
    }
    if report is not None:
        row |= {
            "quality_status": "passed" if report["success"] else "failed",
            "n_checks": report["n_checks"],
            "n_failed": report["n_failed"],
            "n_warnings": report["n_warnings"],
            "quality_report": report,
            "validated_at": _parse_dt(report["validated_at"]),
        }
    stmt = pg_insert(data_snapshots).values(**row)
    stmt = stmt.on_conflict_do_update(
        index_elements=["sha"], set_={k: stmt.excluded[k] for k in row if k != "sha"}
    )
    with get_engine().begin() as conn:
        conn.execute(stmt)


def last_valid_snapshot(exclude_sha: str | None = None) -> dict | None:
    """Dernier snapshot ayant passé les contrôles (référence pour les comparaisons)."""
    q = (
        select(data_snapshots)
        .where(data_snapshots.c.quality_status == "passed")
        .order_by(data_snapshots.c.validated_at.desc())
        .limit(1)
    )
    if exclude_sha:
        q = q.where(data_snapshots.c.sha != exclude_sha)
    with get_engine().connect() as conn:
        row = conn.execute(q).mappings().first()
    return dict(row) if row else None


def record_alert(
    *, severity: str, source: str, title: str, message: str, details: dict, delivered: bool
) -> None:
    with get_engine().begin() as conn:
        conn.execute(
            alerts.insert().values(
                created_at=datetime.now(UTC),
                severity=severity,
                source=source,
                title=title,
                message=message,
                details=details,
                delivered=delivered,
            )
        )


def record_step(
    *,
    run_id: str,
    flow: str,
    step: str,
    status: str,
    started_at: datetime,
    duration_s: float,
    details: dict | None = None,
) -> None:
    with get_engine().begin() as conn:
        conn.execute(
            pipeline_steps.insert().values(
                run_id=run_id,
                flow=flow,
                step=step,
                status=status,
                started_at=started_at,
                duration_s=duration_s,
                details=details or {},
            )
        )


def get_snapshot(sha: str) -> dict | None:
    with get_engine().connect() as conn:
        row = conn.execute(select(data_snapshots).where(data_snapshots.c.sha == sha)).mappings()
        row = row.first()
    return dict(row) if row else None


def record_model_event(**values) -> None:
    with get_engine().begin() as conn:
        conn.execute(model_events.insert().values(created_at=datetime.now(UTC), **values))
