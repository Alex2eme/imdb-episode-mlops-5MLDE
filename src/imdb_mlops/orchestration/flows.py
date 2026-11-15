"""Orchestration Prefect de la chaîne : surveillance de la source → données → entraînement.

Gestion des échecs :
- **erreurs transitoires** (réseau, dépendance momentanément indisponible) : relances
  automatiques avec délai croissant ;
- **erreurs définitives** (lot non conforme, entraînement refusé) : pas de relance, le flow
  échoue immédiatement avec un motif explicite ;
- tout échec de flow déclenche une alerte critique ; chaque étape est tracée dans la table
  `pipeline_steps` (lue par Grafana) en plus de l'historique Prefect.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from datetime import UTC, datetime

from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from prefect.runtime import flow_run

from imdb_mlops import db
from imdb_mlops.alerts import send_alert
from imdb_mlops.config import load_config

log = logging.getLogger(__name__)

# Exceptions pour lesquelles une relance n'a aucun sens : le résultat serait identique.
NON_RETRYABLE = ("DataQualityError", "TrainingBlockedError", "ValueError")


def _is_transient(task, task_run, state) -> bool:
    """Condition de relance Prefect : on ne relance que les erreurs transitoires."""
    try:
        state.result()
    except Exception as exc:
        retry = type(exc).__name__ not in NON_RETRYABLE
        logging.getLogger(__name__).warning(
            "Échec %s (%s) → %s",
            task.name,
            type(exc).__name__,
            "relance" if retry else "pas de relance (erreur définitive)",
        )
        return retry
    return False


@contextmanager
def track_step(flow_name: str, step: str):
    """Trace la durée et le statut d'une étape dans `pipeline_steps`."""
    started, t0 = datetime.now(UTC), time.perf_counter()
    run_id = str(flow_run.get_id() or "local")
    details: dict = {}
    status = "failed"
    try:
        yield details
        status = "success"
    except Exception as exc:
        details["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        raise
    finally:
        try:
            db.record_step(
                run_id=run_id,
                flow=flow_name,
                step=step,
                status=status,
                started_at=started,
                duration_s=time.perf_counter() - t0,
                details=details,
            )
        except Exception as exc:  # la base peut être la dépendance en panne
            log.warning("Étape %s non tracée : %s", step, exc)


def _alert_flow_failure(flow, flow_run_obj, state) -> None:
    """Hook Prefect : alerte critique sur tout échec ou crash de flow."""
    message = state.message or "échec sans message"
    try:
        state.result()
    except Exception as exc:
        message = f"{type(exc).__name__}: {str(exc)[:1500]}"
    send_alert(
        f"Flow « {flow.name} » en échec (run {flow_run_obj.name})",
        message,
        severity="critical",
        source="orchestration",
        details={
            "flow": flow.name,
            "flow_run_id": str(flow_run_obj.id),
            "parameters": flow_run_obj.parameters,
        },
    )


# --- Tâches ---------------------------------------------------------------------------


@task(
    name="ingest-and-validate",
    retries=2,
    retry_delay_seconds=[30, 120],
    retry_condition_fn=_is_transient,
    timeout_seconds=600,
    cache_policy=NO_CACHE,
)
def ingest_and_validate(ref: str | None = None, fault: str | None = None) -> dict:
    from imdb_mlops.data.pipeline import run_data_step

    logger = get_run_logger()
    with track_step("daily-pipeline", "ingest-and-validate") as details:
        _, manifest, report = run_data_step(load_config(), ref=ref, fault=fault)
        details |= {
            "sha": manifest.sha,
            "n_rows": manifest.n_rows,
            "n_warnings": len(report.warnings),
        }
    logger.info("Snapshot %s conforme (%d épisodes)", manifest.sha[:10], manifest.n_rows)
    return {"sha": manifest.sha, "n_rows": manifest.n_rows}


@task(
    name="train-and-promote",
    retries=2,
    retry_delay_seconds=[60, 180],
    retry_condition_fn=_is_transient,
    timeout_seconds=3600,
    cache_policy=NO_CACHE,
)
def train_and_promote(snapshot_sha: str, n_trials: int | None = None) -> dict:
    from imdb_mlops.training.train import train

    cfg = load_config()
    if n_trials is not None:
        cfg.training.n_trials = n_trials
    with track_step("daily-pipeline", "train-and-promote") as details:
        db.init_db()
        result = train(cfg, snapshot_sha=snapshot_sha, trigger="prefect")
        details |= {
            "run_id": result.run_id,
            "version": result.model_version,
            "rmse": result.rmse,
            "promoted": result.promoted,
        }
    get_run_logger().info(
        "v%s %s : %s",
        result.model_version,
        "promue" if result.promoted else "non promue",
        result.reason,
    )
    return {
        "run_id": result.run_id,
        "version": result.model_version,
        "rmse": result.rmse,
        "promoted": result.promoted,
        "reason": result.reason,
    }


@task(name="resolve-source-commit", retries=3, retry_delay_seconds=30, cache_policy=NO_CACHE)
def resolve_source_commit() -> str:
    from imdb_mlops.data.ingest import resolve_commit

    cfg = load_config()
    sha, _ = resolve_commit(cfg.data.source_repo, cfg.data.source_branch)
    return sha


# --- Flows ----------------------------------------------------------------------------


@flow(
    name="daily-pipeline",
    on_failure=[_alert_flow_failure],
    on_crashed=[_alert_flow_failure],
    timeout_seconds=5400,
)
def daily_pipeline(
    ref: str | None = None,
    fault: str | None = None,
    n_trials: int | None = None,
    train: bool = True,
) -> dict:
    """Ingestion + contrôles qualité, puis entraînement et décision de mise en service.

    Un lot non conforme interrompt le flow avant l'entraînement.
    """
    snapshot = ingest_and_validate(ref=ref, fault=fault)
    if not train:
        return {"snapshot": snapshot}
    return {"snapshot": snapshot, "model": train_and_promote(snapshot["sha"], n_trials)}


@flow(name="watch-source", on_failure=[_alert_flow_failure])
def watch_source(n_trials: int | None = None) -> dict:
    """Ingestion dynamique (pull) : détecte un nouveau commit du dépôt source et déclenche
    la chaîne seulement dans ce cas. Planifié à intervalle régulier."""
    logger = get_run_logger()
    sha = resolve_source_commit()
    known = db.get_snapshot(sha)
    if known is not None:
        logger.info("Commit %s déjà traité (%s) : rien à faire", sha[:10], known["quality_status"])
        return {"sha": sha, "triggered": False}
    logger.info("Nouveau commit source %s → déclenchement de la chaîne", sha[:10])
    send_alert(
        "Nouveau snapshot source détecté",
        f"Commit {sha[:10]} : lancement de la chaîne",
        severity="info",
        source="orchestration",
        details={"sha": sha},
    )
    result = daily_pipeline(ref=sha, n_trials=n_trials)
    return {"sha": sha, "triggered": True, "result": result}
