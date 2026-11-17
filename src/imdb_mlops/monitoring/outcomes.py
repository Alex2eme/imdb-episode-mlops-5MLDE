"""Évaluation en production : prédiction (jour J) → note réelle publiée ensuite (J+1).

Chaque prédiction servie par l'API est conservée avec le SHA du snapshot utilisé. Quand un
snapshot plus récent est validé, on y cherche la note réelle de l'épisode prédit :
- l'épisode existait déjà au jour J : on mesure l'erreur sur un épisode connu ;
- l'épisode est **nouveau** (absent au jour J) : c'est une vraie prédiction à l'aveugle,
  le cas d'usage visé.
Les prédictions d'épisodes pas encore publiés restent en attente (au plus
`pending_max_days`). Les métriques live sont agrégées, tracées dans MLflow, et une alerte
est émise si la RMSE dépasse le seuil (dérive).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import and_, select

from imdb_mlops import db
from imdb_mlops.alerts import send_alert
from imdb_mlops.config import ROOT_DIR, AppConfig

log = logging.getLogger(__name__)
KEY = ["series_id", "season", "episode"]


def _snapshot_frame(snap: dict) -> pd.DataFrame:
    path = ROOT_DIR / snap["parquet_path"]
    return pd.read_parquet(path, columns=[*KEY, "rating", "votes"])


def pending_predictions(actual: dict, cfg: AppConfig) -> pd.DataFrame:
    """Prédictions réussies, faites sur un snapshot plus ancien, pas encore évaluées."""
    p, s, o = db.predictions, db.data_snapshots, db.prediction_outcomes
    since = datetime.now(UTC) - timedelta(days=cfg.monitoring.pending_max_days)
    query = (
        select(
            p.c.id,
            p.c.created_at,
            p.c.series_id,
            p.c.season,
            p.c.episode,
            p.c.predicted_rating,
            p.c.model_version,
            p.c.data_sha,
        )
        .join(s, s.c.sha == p.c.data_sha)
        .outerjoin(o, o.c.prediction_id == p.c.id)
        .where(
            and_(
                p.c.status == "ok",
                p.c.source != "loadtest",  # le trafic de charge n'est pas de la production
                p.c.created_at >= since,
                p.c.data_sha != actual["sha"],
                s.c.commit_date < actual["commit_date"],
                o.c.prediction_id.is_(None),
            )
        )
    )
    with db.get_engine().connect() as conn:
        return pd.DataFrame(conn.execute(query).mappings().all())


def match_outcomes(
    pending: pd.DataFrame, actual_df: pd.DataFrame, known_keys: dict[str, set]
) -> pd.DataFrame:
    """Associe chaque prédiction à la note réelle (fonction pure, testée)."""
    if pending.empty:
        return pd.DataFrame()
    merged = pending.merge(actual_df, on=KEY, how="inner")
    if merged.empty:
        return merged
    merged["error"] = merged["predicted_rating"] - merged["rating"]
    merged["abs_error"] = merged["error"].abs()
    merged["is_new_episode"] = [
        (sid, se, ep) not in known_keys.get(sha, set())
        for sid, se, ep, sha in zip(
            merged["series_id"],
            merged["season"],
            merged["episode"],
            merged["data_sha"],
            strict=True,
        )
    ]
    return merged


def live_metrics(window_days: int) -> pd.DataFrame:
    """MAE/RMSE live sur la fenêtre, par version du modèle et type d'épisode."""
    o = db.prediction_outcomes
    since = datetime.now(UTC) - timedelta(days=window_days)
    with db.get_engine().connect() as conn:
        df = pd.DataFrame(
            conn.execute(
                select(o.c.model_version, o.c.is_new_episode, o.c.error).where(
                    o.c.evaluated_at >= since
                )
            )
            .mappings()
            .all()
        )
    if df.empty:
        return df
    rows = []
    for (version, new), g in [
        *df.groupby(["model_version", "is_new_episode"]),
        (("all", "all"), df),
    ]:
        rows.append(
            {
                "model_version": version,
                "is_new_episode": new,
                "n": len(g),
                "mae": float(g["error"].abs().mean()),
                "rmse": float(np.sqrt((g["error"] ** 2).mean())),
                "bias": float(g["error"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _log_to_mlflow(cfg: AppConfig, actual_sha: str, n_new: int, metrics: pd.DataFrame) -> None:
    try:
        import mlflow

        mlflow.set_experiment(cfg.monitoring.experiment_name)
        with mlflow.start_run(run_name=f"live-eval-{actual_sha[:8]}"):
            mlflow.set_tags({"actual_data_sha": actual_sha, "type": "production_evaluation"})
            mlflow.log_metric("n_evaluated", n_new)
            for r in metrics.itertuples():
                scope = (
                    "all"
                    if r.model_version == "all"
                    else (f"v{r.model_version}_{'new' if r.is_new_episode else 'known'}")
                )
                mlflow.log_metrics(
                    {
                        f"live_rmse_{scope}": r.rmse,
                        f"live_mae_{scope}": r.mae,
                        f"live_n_{scope}": r.n,
                        f"live_bias_{scope}": r.bias,
                    }
                )
            mlflow.log_table(metrics, "live_metrics.json")
    except Exception as exc:
        log.warning("Métriques live non tracées dans MLflow : %s", exc)


def evaluate_predictions(cfg: AppConfig, actual_sha: str | None = None) -> dict:
    db.init_db()
    actual = db.get_snapshot(actual_sha) if actual_sha else db.last_valid_snapshot()
    if actual is None or actual["quality_status"] != "passed":
        raise ValueError(f"Snapshot de référence {actual_sha or '(dernier)'} invalide")

    pending = pending_predictions(actual, cfg)
    if pending.empty:
        log.info("Aucune prédiction en attente d'évaluation")
        return {"actual_sha": actual["sha"], "n_pending": 0, "n_evaluated": 0}

    actual_df = _snapshot_frame(actual)
    known_keys = {}
    for sha in pending["data_sha"].unique():
        snap = db.get_snapshot(sha)
        if snap and (ROOT_DIR / snap["parquet_path"]).exists():
            known = _snapshot_frame(snap)
            known_keys[sha] = set(
                zip(known["series_id"], known["season"], known["episode"], strict=True)
            )
    matched = match_outcomes(pending, actual_df, known_keys)

    now = datetime.now(UTC)
    if not matched.empty:
        rows = [
            {
                "prediction_id": r.id,
                "predicted_at": r.created_at,
                "evaluated_at": now,
                "series_id": r.series_id,
                "season": int(r.season),
                "episode": int(r.episode),
                "predicted_rating": float(r.predicted_rating),
                "actual_rating": float(r.rating),
                "actual_votes": int(r.votes) if pd.notna(r.votes) else None,
                "error": float(r.error),
                "abs_error": float(r.abs_error),
                "is_new_episode": bool(r.is_new_episode),
                "model_version": r.model_version,
                "prediction_data_sha": r.data_sha,
                "actual_data_sha": actual["sha"],
            }
            for r in matched.itertuples()
        ]
        with db.get_engine().begin() as conn:
            conn.execute(db.prediction_outcomes.insert(), rows)

    metrics = live_metrics(cfg.monitoring.window_days)
    summary = {
        "actual_sha": actual["sha"],
        "n_pending": len(pending),
        "n_evaluated": len(matched),
        "n_new_episodes": int(matched["is_new_episode"].sum()) if not matched.empty else 0,
        "n_still_pending": len(pending) - len(matched),
    }
    if not metrics.empty:
        overall = metrics[metrics["model_version"] == "all"].iloc[0]
        summary |= {
            "live_n": int(overall["n"]),
            "live_rmse": round(overall["rmse"], 4),
            "live_mae": round(overall["mae"], 4),
            "live_bias": round(overall["bias"], 4),
        }
        _log_to_mlflow(cfg, actual["sha"], len(matched), metrics)
        m = cfg.monitoring
        if overall["n"] >= m.min_samples and overall["rmse"] > m.max_live_rmse:
            send_alert(
                "Dérive du modèle en production",
                f"RMSE live {overall['rmse']:.3f} > seuil {m.max_live_rmse} "
                f"sur {int(overall['n'])} prédictions ({m.window_days} j) : envisager un "
                "réentraînement ou un rollback.",
                severity="critical",
                source="monitoring",
                details=summary,
            )
    log.info("Évaluation J+1 : %s", summary)
    return summary
