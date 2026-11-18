"""Chargement du modèle champion et de l'historique des séries, avec rechargement à chaud.

Le modèle servi est toujours celui portant l'alias `champion` dans le registre MLflow :
une promotion ou un rollback est pris en compte sans redémarrer l'API (vérification
périodique). L'historique provient du dernier snapshot validé (le plus récent).
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from imdb_mlops import db
from imdb_mlops.config import ROOT_DIR, AppConfig
from imdb_mlops.features import FEATURES, SeriesHistory, build_histories, serving_features

log = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Aucun modèle chargé (registre injoignable au démarrage, pas de champion...)."""


@dataclass(frozen=True)
class Loaded:
    model: object
    name: str
    version: str
    run_id: str
    rmse: float | None
    data_sha: str
    history: dict[str, SeriesHistory]
    loaded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


# Snapshot servi épinglé (rejeu, démonstration J+1) ; vide = dernier snapshot validé.
SERVE_SNAPSHOT_SHA = os.environ.get("SERVE_SNAPSHOT_SHA", "")


def _serving_snapshot() -> dict | None:
    if SERVE_SNAPSHOT_SHA:
        snap = db.get_snapshot(SERVE_SNAPSHOT_SHA)
        if snap is None or snap["quality_status"] != "passed":
            raise ModelUnavailableError(f"Snapshot épinglé {SERVE_SNAPSHOT_SHA[:10]} non validé")
        return snap
    return db.last_valid_snapshot()


def _load_history(snap: dict | None) -> tuple[str, dict[str, SeriesHistory]]:
    if snap is None:
        raise ModelUnavailableError("Aucun snapshot validé pour l'historique des séries")
    df = pd.read_parquet(ROOT_DIR / snap["parquet_path"])
    cols = ["series_id", "series_votes", "season", "episode", "rating", "votes"]
    # Pré-indexation par série (tableaux numpy triés) : features calculées en microsecondes.
    history = build_histories(df[cols])
    return snap["sha"], history


def _predict_array(model, X: np.ndarray) -> np.ndarray:
    """Appel direct du booster LightGBM sur un tableau : évite la construction d'un
    DataFrame et les contrôles du wrapper scikit-learn (coûteux pour une seule ligne)."""
    booster = getattr(model, "booster_", None)
    if booster is not None:
        # num_threads=1 : par défaut LightGBM lance un thread OpenMP par cœur *visible* (12),
        # alors que le conteneur n'a droit qu'à 1-2 CPU ; l'attente active de ces threads
        # épuise le quota et Linux bride tout le conteneur (mesuré : p99 ~5 s). La
        # parallélisation se fait par les workers uvicorn, pas à l'intérieur d'une requête.
        return booster.predict(
            X, num_iteration=getattr(model, "best_iteration_", None) or None, num_threads=1
        )
    return model.predict(pd.DataFrame(X, columns=FEATURES))


def check_features(tags: dict) -> None:
    """Refuse une version entraînée avec d'autres features que celles calculées par ce code
    (rollback vers une version antérieure à une évolution des features). Les versions sans
    tag `features` (antérieures à son introduction) sont acceptées."""
    declared = tags.get("features")
    if declared and declared.split(",") != FEATURES:
        raise ModelUnavailableError(
            f"Features incompatibles avec ce code : modèle {declared!r}, attendu "
            f"{','.join(FEATURES)!r}"
        )


class ModelStore:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._loaded: Loaded | None = None
        self._lock = threading.Lock()
        self.last_error: str | None = None

    @property
    def current(self) -> Loaded:
        if self._loaded is None:
            raise ModelUnavailableError(self.last_error or "Modèle non chargé")
        return self._loaded

    @property
    def ready(self) -> bool:
        return self._loaded is not None

    def set(self, loaded: Loaded) -> None:
        """Remplacement atomique (utilisé aussi par les tests)."""
        self._loaded = loaded

    def refresh(self, force: bool = False) -> bool:
        """Recharge si le champion ou le dernier snapshot validé a changé.

        Retourne True si un rechargement a eu lieu. N'interrompt jamais le service : en cas
        d'erreur, l'ancien modèle reste servi.
        """
        import mlflow
        from mlflow import MlflowClient

        name = self.cfg.training.registered_model_name
        with self._lock:
            try:
                champ = MlflowClient().get_model_version_by_alias(name, "champion")
                latest = _serving_snapshot()
                check_features(champ.tags)
                cur = self._loaded
                same_model = cur is not None and cur.version == str(champ.version)
                same_data = cur is not None and latest is not None and cur.data_sha == latest["sha"]
                if same_model and same_data and not force:
                    return False
                model = (
                    cur.model
                    if same_model and not force
                    else mlflow.lightgbm.load_model(f"models:/{name}/{champ.version}")
                )
                if same_data and not force:
                    sha, history = cur.data_sha, cur.history
                else:
                    sha, history = _load_history(latest)
                rmse = champ.tags.get("rmse")
                self._loaded = Loaded(
                    model=model,
                    name=name,
                    version=str(champ.version),
                    run_id=champ.run_id,
                    rmse=float(rmse) if rmse else None,
                    data_sha=sha,
                    history=history,
                )
                self.last_error = None
                log.info(
                    "Modèle servi : %s v%s (données %s, %d séries)",
                    name,
                    champ.version,
                    sha[:10],
                    len(history),
                )
                return True
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.error(
                    "Rechargement impossible, modèle précédent conservé : %s", self.last_error
                )
                return False

    def predict(self, refs: list[tuple[str, int, int]]) -> tuple[np.ndarray, list[str | None]]:
        """Prédit pour une liste (series_id, season, episode).

        Retourne les prédictions (NaN si erreur) et, par élément, le code d'erreur éventuel.
        Les features de tout le lot sont assemblées puis prédites en un seul appel.
        """
        loaded = self.current
        rows, errors, ok_idx = [], [None] * len(refs), []
        for i, (sid, season, episode) in enumerate(refs):
            history = loaded.history.get(sid)
            if history is None:
                errors[i] = "unknown_series"
                continue
            rows.append(serving_features(history, season, episode))
            ok_idx.append(i)
        preds = np.full(len(refs), np.nan)
        if rows:
            X = np.vstack(rows)
            preds[ok_idx] = np.clip(_predict_array(loaded.model, X), 1.0, 10.0)
        return preds, errors
