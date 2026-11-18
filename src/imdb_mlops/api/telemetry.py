"""Télémétrie de l'API : métriques Prometheus et écriture asynchrone des traces en base.

Les traces (requêtes, prédictions) sont mises en file et insérées par lots par un thread
dédié : l'écriture en base n'ajoute pas de latence aux réponses, et une base indisponible
ne bloque pas le service (les traces sont alors comptées comme perdues).
"""

from __future__ import annotations

import logging
import os
import queue
import threading

from prometheus_client import Counter, Gauge, Histogram

log = logging.getLogger(__name__)

_MULTIPROC = bool(os.environ.get("PROMETHEUS_MULTIPROC_DIR"))

REQUESTS = Counter("api_requests_total", "Requêtes HTTP", ["endpoint", "method", "status"])
LATENCY = Histogram(
    "api_request_duration_seconds",
    "Durée de traitement des requêtes",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
PREDICTIONS = Counter("predictions_total", "Prédictions servies", ["model_version", "cached"])
PREDICTION_ERRORS = Counter("prediction_errors_total", "Erreurs de prédiction", ["error"])
PREDICTED_RATING = Histogram(
    "predicted_rating", "Distribution des notes prédites", buckets=tuple(range(1, 11))
)
CACHE_REQUESTS = Counter("cache_requests_total", "Accès au cache", ["result"])
FEEDBACK = Counter("feedback_total", "Feedbacks reçus")
MODEL_INFO = Gauge(
    "model_info",
    "Modèle servi (valeur 1)",
    ["version", "data_sha"],
    **({"multiprocess_mode": "liveall"} if _MULTIPROC else {}),
)
MODEL_RELOADS = Counter("model_reloads_total", "Rechargements du modèle", ["result"])
DB_DROPPED = Counter("db_traces_dropped_total", "Traces non écrites en base", ["table"])
DB_QUEUE = Gauge(
    "db_traces_queue_size",
    "Traces en attente d'écriture",
    **({"multiprocess_mode": "livesum"} if _MULTIPROC else {}),
)


_model_labels: dict[str, str] | None = None


def set_model_info(version: str, data_sha: str) -> None:
    """Marque la version servie à 1 et la précédente à 0.

    `Gauge.clear()` n'est pas implémenté en mode multi-processus de prometheus_client :
    l'ancien label restait à 1 et Grafana affichait deux versions « servies ». On remet
    donc explicitement l'ancien label à 0 (les requêtes filtrent `model_info == 1`).
    """
    global _model_labels  # noqa: PLW0603 (état du processus worker)
    labels = {"version": version, "data_sha": data_sha[:10]}
    if _model_labels is not None and _model_labels != labels:
        MODEL_INFO.labels(**_model_labels).set(0)
    MODEL_INFO.labels(**labels).set(1)
    _model_labels = labels


class TraceWriter:
    """File d'attente bornée + thread d'insertion par lots."""

    def __init__(
        self,
        enabled: bool = True,
        flush_every_s: float = 1.0,
        batch_size: int = 500,
        max_queue: int = 20_000,
    ):
        self.enabled = enabled
        self.flush_every_s = flush_every_s
        self.batch_size = batch_size
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.enabled and self._thread is None:
            self._thread = threading.Thread(target=self._run, name="trace-writer", daemon=True)
            self._thread.start()

    def stop(self, max_batches: int = 20) -> None:
        """Arrêt : vide toute la file (et non un seul lot) ; ce qui reste au-delà de
        `max_batches` lots est compté comme perdu au lieu de disparaître silencieusement."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        for _ in range(max_batches):
            if self._queue.empty():
                return
            self._flush()
        lost: dict[str, int] = {}
        while True:
            try:
                table, _row = self._queue.get_nowait()
            except queue.Empty:
                break
            lost[table] = lost.get(table, 0) + 1
        for table, n in lost.items():
            DB_DROPPED.labels(table=table).inc(n)
            log.warning("%d traces %s perdues à l'arrêt", n, table)

    def put(self, table: str, row: dict) -> None:
        if not self.enabled:
            return
        try:
            self._queue.put_nowait((table, row))
        except queue.Full:
            DB_DROPPED.labels(table=table).inc()

    def _run(self) -> None:
        while not self._stop.wait(self.flush_every_s):
            self._flush()

    def _flush(self) -> None:
        from imdb_mlops import db

        batch: dict[str, list[dict]] = {}
        try:
            while sum(len(v) for v in batch.values()) < self.batch_size * 10:
                table, row = self._queue.get_nowait()
                batch.setdefault(table, []).append(row)
        except queue.Empty:
            pass
        DB_QUEUE.set(self._queue.qsize())
        if not batch:
            return
        tables = {"predictions": db.predictions, "api_requests": db.api_requests}
        for name, rows in batch.items():
            try:
                with db.get_engine().begin() as conn:
                    conn.execute(tables[name].insert(), rows)
            except Exception as exc:
                DB_DROPPED.labels(table=name).inc(len(rows))
                log.warning(
                    "%d traces %s perdues (base indisponible ?) : %s",
                    len(rows),
                    name,
                    str(exc)[:200],
                )
