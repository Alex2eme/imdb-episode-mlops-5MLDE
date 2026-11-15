"""Cache Redis des prédictions, tolérant aux pannes.

- La clé inclut la version du modèle et le SHA des données : une promotion, un rollback ou
  un nouveau snapshot invalident naturellement le cache, sans purge.
- Redis est une optimisation, pas une dépendance critique : timeouts courts et coupe-circuit.
  Si Redis tombe, l'API continue de répondre (sans cache) au lieu d'échouer.
- Le rétablissement est vérifié par un thread de fond, jamais par une requête utilisateur :
  quand le conteneur Redis est arrêté, la seule résolution DNS de son nom peut bloquer
  plusieurs secondes (constaté : 4 à 30 s), ce que les timeouts de socket ne couvrent pas.
"""

from __future__ import annotations

import logging
import threading
import time

import redis

from imdb_mlops.api import telemetry

log = logging.getLogger(__name__)


class PredictionCache:
    def __init__(self, url: str | None, ttl_s: int = 86400, cooldown_s: float = 2.0, client=None):
        self.ttl_s = ttl_s
        self.cooldown_s = cooldown_s
        self._healthy = True  # coupe-circuit : False = Redis contourné
        self._client = client
        if self._client is None and url:
            self._client = redis.Redis.from_url(
                url, socket_timeout=0.05, socket_connect_timeout=0.05, decode_responses=True
            )
            threading.Thread(target=self._watch, name="redis-watch", daemon=True).start()

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @staticmethod
    def key(version: str, data_sha: str, sid: str, season: int, episode: int) -> str:
        return f"pred:v{version}:{data_sha[:12]}:{sid}:{season}:{episode}"

    def _available(self) -> bool:
        return self._client is not None and self._healthy

    def _trip(self, exc: Exception) -> None:
        telemetry.CACHE_REQUESTS.labels(result="error").inc()
        if self._healthy:
            self._healthy = False
            log.warning("Redis indisponible (%s) : cache contourné jusqu'à rétablissement", exc)

    def _watch(self) -> None:
        """Sonde permanente : détecte la panne avant les requêtes et le rétablissement."""
        while True:
            time.sleep(self.cooldown_s)
            try:
                ok = bool(self._client.ping())
            except redis.RedisError:
                ok = False
            if ok and not self._healthy:
                log.info("Redis rétabli : cache réactivé")
            elif not ok and self._healthy:
                log.warning("Redis injoignable (sonde) : cache contourné")
            self._healthy = ok

    def get_many(self, keys: list[str]) -> list[float | None]:
        if not keys or not self._available():
            if self._client is not None:
                telemetry.CACHE_REQUESTS.labels(result="bypass").inc(len(keys))
            return [None] * len(keys)
        try:
            values = self._client.mget(keys)
        except redis.RedisError as exc:
            self._trip(exc)
            return [None] * len(keys)
        out = [float(v) if v is not None else None for v in values]
        hits = sum(v is not None for v in out)
        telemetry.CACHE_REQUESTS.labels(result="hit").inc(hits)
        telemetry.CACHE_REQUESTS.labels(result="miss").inc(len(out) - hits)
        return out

    def set_many(self, items: dict[str, float]) -> None:
        if not items or not self._available():
            return
        try:
            pipe = self._client.pipeline(transaction=False)
            for k, v in items.items():
                pipe.setex(k, self.ttl_s, f"{v:.6f}")
            pipe.execute()
        except redis.RedisError as exc:
            self._trip(exc)

    def ping(self) -> bool:
        """État vu par /ready (ne bloque pas si le coupe-circuit est ouvert)."""
        if self._client is None or not self._healthy:
            return False
        try:
            return bool(self._client.ping())
        except redis.RedisError:
            return False
