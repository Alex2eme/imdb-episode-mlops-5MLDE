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
from collections import OrderedDict

import redis

from imdb_mlops.api import telemetry

log = logging.getLogger(__name__)


class LocalLRU:
    """Cache local au worker (niveau 1) : aucun aller-retour réseau.

    Ajouté après les tests de charge : avec un calcul de prédiction à ~1,5 ms, deux
    allers-retours Redis (~1,1 ms chacun sur la plateforme de mesure) coûtaient plus cher que
    le calcul pour un taux de hit de ~17 % (débit -40 %).
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self._data: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> float | None:
        with self._lock:
            value = self._data.get(key)
            if value is not None:
                self._data.move_to_end(key)
            return value

    def set(self, key: str, value: float) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            if len(self._data) > self.max_size:
                self._data.popitem(last=False)


class PredictionCache:
    """Cache à deux niveaux : LRU local (par worker) puis Redis (partagé, optionnel)."""

    def __init__(
        self,
        url: str | None,
        ttl_s: int = 86400,
        cooldown_s: float = 2.0,
        client=None,
        local_size: int = 0,
    ):
        self.local = LocalLRU(local_size) if local_size > 0 else None
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
    def redis_configured(self) -> bool:
        return self._client is not None

    @property
    def enabled(self) -> bool:
        return self._client is not None or self.local is not None

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
        out: list[float | None] = [None] * len(keys)
        if self.local is not None:
            out = [self.local.get(k) for k in keys]
            n_local = sum(v is not None for v in out)
            telemetry.CACHE_REQUESTS.labels(result="hit_local").inc(n_local)
            missing = [i for i, v in enumerate(out) if v is None]
            if not missing or self._client is None:
                telemetry.CACHE_REQUESTS.labels(result="miss").inc(len(missing))
                return out
            remote = self._get_remote([keys[i] for i in missing])
            for i, v in zip(missing, remote, strict=True):
                if v is not None:
                    out[i] = v
                    self.local.set(keys[i], v)
            return out
        return self._get_remote(keys)

    def _get_remote(self, keys: list[str]) -> list[float | None]:
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
        if self.local is not None:
            for k, v in items.items():
                self.local.set(k, v)
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
