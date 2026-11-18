"""API : contrats, erreurs, cache, persistance — sans infrastructure (SQLite, Redis simulé)."""

import time

import pandas as pd
import pytest
import redis
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select

from imdb_mlops import db
from imdb_mlops.api import app as api_module
from imdb_mlops.api.cache import PredictionCache
from imdb_mlops.api.model_store import Loaded, ModelStore
from imdb_mlops.api.telemetry import TraceWriter
from imdb_mlops.config import load_config
from imdb_mlops.features import SeriesHistory


class FakeModel:
    def predict(self, X):
        return X["series_prev_mean"].fillna(7.0).to_numpy()


class FakeRedis:
    def __init__(self, fail=False):
        self.data, self.fail = {}, fail

    def _check(self):
        if self.fail:
            raise redis.ConnectionError("down")

    def mget(self, keys):
        self._check()
        return [self.data.get(k) for k in keys]

    def pipeline(self, transaction=False):
        return self

    def setex(self, k, ttl, v):
        self.data[k] = v

    def execute(self):
        self._check()

    def ping(self):
        self._check()
        return True


class StaticStore(ModelStore):
    def refresh(self, force=False):
        return False


def _history():
    df = pd.DataFrame(
        {
            "series_id": ["tt0000001"] * 4,
            "series_votes": [1000] * 4,
            "season": [1, 1, 1, 2],
            "episode": [1, 2, 3, 1],
            "rating": [8.0, 7.0, 9.0, 8.5],
            "votes": [100, 90, 120, 110],
        }
    )
    return {"tt0000001": SeriesHistory.from_frame(df)}


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setattr(db, "get_engine", lambda: engine)
    clients = []

    def _make(loaded=True, redis_client=None):
        store = StaticStore(load_config())
        if loaded:
            store.set(
                Loaded(
                    model=FakeModel(),
                    name="m",
                    version="7",
                    run_id="r" * 32,
                    rmse=0.5,
                    data_sha="d" * 40,
                    history=_history(),
                )
            )
        api_module.app.state.store = store
        api_module.app.state.cache = PredictionCache(None, client=redis_client or FakeRedis())
        api_module.app.state.writer = TraceWriter(flush_every_s=0.05)
        client = TestClient(api_module.app)
        client.__enter__()
        clients.append(client)
        return client, engine

    yield _make
    for c in clients:
        c.__exit__(None, None, None)
    for attr in ("store", "cache", "writer"):
        if hasattr(api_module.app.state, attr):
            delattr(api_module.app.state, attr)


EP = {"series_id": "tt0000001", "season": 2, "episode": 2}


def test_predict_returns_version_and_uses_cache(make_client):
    client, _ = make_client()
    r1 = client.post("/predict", json=EP)
    assert r1.status_code == 200
    body = r1.json()
    assert body["model_version"] == "7"
    assert r1.headers["X-Model-Version"] == "7"
    assert body["predicted_rating"] == pytest.approx(8.125)  # moyenne des 4 précédents
    assert body["cached"] is False
    r2 = client.post("/predict", json=EP)
    assert r2.json()["cached"] is True
    assert r2.json()["predicted_rating"] == body["predicted_rating"]


@pytest.mark.parametrize(
    "payload",
    [
        {"series_id": "abc", "season": 1, "episode": 1},
        {"series_id": "tt0000001", "season": -1, "episode": 1},
        {"series_id": "tt0000001", "season": 1},
        {"series_id": "tt0000001", "season": 1, "episode": 1, "extra": 1},
    ],
)
def test_invalid_input_is_rejected(make_client, payload):
    client, _ = make_client()
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_input"
    assert r.json()["request_id"]


def test_unknown_series(make_client):
    client, _ = make_client()
    r = client.post("/predict", json={"series_id": "tt9999999", "season": 1, "episode": 1})
    assert r.status_code == 404
    assert r.json()["error"] == "unknown_series"


def test_batch_partial_success(make_client):
    client, _ = make_client()
    items = [
        EP,
        {"series_id": "tt9999999", "season": 1, "episode": 1},
        {"series_id": "tt0000001", "season": 1, "episode": 2},
    ]
    r = client.post("/predict/batch", json={"items": items})
    assert r.status_code == 200
    body = r.json()
    assert (body["n_ok"], body["n_errors"]) == (2, 1)
    assert body["errors"][0]["index"] == 1


def test_batch_size_limit(make_client):
    client, _ = make_client()
    r = client.post("/predict/batch", json={"items": [EP] * 1001})
    assert r.status_code == 422


def test_traces_and_feedback(make_client):
    client, engine = make_client()
    pid = client.post("/predict", json=EP).json()["prediction_id"]
    client.post("/predict", json={"series_id": "tt9999999", "season": 1, "episode": 1})
    time.sleep(0.3)  # écriture asynchrone par lots
    with engine.connect() as conn:
        statuses = {
            status: n
            for status, n in conn.execute(
                select(db.predictions.c.status, func.count()).group_by(db.predictions.c.status)
            ).all()
        }
        n_requests = conn.execute(select(func.count()).select_from(db.api_requests)).scalar()
    assert statuses == {"ok": 1, "error": 1}
    assert n_requests >= 2

    r = client.post("/feedback", json={"prediction_id": pid, "rating": 9.1})
    assert r.status_code == 201
    r = client.post("/feedback", json={"prediction_id": "x" * 32, "rating": 9.1})
    assert r.status_code == 404
    r = client.post("/feedback", json={"prediction_id": pid, "rating": 11})
    assert r.status_code == 422


def test_model_unavailable_gives_503(make_client):
    client, _ = make_client(loaded=False)
    assert client.get("/ready").status_code == 503
    r = client.post("/predict", json=EP)
    assert r.status_code == 503
    assert r.json()["error"] == "model_unavailable"


def test_redis_down_degrades_gracefully(make_client):
    client, _ = make_client(redis_client=FakeRedis(fail=True))
    r = client.post("/predict", json=EP)
    assert r.status_code == 200
    assert r.json()["cached"] is False
    assert client.get("/ready").json()["cache"] == "down"


def test_ready_reports_disabled_redis(make_client):
    client, _ = make_client()
    api_module.app.state.cache = PredictionCache(None, local_size=10)
    body = client.get("/ready").json()
    assert (body["cache"], body["local_cache"]) == ("disabled", "enabled")


def test_metrics_exposed(make_client):
    client, _ = make_client()
    client.post("/predict", json=EP)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "predictions_total" in text


def test_local_lru_cache_without_redis():
    from imdb_mlops.api.cache import LocalLRU

    cache = PredictionCache(None, local_size=2)
    assert cache.enabled
    cache.set_many({"a": 1.0, "b": 2.0})
    assert cache.get_many(["a", "b", "c"]) == [1.0, 2.0, None]
    cache.set_many({"c": 3.0})  # éviction du moins récemment utilisé
    lru = LocalLRU(2)
    lru.set("x", 1.0)
    lru.set("y", 2.0)
    lru.get("x")
    lru.set("z", 3.0)
    assert lru.get("y") is None and lru.get("x") == 1.0


def test_local_cache_in_front_of_redis(make_client):
    redis_client = FakeRedis()
    client, _ = make_client(redis_client=redis_client)
    api_module.app.state.cache = PredictionCache(None, client=redis_client, local_size=100)
    assert client.post("/predict", json=EP).json()["cached"] is False
    redis_client.fail = True  # Redis en panne : le niveau local sert encore
    assert client.post("/predict", json=EP).json()["cached"] is True


def test_traffic_source_is_recorded(make_client):
    client, engine = make_client()
    client.post("/predict", json=EP, headers={"X-Traffic-Source": "loadtest"})
    client.post("/predict", json=EP, headers={"X-Traffic-Source": "nimporte"})
    client.post("/predict", json=EP)
    time.sleep(0.3)
    with engine.connect() as conn:
        sources = sorted(r[0] for r in conn.execute(select(db.predictions.c.source)))
    assert sources == ["loadtest", "user", "user"]


def test_trace_writer_drains_queue_on_stop(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{(tmp_path / 'w.db').as_posix()}")
    monkeypatch.setattr(db, "get_engine", lambda: engine)
    db.metadata.create_all(engine)
    writer = TraceWriter(batch_size=2)  # lots de 20 lignes : 3 lots pour 45 traces
    row = {
        "created_at": pd.Timestamp.now(tz="UTC").to_pydatetime(),
        "method": "GET",
        "path": "/x",
        "status_code": 200,
        "duration_ms": 1.0,
    }
    for _ in range(45):
        writer.put("api_requests", dict(row))
    writer.stop()
    with engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(db.api_requests)).scalar() == 45


def test_admin_reload_requires_token(make_client, monkeypatch):
    client, _ = make_client()
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    assert client.post("/admin/reload").status_code == 403
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    assert client.post("/admin/reload").status_code == 401
    assert client.post("/admin/reload", headers={"X-Admin-Token": "faux"}).status_code == 401
    r = client.post("/admin/reload", headers={"X-Admin-Token": "s3cret"})
    assert r.status_code == 200
    assert r.json()["model_version"] == "7"


def test_check_features_rejects_incompatible_model():
    from imdb_mlops.api.model_store import ModelUnavailableError, check_features
    from imdb_mlops.features import FEATURES

    check_features({})  # versions anciennes sans tag : acceptées
    check_features({"features": ",".join(FEATURES)})
    with pytest.raises(ModelUnavailableError):
        check_features({"features": ",".join(reversed(FEATURES))})


def test_framework_errors_use_unified_format(make_client):
    client, _ = make_client()
    r = client.get("/inexistant")
    assert r.status_code == 404
    assert r.json()["error"] == "not_found" and r.json()["request_id"]
    r = client.post(
        "/predict", content=b"{pas du json", headers={"Content-Type": "application/json"}
    )
    assert r.status_code in (400, 422)
    assert "error" in r.json() and "request_id" in r.json()
