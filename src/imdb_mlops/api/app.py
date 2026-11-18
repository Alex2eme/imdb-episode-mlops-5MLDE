"""API de prédiction de la note d'épisodes IMDb.

Lancement : uvicorn imdb_mlops.api.app:app --host 0.0.0.0 --port 8000
Documentation interactive : /docs
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from sqlalchemy import select
from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException

from imdb_mlops import __version__, db
from imdb_mlops.api import telemetry
from imdb_mlops.api.cache import PredictionCache
from imdb_mlops.api.model_store import ModelStore, ModelUnavailableError
from imdb_mlops.api.schemas import (
    BatchItemError,
    BatchRequest,
    BatchResponse,
    EpisodeRef,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    ModelInfo,
    Prediction,
)
from imdb_mlops.config import get_settings, load_config

log = logging.getLogger("imdb_mlops.api")

REFRESH_S = float(os.environ.get("MODEL_REFRESH_S", "30"))
CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "true").lower() == "true"
TRACES_ENABLED = os.environ.get("TRACES_ENABLED", "true").lower() == "true"
# Cache local LRU par worker (entrées) ; 0 = désactivé. Voir api/cache.py.
LOCAL_CACHE_SIZE = int(os.environ.get("LOCAL_CACHE_SIZE", "50000"))
STARTUP_LOAD_ATTEMPTS = int(os.environ.get("STARTUP_LOAD_ATTEMPTS", "5"))

ERROR_RESPONSES = {
    404: {"model": ErrorResponse, "description": "Série ou prédiction inconnue"},
    422: {"model": ErrorResponse, "description": "Entrée invalide"},
    503: {"model": ErrorResponse, "description": "Modèle indisponible"},
}


class ApiError(HTTPException):
    def __init__(self, status: int, code: str, message: str, details=None):
        super().__init__(status_code=status, detail=message)
        self.code, self.details = code, details


def _refresh_loop(app: FastAPI, stop: threading.Event) -> None:
    while not stop.wait(REFRESH_S):
        _refresh(app)


def _refresh(app: FastAPI, force: bool = False) -> bool:
    store: ModelStore = app.state.store
    changed = store.refresh(force=force)
    if changed:
        telemetry.MODEL_RELOADS.labels(result="success").inc()
        telemetry.set_model_info(store.current.version, store.current.data_sha)
    elif store.last_error:
        telemetry.MODEL_RELOADS.labels(result="error").inc()
    return changed


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    app.state.store = getattr(app.state, "store", None) or ModelStore(cfg)
    app.state.cache = getattr(app.state, "cache", None) or PredictionCache(
        get_settings().redis_url if CACHE_ENABLED else None, local_size=LOCAL_CACHE_SIZE
    )
    app.state.writer = getattr(app.state, "writer", None) or telemetry.TraceWriter(
        enabled=TRACES_ENABLED
    )
    if TRACES_ENABLED:
        try:
            db.init_db()
        except Exception as exc:
            log.warning("Base indisponible au démarrage : %s", exc)
    # Plusieurs tentatives au démarrage : avec N workers qui téléchargent le modèle en même
    # temps, MLflow peut refuser une requête (constaté à 4 workers : 503 pendant 30 s). Après
    # ces tentatives, démarrage en mode dégradé (503) et nouvel essai toutes les 30 s.
    for attempt in range(STARTUP_LOAD_ATTEMPTS):
        if app.state.store.ready:
            break
        if attempt:
            await asyncio.sleep(2 * attempt)
        _refresh(app, force=True)
    app.state.writer.start()
    stop = threading.Event()
    threading.Thread(target=_refresh_loop, args=(app, stop), daemon=True).start()
    yield
    stop.set()
    app.state.writer.stop()


app = FastAPI(
    title="IMDb Episode Rating API",
    version=__version__,
    description="Prédit la note IMDb d'un épisode à partir de l'historique de sa série. "
    "Chaque réponse indique la version du modèle utilisée.",
    lifespan=lifespan,
)


# --- Middleware : identifiant de requête, durée, traces --------------------------------


class ObserveMiddleware:
    """Middleware ASGI pur : identifiant de requête, version, métriques, traces.

    Remplace `@app.middleware("http")` (BaseHTTPMiddleware de Starlette), dont le surcoût
    mesuré était d'environ 6 ms par requête dans ce conteneur (tests de charge).
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = (
            next((v.decode() for k, v in scope["headers"] if k == b"x-request-id"), None)
            or uuid.uuid4().hex
        )
        state = scope.setdefault("state", {})
        state["request_id"], state["error"] = request_id, None
        store: ModelStore = scope["app"].state.store
        version = store.current.version if store.ready else None
        status = {"code": 500}
        start = time.perf_counter()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                if version:
                    headers["X-Model-Version"] = version
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            self._record(scope, state, status["code"], time.perf_counter() - start, version)

    @staticmethod
    def _record(scope, state, status_code: int, duration: float, version) -> None:
        endpoint = getattr(scope.get("route"), "path", "unmatched")
        if endpoint == "/metrics":
            return
        method = scope["method"]
        telemetry.REQUESTS.labels(endpoint, method, str(status_code)).inc()
        telemetry.LATENCY.labels(endpoint).observe(duration)
        # Sondes de santé (Docker) : comptées dans Prometheus, pas tracées en base.
        if endpoint != "/health":
            scope["app"].state.writer.put(
                "api_requests",
                {
                    "created_at": datetime.now(UTC),
                    "request_id": state["request_id"],
                    "method": method,
                    "path": endpoint,
                    "status_code": status_code,
                    "duration_ms": duration * 1000,
                    "error": state.get("error"),
                    "model_version": version,
                    "worker_pid": os.getpid(),
                },
            )


app.add_middleware(ObserveMiddleware)


def _error(request: Request, status: int, code: str, message: str, details=None):
    request.state.error = code
    body = ErrorResponse(
        error=code,
        message=message,
        request_id=getattr(request.state, "request_id", ""),
        details=details,
    )
    return JSONResponse(status_code=status, content=body.model_dump())


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError):
    return _error(request, exc.status_code, exc.code, exc.detail, exc.details)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    details = [
        {"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]} for e in exc.errors()
    ]
    return _error(request, 422, "invalid_input", "Entrée invalide", details)


@app.exception_handler(ModelUnavailableError)
async def model_unavailable_handler(request: Request, exc: ModelUnavailableError):
    return _error(request, 503, "model_unavailable", f"Modèle indisponible : {exc}")


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    """Erreurs HTTP du framework (route inconnue, méthode, corps illisible) : même format
    que les erreurs métier, avec request_id."""
    codes = {400: "bad_request", 404: "not_found", 405: "method_not_allowed"}
    return _error(
        request, exc.status_code, codes.get(exc.status_code, "http_error"), str(exc.detail)
    )


@app.exception_handler(Exception)
async def unexpected_handler(request: Request, exc: Exception):
    log.exception("Erreur inattendue")
    return _error(request, 500, "internal_error", "Erreur interne")


# --- Prédiction ------------------------------------------------------------------------


TRAFFIC_SOURCES = {"user", "simulation", "loadtest"}


def _traffic_source(request: Request) -> str:
    """Origine déclarée du trafic ; toute valeur inconnue compte comme utilisateur réel."""
    value = request.headers.get("X-Traffic-Source", "user").lower()
    return value if value in TRAFFIC_SOURCES else "user"


def _predict_refs(request: Request, refs: list[EpisodeRef], endpoint: str):
    """Cache → modèle pour les absents → persistance. Retourne (prédictions, erreurs)."""
    store: ModelStore = request.app.state.store
    cache: PredictionCache = request.app.state.cache
    loaded = store.current
    start = time.perf_counter()

    keys = [
        cache.key(loaded.version, loaded.data_sha, r.series_id, r.season, r.episode) for r in refs
    ]
    cached = cache.get_many(keys) if cache.enabled else [None] * len(refs)
    miss_idx = [i for i, v in enumerate(cached) if v is None]
    values: list[float | None] = list(cached)
    errors: list[str | None] = [None] * len(refs)
    if miss_idx:
        preds, errs = store.predict(
            [(refs[i].series_id, refs[i].season, refs[i].episode) for i in miss_idx]
        )
        to_cache = {}
        for j, i in enumerate(miss_idx):
            if errs[j]:
                errors[i] = errs[j]
            else:
                values[i] = float(preds[j])
                to_cache[keys[i]] = values[i]
        cache.set_many(to_cache)

    latency_ms = (time.perf_counter() - start) * 1000
    now = datetime.now(UTC)
    out, out_errors = [], []
    for i, ref in enumerate(refs):
        pid = uuid.uuid4().hex
        is_cached = cached[i] is not None
        row = {
            "id": pid,
            "created_at": now,
            "request_id": request.state.request_id,
            "endpoint": endpoint,
            "source": _traffic_source(request),
            "series_id": ref.series_id,
            "season": ref.season,
            "episode": ref.episode,
            "model_version": loaded.version,
            "model_run_id": loaded.run_id,
            "data_sha": loaded.data_sha,
            "latency_ms": latency_ms,
            "cached": is_cached,
        }
        if errors[i]:
            telemetry.PREDICTION_ERRORS.labels(error=errors[i]).inc()
            request.app.state.writer.put(
                "predictions",
                row | {"status": "error", "error": errors[i], "predicted_rating": None},
            )
            out_errors.append((i, ref, errors[i]))
            continue
        rating = round(values[i], 3)
        telemetry.PREDICTIONS.labels(loaded.version, str(is_cached).lower()).inc()
        telemetry.PREDICTED_RATING.observe(rating)
        request.app.state.writer.put(
            "predictions", row | {"status": "ok", "error": None, "predicted_rating": rating}
        )
        out.append(
            Prediction(
                prediction_id=pid,
                series_id=ref.series_id,
                season=ref.season,
                episode=ref.episode,
                predicted_rating=rating,
                model_version=loaded.version,
                data_sha=loaded.data_sha[:10],
                cached=is_cached,
                latency_ms=round(latency_ms, 2),
            )
        )
    return out, out_errors, latency_ms


@app.post("/predict", response_model=Prediction, responses=ERROR_RESPONSES, tags=["prédiction"])
def predict(ref: EpisodeRef, request: Request) -> Prediction:
    """Prédit la note d'un épisode (existant ou futur) d'une série connue."""
    preds, errors, _ = _predict_refs(request, [ref], "single")
    if errors:
        raise ApiError(
            404, "unknown_series", f"Série {ref.series_id} absente du dernier snapshot validé"
        )
    return preds[0]


@app.post(
    "/predict/batch", response_model=BatchResponse, responses=ERROR_RESPONSES, tags=["prédiction"]
)
def predict_batch(body: BatchRequest, request: Request) -> BatchResponse:
    """Prédiction par lot (1 à 1000 épisodes). Succès partiel : les éléments en erreur sont
    listés sans faire échouer le lot."""
    preds, errors, latency_ms = _predict_refs(request, body.items, "batch")
    return BatchResponse(
        model_version=request.app.state.store.current.version,
        n_ok=len(preds),
        n_errors=len(errors),
        latency_ms=round(latency_ms, 2),
        predictions=preds,
        errors=[
            BatchItemError(
                index=i,
                series_id=r.series_id,
                season=r.season,
                episode=r.episode,
                error=code,
                message="Série absente du dernier snapshot validé",
            )
            for i, r, code in errors
        ],
    )


# --- Feedback --------------------------------------------------------------------------


@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=201,
    responses=ERROR_RESPONSES,
    tags=["feedback"],
)
def post_feedback(body: FeedbackRequest, request: Request) -> FeedbackResponse:
    """Enregistre une note observée pour une prédiction (évaluation en production)."""
    try:
        with db.get_engine().begin() as conn:
            exists = conn.execute(
                select(db.predictions.c.id).where(db.predictions.c.id == body.prediction_id)
            ).first()
            if exists is None:
                raise ApiError(
                    404,
                    "prediction_not_found",
                    f"Prédiction {body.prediction_id} inconnue (les traces sont "
                    "écrites en base sous une seconde : réessayer)",
                )
            fid = uuid.uuid4().hex
            conn.execute(
                db.feedback.insert().values(
                    id=fid,
                    created_at=datetime.now(UTC),
                    prediction_id=body.prediction_id,
                    rating=body.rating,
                    comment=body.comment,
                )
            )
    except ApiError:
        raise
    except Exception as exc:
        log.error("Feedback non enregistré : %s", exc)
        raise ApiError(503, "storage_unavailable", "Stockage des feedbacks indisponible") from exc
    telemetry.FEEDBACK.inc()
    return FeedbackResponse(feedback_id=fid, prediction_id=body.prediction_id)


# --- Exploitation ----------------------------------------------------------------------


@app.get("/health", tags=["exploitation"])
async def health() -> dict:
    """Liveness : le processus répond."""
    return {"status": "ok"}


def _cache_status(cache: PredictionCache) -> str:
    """État de Redis : `disabled` (non configuré), `up` ou `down` (contourné)."""
    if not cache.redis_configured:
        return "disabled"
    return "up" if cache.ping() else "down"


@app.get("/ready", tags=["exploitation"])
def ready(request: Request) -> JSONResponse:
    """Readiness : modèle chargé ; état des dépendances (Redis non bloquant)."""
    store: ModelStore = request.app.state.store
    body = {
        "model_loaded": store.ready,
        "model_version": store.current.version if store.ready else None,
        "last_error": store.last_error,
        "cache": _cache_status(request.app.state.cache),
        "local_cache": "enabled" if request.app.state.cache.local is not None else "disabled",
    }
    return JSONResponse(status_code=200 if store.ready else 503, content=body)


@app.get("/model", response_model=ModelInfo, responses=ERROR_RESPONSES, tags=["exploitation"])
async def model_info(request: Request) -> ModelInfo:
    """Version servie (alias `champion` du registre) et données d'historique utilisées."""
    m = request.app.state.store.current
    return ModelInfo(
        name=m.name,
        version=m.version,
        run_id=m.run_id,
        data_sha=m.data_sha,
        rmse=m.rmse,
        loaded_at=m.loaded_at,
    )


@app.post("/admin/reload", tags=["exploitation"])
def reload_model(request: Request) -> dict:
    """Force la relecture du champion (sinon automatique toutes les MODEL_REFRESH_S).

    Protégé par l'en-tête `X-Admin-Token` (variable ADMIN_TOKEN) ; désactivé si non défini.
    """
    expected = os.environ.get("ADMIN_TOKEN", "")
    if not expected:
        raise ApiError(403, "admin_disabled", "Endpoint d'administration désactivé (ADMIN_TOKEN)")
    given = request.headers.get("X-Admin-Token", "")
    if not hmac.compare_digest(given.encode(), expected.encode()):
        raise ApiError(401, "unauthorized", "Jeton d'administration invalide")
    changed = _refresh(request.app)
    store: ModelStore = request.app.state.store
    if not store.ready:
        raise ModelUnavailableError(store.last_error)
    return {
        "reloaded": changed,
        "model_version": store.current.version,
        "last_error": store.last_error,
    }


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        from prometheus_client import multiprocess

        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
