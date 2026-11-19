"""Scénarios d'incidents contrôlés (présentation du projet uniquement).

Chaque scénario reçoit un contexte `ctx` (journal, pilotage Docker, Prefect, URL des services)
et journalise ses étapes. Tout service arrêté par un scénario est redémarré dans un `finally`,
y compris si le scénario échoue.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy import select

from imdb_mlops import db
from imdb_mlops.data.faults import FAULTS

DAILY = "daily-pipeline/daily-pipeline"
FINAL_STATES = {"COMPLETED", "FAILED", "CRASHED", "CANCELLED"}


@dataclass
class Scenario:
    key: str
    title: str
    summary: str
    expected: str
    look_at: list[tuple[str, str]]
    duration: str
    run: Any
    params: dict = field(default_factory=dict)


# --- Outils ---------------------------------------------------------------------------


class Prefect:
    """Accès minimal à Prefect : lancer le déploiement et suivre les états."""

    def start(self, parameters: dict) -> str:
        from prefect.deployments import run_deployment

        run = run_deployment(DAILY, parameters=parameters, timeout=0)
        return str(run.id)

    def states(self, flow_run_id: str) -> tuple[str, dict[str, str]]:
        from prefect.client.orchestration import get_client
        from prefect.client.schemas.filters import FlowRunFilter, FlowRunFilterId

        with get_client(sync_client=True) as client:
            fr = client.read_flow_run(flow_run_id)
            tasks = client.read_task_runs(
                flow_run_filter=FlowRunFilter(id=FlowRunFilterId(any_=[flow_run_id]))
            )
        flow_state = fr.state.type.value if fr.state else "PENDING"
        return flow_state, {t.name: t.state.name for t in tasks if t.state}


def _wait_flow(ctx, run_id: str, timeout_s: float, until_task_state: str | None = None):
    """Suit un run Prefect en journalisant chaque changement d'état.

    Retourne l'état final du flow, ou `until_task_state` dès qu'une tâche l'atteint.
    """
    seen: dict[str, str] = {}
    deadline = time.monotonic() + timeout_s
    flow_state = "PENDING"
    while time.monotonic() < deadline:
        flow_state, tasks = ctx.prefect.states(run_id)
        for name, state in tasks.items():
            if seen.get(name) != state:
                ctx.log(f"Prefect · tâche {name.rsplit('-', 1)[0]} → {state}")
                seen[name] = state
            if until_task_state and state == until_task_state:
                return until_task_state
        if flow_state in FINAL_STATES:
            ctx.log(f"Prefect · flow → {flow_state}")
            return flow_state
        time.sleep(4)
    ctx.log(f"Délai dépassé ({timeout_s:.0f} s), dernier état du flow : {flow_state}")
    return flow_state


def _alerts_since(ctx, since: datetime) -> None:
    try:
        with db.get_engine().connect() as conn:
            rows = conn.execute(
                select(db.alerts.c.severity, db.alerts.c.title)
                .where(db.alerts.c.created_at >= since)
                .order_by(db.alerts.c.id)
            ).all()
    except Exception as exc:
        ctx.log(f"Alertes non lisibles : {exc}")
        return
    if not rows:
        ctx.log("Aucune alerte émise")
    for severity, title in rows:
        ctx.log(f"Alerte [{severity}] {title}")


def _ready(ctx) -> dict:
    try:
        return httpx.get(f"{ctx.api_url}/ready", timeout=10).json()
    except httpx.HTTPError as exc:
        return {"erreur": str(exc)}


# --- Scénarios ------------------------------------------------------------------------


def invalid_batch(ctx, fault: str = "ratings_out_of_range") -> str:
    if fault not in FAULTS:
        raise ValueError(f"Défaut inconnu : {fault}")
    since = datetime.now(UTC)
    ctx.log(f"Lancement de daily-pipeline avec le défaut « {fault} » : {FAULTS[fault]}")
    run_id = ctx.prefect.start({"fault": fault})
    ctx.log(f"Run Prefect {run_id[:8]}")
    final = _wait_flow(ctx, run_id, timeout_s=600)
    with db.get_engine().connect() as conn:
        snap = conn.execute(
            select(db.data_snapshots.c.quality_report)
            .where(db.data_snapshots.c.sha.like(f"%-fault-{fault}"))
            .order_by(db.data_snapshots.c.validated_at.desc())
            .limit(1)
        ).scalar()
    for check in (snap or {}).get("checks", []):
        if not check["success"] and check["severity"] == "critical":
            ctx.log(f"Contrôle bloquant : {check['name']} {check['details']}")
    _alerts_since(ctx, since)
    return f"Flow {final} : lot rejeté, entraînement non lancé" if final == "FAILED" else final


def mlflow_recovery(ctx) -> str:
    since = datetime.now(UTC)
    try:
        ctx.log("Arrêt de MLflow")
        ctx.docker.stop("mlflow")
        ctx.log("Lancement de daily-pipeline (2 trials)")
        run_id = ctx.prefect.start({"n_trials": 2})
        state = _wait_flow(ctx, run_id, timeout_s=420, until_task_state="AwaitingRetry")
        if state != "AwaitingRetry":
            return f"Relance non observée (état : {state})"
        ctx.log("Relance programmée par Prefect : redémarrage de MLflow pendant le délai")
    finally:
        ctx.docker.start("mlflow")
    ctx.log(f"MLflow : {ctx.docker.wait_healthy('mlflow')}")
    final = _wait_flow(ctx, run_id, timeout_s=900)
    _alerts_since(ctx, since)
    return f"Flow {final} après reprise automatique"


def mlflow_durable(ctx) -> str:
    since = datetime.now(UTC)
    try:
        ctx.log("Arrêt de MLflow pour toute la durée du flow")
        ctx.docker.stop("mlflow")
        ctx.log("Lancement de daily-pipeline (2 trials) : 3 tentatives attendues")
        run_id = ctx.prefect.start({"n_trials": 2})
        final = _wait_flow(ctx, run_id, timeout_s=900)
    finally:
        ctx.log("Redémarrage de MLflow (fin du scénario)")
        ctx.docker.start("mlflow")
    ctx.log(f"MLflow : {ctx.docker.wait_healthy('mlflow')}")
    _alerts_since(ctx, since)
    return f"Flow {final} : échec définitif après relances, alerte critique émise"


def redis_outage(ctx, n_requests: int = 10) -> str:
    ctx.log(f"État initial /ready : {_ready(ctx)}")
    codes = []
    try:
        ctx.log("Arrêt de Redis")
        ctx.docker.stop("redis")
        for e in range(1, n_requests + 1):
            t = time.perf_counter()
            r = httpx.post(
                f"{ctx.api_url}/predict",
                timeout=30,
                headers={"X-Traffic-Source": "simulation"},
                json={"series_id": "tt0108778", "season": 4, "episode": e},
            )
            codes.append(r.status_code)
            elapsed_ms = (time.perf_counter() - t) * 1000
            ctx.log(f"Prédiction {e} : HTTP {r.status_code} en {elapsed_ms:.0f} ms")
        ctx.log(f"/ready pendant la panne : {_ready(ctx)}")
    finally:
        ctx.log("Redémarrage de Redis")
        ctx.docker.start("redis")
    ctx.log(f"Redis : {ctx.docker.wait_healthy('redis')}")
    time.sleep(4)  # sonde de l'API : 2 s
    ctx.log(f"/ready après rétablissement : {_ready(ctx)}")
    ok = sum(c == 200 for c in codes)
    return f"{ok}/{len(codes)} prédictions servies pendant la panne de Redis"


def api_outage(ctx) -> str:
    def alert_state() -> str:
        try:
            r = httpx.get(f"{ctx.prometheus_url}/api/v1/alerts", timeout=10).json()
            states = [
                a["state"] for a in r["data"]["alerts"] if a["labels"].get("alertname") == "ApiDown"
            ]
            return states[0] if states else "inactive"
        except (httpx.HTTPError, KeyError, ValueError):
            return "inconnu"

    try:
        ctx.log("Arrêt de l'API")
        ctx.docker.stop("api")
        deadline, last = time.monotonic() + 150, None
        while time.monotonic() < deadline:
            state = alert_state()
            if state != last:
                ctx.log(f"Prometheus · alerte ApiDown : {state}")
                last = state
            if state == "firing":
                break
            time.sleep(5)
    finally:
        ctx.log("Redémarrage de l'API")
        ctx.docker.start("api")
    ctx.log(f"API : {ctx.docker.wait_healthy('api', timeout_s=240)}")
    return f"Alerte ApiDown observée : {last}"


def rollback_cycle(ctx) -> str:
    from mlflow import MlflowClient

    from imdb_mlops.config import load_config
    from imdb_mlops.training import registry

    cfg = load_config()
    name = cfg.training.registered_model_name
    client = MlflowClient()
    champ = registry.get_alias_version(client, name, registry.CHAMPION)
    prev = registry.get_alias_version(client, name, registry.PREVIOUS)
    ctx.log(
        f"Champion actuel : v{champ.version if champ else '-'} · "
        f"précédent : v{prev.version if prev else '-'}"
    )
    if prev is None:
        return "Aucun previous_champion : rien à remettre en service"
    version = registry.rollback(cfg, None, "Console de démonstration")
    ctx.log(f"Registre : champion → v{version}")
    r = httpx.post(
        f"{ctx.api_url}/admin/reload", headers={"X-Admin-Token": ctx.admin_token}, timeout=60
    )
    ctx.log(f"/admin/reload : HTTP {r.status_code} {r.json()}")
    time.sleep(2)
    model = httpx.get(f"{ctx.api_url}/model", timeout=10).json()
    ctx.log(f"/model (un des workers ; les autres suivent en 30 s) : v{model.get('version')}")
    return f"Version v{version} servie sans redémarrage — relancer pour revenir"


def simulate_traffic(ctx, n: int = 300) -> str:
    from imdb_mlops.monitoring.simulate import simulate

    ctx.log(f"Envoi de {n} requêtes réalistes (épisodes connus, futurs, erreurs, lots)")
    stats = simulate(ctx.api_url, n=n)
    for k, v in stats.items():
        ctx.log(f"{k} : {v}")
    return f"{n} requêtes envoyées : voir le dashboard Grafana « API »"


SCENARIOS: dict[str, Scenario] = {
    s.key: s
    for s in [
        Scenario(
            "invalid_batch",
            "Lot de données invalide",
            "Injecte un défaut dans le snapshot du jour puis lance la chaîne complète.",
            "Ingestion en échec, aucune relance, aucun entraînement, deux alertes critiques.",
            [("Prefect", ":4200/runs"), ("Grafana · Traitements", ":3000/d/imdb-pipeline")],
            "≈ 1 min",
            invalid_batch,
            {"fault": sorted(FAULTS)},
        ),
        Scenario(
            "mlflow_recovery",
            "MLflow indisponible, puis rétabli",
            "Arrête MLflow, lance la chaîne, puis redémarre MLflow pendant le délai de relance.",
            "train-and-promote : échec, AwaitingRetry, puis succès ; flow Completed.",
            [("Prefect", ":4200/runs"), ("MLflow", ":5000")],
            "≈ 4 min",
            mlflow_recovery,
        ),
        Scenario(
            "mlflow_durable",
            "MLflow durablement indisponible",
            "Arrête MLflow pendant tout le flow (redémarré à la fin du scénario).",
            "3 tentatives de train-and-promote, flow Failed, alerte critique.",
            [("Prefect", ":4200/runs"), ("Grafana · Traitements", ":3000/d/imdb-pipeline")],
            "≈ 6 min",
            mlflow_durable,
        ),
        Scenario(
            "redis_outage",
            "Redis indisponible",
            "Arrête Redis et envoie 10 prédictions. Redis est désactivé par défaut "
            "(CACHE_ENABLED=false) : il ne doit pas être une dépendance critique.",
            "Toutes les prédictions en HTTP 200 ; /ready indique l'état du cache.",
            [("API /ready", ":8000/ready"), ("Grafana · API", ":3000/d/imdb-api")],
            "≈ 30 s",
            redis_outage,
        ),
        Scenario(
            "api_outage",
            "API arrêtée : alerte Prometheus",
            "Arrête l'API jusqu'au déclenchement de l'alerte ApiDown, puis la redémarre.",
            "ApiDown : pending puis firing (≈ 30 s), puis API de nouveau healthy.",
            [("Prometheus · alertes", ":9090/alerts"), ("Grafana · API", ":3000/d/imdb-api")],
            "≈ 2 min",
            api_outage,
        ),
        Scenario(
            "rollback",
            "Rollback à chaud",
            "Remet en service previous_champion dans le registre MLflow, puis recharge l'API.",
            "L'API sert l'autre version sans redémarrage (relancer pour revenir).",
            [("MLflow · Models", ":5000/#/models"), ("API /model", ":8000/model")],
            "≈ 15 s",
            rollback_cycle,
        ),
        Scenario(
            "traffic",
            "Trafic simulé",
            "Envoie 300 requêtes réalistes pour alimenter les dashboards.",
            "Débit, latence, erreurs 4xx attendues et feedbacks visibles dans Grafana.",
            [("Grafana · API", ":3000/d/imdb-api")],
            "≈ 30 s",
            simulate_traffic,
        ),
    ]
}
