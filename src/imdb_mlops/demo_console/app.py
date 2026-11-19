"""Console web de démonstration des incidents contrôlés.

⚠ PRÉSENTATION DU PROJET UNIQUEMENT. Cette console arrête et redémarre des conteneurs via
le socket Docker et modifie le registre de modèles : elle ne doit jamais être déployée en
production. Elle n'est démarrée qu'avec le profil Compose `demo`, sur 127.0.0.1.

Lancement : docker compose --profile demo up -d demo-console → http://localhost:8080
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from imdb_mlops.demo_console.docker_ctl import ALLOWED_SERVICES, DockerControl
from imdb_mlops.demo_console.scenarios import SCENARIOS, Prefect

log = logging.getLogger("imdb_mlops.demo_console")
PAGE = Path(__file__).with_name("index.html")


@dataclass
class Job:
    id: str
    scenario: str
    title: str
    params: dict
    status: str = "running"  # running | succeeded | failed
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    ended_at: str | None = None
    result: str | None = None
    logs: list[dict] = field(default_factory=list)


class Context:
    def __init__(self, job: Job, docker: DockerControl, prefect: Prefect):
        self.job, self.docker, self.prefect = job, docker, prefect
        self.api_url = os.environ.get("API_URL", "http://api:8000")
        self.prometheus_url = os.environ.get("PROMETHEUS_URL", "http://prometheus:9090")
        self.admin_token = os.environ.get("ADMIN_TOKEN", "")

    def log(self, message: str) -> None:
        # Horodatage ISO (UTC) : la page l'affiche en heure locale du navigateur.
        self.job.logs.append({"t": datetime.now(UTC).isoformat(), "msg": message})
        log.info("[%s] %s", self.job.scenario, message)


class Runner:
    """Un seul scénario à la fois : deux incidents simultanés rendraient la démo illisible."""

    def __init__(self, docker: DockerControl | None = None, prefect: Prefect | None = None):
        self.docker = docker or DockerControl()
        self.prefect = prefect or Prefect()
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.current: str | None = None

    def submit(self, key: str, params: dict) -> Job:
        scenario = SCENARIOS.get(key)
        if scenario is None:
            raise HTTPException(404, f"Scénario inconnu : {key}")
        with self._lock:
            if self.current and self.jobs[self.current].status == "running":
                raise HTTPException(409, "Un scénario est déjà en cours")
            job = Job(uuid.uuid4().hex[:12], key, scenario.title, params)
            self.jobs[job.id] = job
            self.current = job.id
        threading.Thread(target=self._run, args=(job, scenario), daemon=True).start()
        return job

    def _run(self, job: Job, scenario) -> None:
        ctx = Context(job, self.docker, self.prefect)
        try:
            job.result = scenario.run(ctx, **job.params)
            job.status = "succeeded"
        except Exception as exc:
            ctx.log(f"Erreur : {type(exc).__name__}: {exc}")
            job.result, job.status = str(exc), "failed"
        finally:
            job.ended_at = datetime.now(UTC).isoformat()


class RunRequest(BaseModel):
    params: dict = {}


app = FastAPI(
    title="Console de démonstration (présentation uniquement)", docs_url=None, redoc_url=None
)
app.state.runner = None


def runner() -> Runner:
    if app.state.runner is None:
        app.state.runner = Runner()
    return app.state.runner


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE.read_text(encoding="utf-8")


@app.get("/api/scenarios")
def list_scenarios() -> list[dict]:
    return [
        {
            "key": s.key,
            "title": s.title,
            "summary": s.summary,
            "expected": s.expected,
            "look_at": s.look_at,
            "duration": s.duration,
            "params": s.params,
        }
        for s in SCENARIOS.values()
    ]


@app.post("/api/scenarios/{key}/run")
def run_scenario(key: str, body: RunRequest) -> dict:
    scenario = SCENARIOS.get(key)
    if scenario is None:
        raise HTTPException(404, f"Scénario inconnu : {key}")
    params = {k: v for k, v in body.params.items() if k in scenario.params}
    for k, v in params.items():
        if v not in scenario.params[k]:
            raise HTTPException(422, f"Valeur {v!r} non autorisée pour {k}")
    return {"job_id": runner().submit(key, params).id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> Job:
    job = runner().jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Exécution inconnue")
    return job


@app.get("/api/status")
def status() -> dict:
    r = runner()
    services = {}
    for service in sorted(ALLOWED_SERVICES):
        try:
            services[service] = r.docker.state(service)
        except Exception as exc:
            services[service] = f"inconnu ({type(exc).__name__})"
    current = r.jobs.get(r.current) if r.current else None
    return {
        "services": services,
        "current": current.id if current else None,
        "history": [
            {
                "id": j.id,
                "title": j.title,
                "status": j.status,
                "started_at": j.started_at,
                "result": j.result,
            }
            for j in sorted(r.jobs.values(), key=lambda j: j.started_at, reverse=True)[:10]
        ],
    }


@app.post("/api/restore")
def restore() -> dict:
    """Bouton de secours : redémarre tout service autorisé laissé arrêté."""
    r = runner()
    done = {}
    for service in sorted(ALLOWED_SERVICES):
        try:
            r.docker.start(service)
            done[service] = "démarré"
        except Exception as exc:
            done[service] = f"erreur : {exc}"
    return done
