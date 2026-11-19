"""Console de démonstration : restrictions de sécurité, page, exécution des scénarios."""

import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from imdb_mlops.demo_console import app as console
from imdb_mlops.demo_console import scenarios
from imdb_mlops.demo_console.docker_ctl import DockerControl, DockerControlError


def _docker(calls):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/containers/json":
            labels = json.loads(request.url.params["filters"])["label"]
            service = labels[1].split("=")[1]
            return httpx.Response(200, json=[{"Id": f"id-{service}"}])
        if request.url.path.endswith("/json"):
            return httpx.Response(
                200, json={"State": {"Status": "running", "Health": {"Status": "healthy"}}}
            )
        return httpx.Response(204)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://docker")
    return DockerControl(client=client, project="demo")


def test_only_allowed_services_can_be_controlled():
    calls = []
    docker = _docker(calls)
    for forbidden in ("postgres", "prefect-server", "worker", "../etc"):
        with pytest.raises(DockerControlError):
            docker.stop(forbidden)
    assert calls == []  # refus avant tout appel au moteur Docker


def test_stop_start_and_state():
    calls = []
    docker = _docker(calls)
    docker.stop("mlflow")
    docker.start("mlflow")
    assert docker.state("redis") == "running (healthy)"
    assert ("POST", "/containers/id-mlflow/stop") in calls
    assert ("POST", "/containers/id-mlflow/start") in calls


@pytest.fixture
def client(monkeypatch):
    calls = []
    console.app.state.runner = console.Runner(docker=_docker(calls), prefect=object())
    yield TestClient(console.app), calls
    console.app.state.runner = None


def test_page_states_presentation_only(client):
    c, _ = client
    html = c.get("/").text
    assert "réservée à la présentation du projet" in html
    assert "ne doit jamais être déployée en" in html


def test_scenarios_listed_and_params_validated(client):
    c, _ = client
    keys = {s["key"] for s in c.get("/api/scenarios").json()}
    assert {"invalid_batch", "mlflow_recovery", "mlflow_durable", "redis_outage"} <= keys
    r = c.post("/api/scenarios/invalid_batch/run", json={"params": {"fault": "rm -rf"}})
    assert r.status_code == 422
    assert c.post("/api/scenarios/inexistant/run", json={}).status_code == 404


def test_one_scenario_at_a_time_and_service_restarted(client, monkeypatch):
    c, calls = client

    def slow_failing(ctx):
        try:
            ctx.docker.stop("redis")
            ctx.log("panne simulée")
            time.sleep(0.5)
            raise RuntimeError("incident inattendu")
        finally:
            ctx.docker.start("redis")

    fake = scenarios.Scenario("fake", "Faux", "", "", [], "", slow_failing)
    monkeypatch.setitem(console.SCENARIOS, "fake", fake)
    job_id = c.post("/api/scenarios/fake/run", json={}).json()["job_id"]
    assert c.post("/api/scenarios/fake/run", json={}).status_code == 409  # déjà en cours
    for _ in range(50):
        job = c.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            break
        time.sleep(0.05)
    assert job["status"] == "failed"
    assert any("panne simulée" in line["msg"] for line in job["logs"])
    assert ("POST", "/containers/id-redis/start") in calls  # redémarré malgré l'erreur
