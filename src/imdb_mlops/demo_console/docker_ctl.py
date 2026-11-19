"""Pilotage minimal du moteur Docker (socket Unix), réservé à la console de démonstration.

Restrictions volontaires :
- seuls les services de `ALLOWED_SERVICES` peuvent être arrêtés ou démarrés ;
- uniquement dans le projet Compose de la console (lu sur son propre conteneur) ;
- aucune autre opération (pas de suppression, pas d'exécution de commande).
"""

from __future__ import annotations

import json
import os
import socket
import time

import httpx

ALLOWED_SERVICES = frozenset({"mlflow", "redis", "api"})
SOCKET = os.environ.get("DOCKER_SOCKET", "/var/run/docker.sock")


class DockerControlError(RuntimeError):
    pass


class DockerControl:
    def __init__(self, client: httpx.Client | None = None, project: str | None = None):
        self._client = client or httpx.Client(
            transport=httpx.HTTPTransport(uds=SOCKET), base_url="http://docker", timeout=30
        )
        self._project = project

    @property
    def project(self) -> str:
        """Projet Compose de la console : lu sur son propre conteneur (hostname = id)."""
        if self._project is None:
            self._project = os.environ.get("COMPOSE_PROJECT_NAME") or ""
            if not self._project:
                r = self._client.get(f"/containers/{socket.gethostname()}/json")
                r.raise_for_status()
                self._project = r.json()["Config"]["Labels"]["com.docker.compose.project"]
        return self._project

    def _container(self, service: str) -> dict:
        if service not in ALLOWED_SERVICES:
            raise DockerControlError(f"Service {service!r} non autorisé par la console")
        filters = json.dumps(
            {
                "label": [
                    f"com.docker.compose.project={self.project}",
                    f"com.docker.compose.service={service}",
                ]
            }
        )
        r = self._client.get("/containers/json", params={"all": "true", "filters": filters})
        r.raise_for_status()
        found = r.json()
        if not found:
            raise DockerControlError(f"Conteneur du service {service!r} introuvable")
        return found[0]

    def state(self, service: str) -> str:
        """`running (healthy)`, `exited`, etc."""
        c = self._container(service)
        r = self._client.get(f"/containers/{c['Id']}/json")
        r.raise_for_status()
        st = r.json()["State"]
        health = (st.get("Health") or {}).get("Status")
        return f"{st['Status']} ({health})" if health else st["Status"]

    def stop(self, service: str) -> None:
        c = self._container(service)
        r = self._client.post(f"/containers/{c['Id']}/stop", params={"t": 10})
        if r.status_code not in (204, 304):  # 304 : déjà arrêté
            raise DockerControlError(f"Arrêt de {service} refusé ({r.status_code})")

    def start(self, service: str) -> None:
        c = self._container(service)
        r = self._client.post(f"/containers/{c['Id']}/start")
        if r.status_code not in (204, 304):  # 304 : déjà démarré
            raise DockerControlError(f"Démarrage de {service} refusé ({r.status_code})")

    def wait_healthy(self, service: str, timeout_s: float = 180) -> str:
        deadline = time.monotonic() + timeout_s
        state = self.state(service)
        while time.monotonic() < deadline:
            state = self.state(service)
            if "healthy" in state and "unhealthy" not in state:
                return state
            time.sleep(3)
        return state
