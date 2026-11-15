"""Alertes : log + webhook optionnel (Discord/Slack) + trace en base pour Grafana.

Une alerte ne doit jamais faire échouer le traitement qui la déclenche : chaque canal
est tenté indépendamment et ses erreurs sont seulement journalisées.
"""

from __future__ import annotations

import json
import logging

import httpx

from imdb_mlops.config import get_settings

log = logging.getLogger("imdb_mlops.alerts")

_LEVELS = {"info": logging.INFO, "warning": logging.WARNING, "critical": logging.CRITICAL}
_ICONS = {"warning": "⚠️", "critical": "🚨"}


def send_alert(
    title: str,
    message: str,
    severity: str = "warning",
    source: str = "pipeline",
    details: dict | None = None,
) -> bool:
    """Émet une alerte. Retourne True si le webhook l'a reçue."""
    details = details or {}
    log.log(_LEVELS.get(severity, logging.WARNING), "[ALERTE %s] %s — %s", severity, title, message)

    delivered = False
    if url := get_settings().alert_webhook_url:
        text = f"{_ICONS.get(severity, '')} **{title}**\n{message}"[:1900]
        try:
            # "content" pour Discord, "text" pour Slack : chacun ignore la clé de l'autre.
            r = httpx.post(url, json={"content": text, "text": text}, timeout=5)
            r.raise_for_status()
            delivered = True
        except httpx.HTTPError as exc:
            log.error("Webhook d'alerte injoignable : %s", exc)

    try:
        from imdb_mlops import db

        db.record_alert(
            severity=severity,
            source=source,
            title=title,
            message=message,
            details=json.loads(json.dumps(details, default=str)),
            delivered=delivered,
        )
    except Exception as exc:  # la base peut être justement la dépendance en panne
        log.error("Alerte non persistée en base : %s", exc)
    return delivered
