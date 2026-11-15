"""Registre MLflow : versions, alias, règles de mise en service et retour arrière.

Alias utilisés :
- `champion` : version servie par l'API ;
- `challenger` : dernière version entraînée, en attente ou rejetée ;
- `previous_champion` : champion précédent, cible par défaut d'un rollback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from imdb_mlops.alerts import send_alert
from imdb_mlops.config import AppConfig, PromotionConfig

log = logging.getLogger(__name__)

CHAMPION, CHALLENGER, PREVIOUS = "champion", "challenger", "previous_champion"


@dataclass
class PromotionDecision:
    promote: bool
    reason: str


def decide_promotion(
    rmse: float,
    baseline_rmse: float,
    champion_rmse: float | None,
    rules: PromotionConfig,
) -> PromotionDecision:
    """Règle de mise en service (fonction pure, testée unitairement)."""
    if rmse > rules.max_rmse:
        return PromotionDecision(False, f"RMSE {rmse:.4f} > plafond {rules.max_rmse}")
    if rules.must_beat_baseline and rmse >= baseline_rmse:
        return PromotionDecision(
            False, f"RMSE {rmse:.4f} ne bat pas la baseline ({baseline_rmse:.4f})"
        )
    if champion_rmse is None:
        return PromotionDecision(True, f"Aucun champion : RMSE {rmse:.4f} conforme aux règles")
    if rmse <= champion_rmse - rules.min_improvement:
        return PromotionDecision(
            True,
            f"RMSE {rmse:.4f} ≤ champion {champion_rmse:.4f} - {rules.min_improvement}",
        )
    return PromotionDecision(
        False,
        f"RMSE {rmse:.4f} n'améliore pas le champion ({champion_rmse:.4f}) "
        f"d'au moins {rules.min_improvement}",
    )


def get_alias_version(client: MlflowClient, name: str, alias: str):
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def _record(event: str, **values) -> None:
    try:
        from imdb_mlops import db

        db.record_model_event(event=event, **values)
    except Exception as exc:
        log.warning("Événement modèle non enregistré en base : %s", exc)


def apply_decision(
    cfg: AppConfig,
    version: str,
    run_id: str,
    rmse: float,
    baseline_rmse: float,
    champion_rmse: float | None,
    data_sha: str,
) -> PromotionDecision:
    """Pose les alias et tags selon la règle de promotion et trace la décision."""
    client = MlflowClient()
    name = cfg.training.registered_model_name
    decision = decide_promotion(rmse, baseline_rmse, champion_rmse, cfg.promotion)
    now = datetime.now(UTC).isoformat()
    common = dict(
        model_name=name,
        version=version,
        run_id=run_id,
        data_sha=data_sha,
        rmse=rmse,
        champion_rmse=champion_rmse,
        baseline_rmse=baseline_rmse,
    )

    client.set_registered_model_alias(name, CHALLENGER, version)
    client.set_model_version_tag(name, version, "decision_reason", decision.reason)
    client.set_model_version_tag(name, version, "decided_at", now)
    _record("registered", **common, reason="nouvelle version entraînée")

    if decision.promote:
        current = get_alias_version(client, name, CHAMPION)
        if current is not None and str(current.version) != str(version):
            client.set_registered_model_alias(name, PREVIOUS, current.version)
            client.set_model_version_tag(name, current.version, "status", "archived")
        client.set_registered_model_alias(name, CHAMPION, version)
        client.set_model_version_tag(name, version, "status", "champion")
        client.set_model_version_tag(name, version, "promoted_at", now)
        _record("promoted", **common, reason=decision.reason)
        send_alert(
            f"Modèle {name} v{version} mis en service",
            decision.reason,
            severity="info",
            source="model_registry",
            details={"version": version, "rmse": rmse},
        )
    else:
        client.set_model_version_tag(name, version, "status", "rejected")
        _record("rejected", **common, reason=decision.reason)
        send_alert(
            f"Modèle {name} v{version} non promu",
            decision.reason,
            severity="warning",
            source="model_registry",
            details={"version": version, "rmse": rmse},
        )
    log.info(
        "Décision v%s : %s — %s",
        version,
        "PROMU" if decision.promote else "REJETÉ",
        decision.reason,
    )
    return decision


def rollback(cfg: AppConfig, to_version: str | None = None, reason: str = "") -> str:
    """Remet en service une version antérieure (par défaut : `previous_champion`)."""
    client = MlflowClient()
    name = cfg.training.registered_model_name
    current = get_alias_version(client, name, CHAMPION)
    if to_version is None:
        previous = get_alias_version(client, name, PREVIOUS)
        if previous is None:
            raise RuntimeError("Aucun `previous_champion` : préciser --version")
        to_version = str(previous.version)
    client.get_model_version(name, to_version)  # lève si la version n'existe pas
    if current is not None and str(current.version) == str(to_version):
        raise RuntimeError(f"La version {to_version} est déjà champion")

    client.set_registered_model_alias(name, CHAMPION, to_version)
    client.set_model_version_tag(name, to_version, "status", "champion")
    client.set_model_version_tag(name, to_version, "rolled_back_at", datetime.now(UTC).isoformat())
    if current is not None:
        client.set_registered_model_alias(name, PREVIOUS, current.version)
        client.set_model_version_tag(name, current.version, "status", "rolled_back")
    msg = (
        f"Champion : v{current.version if current else '-'} → v{to_version}. "
        f"{reason or 'Rollback manuel'}"
    )
    _record("rollback", model_name=name, version=to_version, reason=msg)
    send_alert(
        f"Rollback du modèle {name}",
        msg,
        severity="warning",
        source="model_registry",
        details={"to_version": to_version},
    )
    log.info(msg)
    return to_version


def list_versions(cfg: AppConfig) -> list[dict]:
    client = MlflowClient()
    name = cfg.training.registered_model_name
    rows = []
    for found in client.search_model_versions(f"name='{name}'"):
        # La recherche ne renvoie pas les alias : relecture de chaque version.
        mv = client.get_model_version(name, str(found.version))
        rows.append(
            {
                "version": int(mv.version),
                "aliases": ",".join(mv.aliases),
                "status": mv.tags.get("status", ""),
                "rmse": mv.tags.get("rmse", ""),
                "data_sha": mv.tags.get("data_sha", "")[:10],
                "git_sha": mv.tags.get("git_sha", "")[:10],
                "created": datetime.fromtimestamp(mv.creation_timestamp / 1000, UTC).strftime(
                    "%Y-%m-%d %H:%M"
                ),
                "reason": mv.tags.get("decision_reason", ""),
            }
        )
    return sorted(rows, key=lambda r: r["version"], reverse=True)
