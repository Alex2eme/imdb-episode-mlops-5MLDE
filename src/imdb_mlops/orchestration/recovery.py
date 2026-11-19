"""Nettoyage des runs orphelins au démarrage du worker.

Constaté le 19/11 : après un redémarrage de la machine, un run `watch-source` est resté
« Running » côté serveur depuis le 16/11 (le processus qui l'exécutait avait disparu). La
limite de concurrence du déploiement (1) étant occupée par ce run fantôme, les exécutions
planifiées suivantes ne démarraient plus : la surveillance du dépôt source était arrêtée.

Ce worker est le seul exécutant des déploiements du projet : au moment où il démarre,
aucun de leurs runs ne peut être réellement en cours. Ceux marqués Running/Pending sont
donc orphelins (passés à Crashed), et les exécutions planifiées en retard sont annulées
(la planification en recrée).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

log = logging.getLogger(__name__)

DEPLOYMENTS = ("watch-source/watch-source", "daily-pipeline/daily-pipeline")


def plan_recovery(runs, now: datetime, late_after: timedelta) -> tuple[list, list]:
    """Sépare les runs orphelins (à marquer Crashed) des runs planifiés en retard (à annuler).

    Fonction pure : `runs` est une liste d'objets avec `.state.type.value` et
    `.expected_start_time`.
    """
    orphans, late = [], []
    for run in runs:
        state = run.state.type.value if run.state else None
        if state in {"RUNNING", "PENDING"}:
            orphans.append(run)
        elif (
            state == "SCHEDULED"
            and run.expected_start_time
            and (now - run.expected_start_time > late_after)
        ):
            late.append(run)
    return orphans, late


def recover_orphan_runs(late_after: timedelta = timedelta(minutes=10)) -> dict[str, int]:
    from prefect.client.orchestration import get_client
    from prefect.client.schemas.filters import (
        DeploymentFilter,
        DeploymentFilterId,
        FlowRunFilter,
        FlowRunFilterState,
        FlowRunFilterStateType,
    )
    from prefect.client.schemas.objects import StateType
    from prefect.states import Cancelled, Crashed

    with get_client(sync_client=True) as client:
        ids = []
        for name in DEPLOYMENTS:
            try:
                ids.append(client.read_deployment_by_name(name).id)
            except Exception:  # premier démarrage : déploiement pas encore créé
                continue
        if not ids:
            return {"crashed": 0, "cancelled": 0}
        runs = client.read_flow_runs(
            deployment_filter=DeploymentFilter(id=DeploymentFilterId(any_=ids)),
            flow_run_filter=FlowRunFilter(
                state=FlowRunFilterState(
                    type=FlowRunFilterStateType(
                        any_=[StateType.RUNNING, StateType.PENDING, StateType.SCHEDULED]
                    )
                )
            ),
            limit=200,  # maximum accepté par l'API Prefect
        )
        orphans, late = plan_recovery(runs, datetime.now(UTC), late_after)
        for run in orphans:
            client.set_flow_run_state(
                run.id,
                Crashed(
                    message="Run orphelin : le worker qui l'exécutait a été arrêté "
                    "(redémarrage de la machine). Nettoyé au démarrage du worker."
                ),
                force=True,
            )
            log.warning("Run orphelin %s (%s) → Crashed", run.name, run.state.type.value)
        for run in late:
            client.set_flow_run_state(
                run.id,
                Cancelled(message="Exécution planifiée en retard, annulée au démarrage du worker"),
                force=True,
            )
        if late:
            log.info("%d exécutions planifiées en retard annulées", len(late))
    return {"crashed": len(orphans), "cancelled": len(late)}
