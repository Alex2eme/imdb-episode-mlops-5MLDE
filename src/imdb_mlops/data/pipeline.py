"""Étape « données » de la chaîne : ingestion → contrôles qualité → traçabilité → alertes.

Fonction pure Python, enveloppée ensuite par l'orchestrateur (Prefect).
"""

from __future__ import annotations

import logging

import pandas as pd

from imdb_mlops import db
from imdb_mlops.alerts import send_alert
from imdb_mlops.config import AppConfig
from imdb_mlops.data.faults import FAULTS, inject_fault
from imdb_mlops.data.ingest import SnapshotManifest, ingest
from imdb_mlops.data.quality import DataQualityError, QualityReport, validate

log = logging.getLogger(__name__)


def _try(action: str, fn, *args, **kwargs):
    """Les traces en base sont utiles mais ne doivent pas masquer le résultat qualité."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        log.warning("%s impossible (base indisponible ?) : %s", action, exc)
        return None


def run_data_step(
    cfg: AppConfig, ref: str | None = None, fault: str | None = None
) -> tuple[pd.DataFrame, SnapshotManifest, QualityReport]:
    _try("Initialisation de la base", db.init_db)
    df, manifest = ingest(cfg, ref)

    if fault:
        log.warning("Injection du défaut %r : %s", fault, FAULTS[fault])
        df = inject_fault(df, fault, seed=cfg.project.seed)
        # Identifiant distinct : le lot corrompu ne doit pas écraser le snapshot sain.
        manifest.sha = f"{manifest.sha}-fault-{fault}"
        manifest.n_rows = len(df)

    previous = _try(
        "Lecture du snapshot de référence", db.last_valid_snapshot, exclude_sha=manifest.sha
    )
    report = validate(
        df,
        manifest,
        cfg,
        previous_n_rows=previous["n_rows"] if previous else None,
        declared_ep_count=manifest.declared_ep_count,
        raise_on_failure=False,
    )
    _try("Enregistrement du snapshot", db.upsert_snapshot, manifest.to_dict(), report.to_dict())

    details = {"sha": manifest.sha, "n_rows": manifest.n_rows}
    if report.warnings:
        send_alert(
            "Qualité des données : avertissements",
            report.summary(),
            severity="warning",
            source="data_quality",
            details=details | {"warnings": [c.name for c in report.warnings]},
        )
    if not report.success:
        send_alert(
            "Qualité des données : lot rejeté, chaîne interrompue",
            report.summary(),
            severity="critical",
            source="data_quality",
            details=details | {"failed": [c.name for c in report.failed]},
        )
        raise DataQualityError(report)

    log.info("Lot conforme : %s", report.summary())
    return df, manifest, report
