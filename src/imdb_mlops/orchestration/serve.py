"""Processus worker : enregistre les déploiements Prefect et exécute leurs runs.

Lancé par le service `worker` de docker-compose. Les planifications sont définies ici
(versionnées avec le code) et visibles/modifiables dans l'interface Prefect.
"""

from __future__ import annotations

import logging
import os

from prefect import serve

from imdb_mlops import db
from imdb_mlops.orchestration.flows import daily_pipeline, watch_source

WATCH_INTERVAL_S = int(os.environ.get("WATCH_INTERVAL_S", "1800"))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        db.init_db()
    except Exception as exc:
        logging.warning("Base indisponible au démarrage (%s) : tables créées plus tard", exc)

    watch = watch_source.to_deployment(
        name="watch-source",
        interval=WATCH_INTERVAL_S,
        description="Toutes les 30 min : nouveau commit du dépôt source → chaîne complète.",
        tags=["auto", "ingestion"],
        concurrency_limit=1,
    )
    pipeline = daily_pipeline.to_deployment(
        name="daily-pipeline",
        description="Chaîne complète à la demande (paramètres : ref, fault, n_trials, train).",
        tags=["manual"],
        concurrency_limit=1,
    )
    serve(watch, pipeline, limit=2)


if __name__ == "__main__":
    main()
