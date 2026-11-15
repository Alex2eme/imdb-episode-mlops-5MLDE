# IMDb Episode Rating — projet MLOps

Industrialisation d'un modèle simple qui **prédit la note IMDb d'un épisode de série**.
L'objectif n'est pas la performance du modèle, mais une chaîne opérationnelle complète,
reproductible et supervisée : ingestion → contrôle qualité → entraînement → registre →
API → feedback → supervision.

> Statut : étape 0 (socle et infrastructure) en place. Les sections marquées 🚧 seront
> complétées au fil des étapes.

## Données

- **Source** : [ya0903/imdb-episode-dataset](https://github.com/ya0903/imdb-episode-dataset),
  dépôt public reconstruit chaque jour par GitHub Actions à partir des
  [IMDb Non-Commercial Datasets](https://developer.imdb.com/non-commercial-datasets/)
  (`title.basics`, `title.episode`, `title.ratings`) : top 10 000 séries, épisodes à 10 votes ou plus.
- **Conditions d'utilisation** : les données IMDb sont réservées à un **usage personnel et
  non commercial**, conformément aux conditions d'IMDb. Le dépôt source ne déclare pas de
  licence propre. Ce projet est académique et non commercial, et **aucune donnée n'est
  commitée dans ce dépôt** : elles sont téléchargées à l'exécution.
- **Reproductibilité** : chaque ingestion télécharge l'archive du dépôt source **à un commit
  précis** (SHA). Ce SHA est enregistré avec chaque run d'entraînement, et le champ
  `data.pinned_sha` de [configs/config.yaml](configs/config.yaml) permet de rejouer
  exactement les mêmes données.

## Stack technique

| Besoin | Outil |
|---|---|
| Dépendances, lint, tests | uv (lockfile), ruff, pytest, pre-commit, GitHub Actions |
| Configuration | YAML (métier) + variables d'environnement `.env` (infra, secrets) |
| Qualité des données | Great Expectations |
| Orchestration | Prefect 3 |
| Suivi et registre des modèles | MLflow (métadonnées dans PostgreSQL, artefacts servis par MLflow sur un volume Docker) |
| Modèle | LightGBM + Optuna |
| API | FastAPI + cache Redis |
| Persistance | PostgreSQL |
| Tests de charge | Locust |
| Supervision | Prometheus + Grafana 🚧 |

## Installation

Prérequis : Docker Desktop, [uv](https://docs.astral.sh/uv/), Git.

```bash
cp .env.example .env          # puis remplacer les mots de passe
uv sync                       # installe Python 3.12 et les dépendances figées (uv.lock)
uv run pre-commit install     # hooks ruff avant chaque commit
docker compose up -d --wait   # démarre l'infrastructure
```

> **Windows + Smart App Control** : si `.venv\Scripts\python.exe` est bloqué, recréer
> l'environnement avec le module venv standard, puis relancer `uv sync` :
> `rm -rf .venv && "$(uv python find 3.12)" -m venv .venv && uv sync`
>
> **Windows, encodage console** : exporter `PYTHONUTF8=1` (MLflow affiche des emojis
> que la console cp1252 ne sait pas encoder).

### Services

| Service | URL |
|---|---|
| MLflow | http://localhost:5000 |
| Prefect | http://localhost:4200 |
| PostgreSQL | localhost:5432 (bases `mlflow`, `prefect`, `app`) |
| Redis | localhost:6379 |

## Qualité et tests

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -m "not integration"   # tests unitaires
uv run pytest -m integration         # nécessite docker compose up
```

## Décisions techniques (journal)

- **Python 3.12 figé** (`.python-version`, `requires-python`) : MLflow, Great Expectations et
  LightGBM ne garantissent pas encore le support de 3.14. uv télécharge 3.12 sans toucher au
  Python du système.
- **Ingestion par archive au commit** plutôt que fichier par fichier : un seul téléchargement
  (≈ 5 Mo) au lieu d'environ 10 000, et un SHA immuable qui identifie exactement les données.
- **Pas de MinIO** : les images Docker communautaires de MinIO ne sont plus distribuées
  publiquement. MLflow sert lui-même les artefacts (`--serve-artifacts`, volume
  `mlflow_artifacts`), ce qui supprime un service et un jeu de secrets. Le client n'a besoin
  que de l'URL de MLflow.
- **Jobs serveur MLflow désactivés** (`MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`) : avec les
  workers GenAI activés, le serveur dépassait 768 Mo et redémarrait en boucle. Sans eux, il
  consomme environ 570 Mo.
- **Limites mémoire par conteneur** : Docker ne dispose que d'environ 3,9 Go sur la machine de
  développement. Chaque service est borné pour qu'aucun ne puisse affamer les autres.

## Architecture 🚧

## Démonstration (cas d'usage simulés) 🚧

## Synthèse OPS 🚧
