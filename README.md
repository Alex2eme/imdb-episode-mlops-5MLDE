# IMDb Episode Rating — projet MLOps

Industrialisation d'un modèle simple qui **prédit la note IMDb d'un épisode de série**.
L'objectif n'est pas la performance du modèle, mais une chaîne opérationnelle complète,
reproductible et supervisée : ingestion → contrôle qualité → entraînement → registre →
API → feedback → supervision.

> Statut : étapes 0 (socle, infrastructure) et 1 (ingestion, contrôles qualité) en place. Les sections marquées 🚧 seront
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

## Chaîne données : ingestion et contrôles qualité

Les traitements s'exécutent dans l'image applicative (`Dockerfile`), construite depuis
`uv.lock` : le même environnement sert au pipeline, au worker Prefect et à l'API.

```bash
docker compose --profile jobs build pipeline
docker compose run --rm pipeline python -m imdb_mlops data            # dernier snapshot
docker compose run --rm pipeline python -m imdb_mlops data --ref <SHA> # snapshot figé
```

1. **Ingestion** : résolution du commit (API GitHub), téléchargement de l'archive à ce commit
   (mise en cache par SHA), aplatissement en Parquet (une ligne par épisode) et manifeste JSON
   (SHA, dates, volumes, empreinte SHA-256 du Parquet).
2. **Contrôles qualité** (Great Expectations et règles métier) : un rapport JSON est écrit dans
   `data/reports/` et le snapshot est tracé dans la table `data_snapshots`.

| Niveau | Contrôles |
|---|---|
| Critique (bloque) | schéma exact ; pas de valeurs nulles sur les clés ; note entre 1 et 10 *(custom)* ; format `tt…` *(custom)* ; clé (série, saison, épisode) unique *(custom)* ; votes ≥ seuil source *(custom)* ; bornes saison/épisode ; volume de lignes ; nombre de séries ; **variation ≤ 20 % par rapport au dernier snapshot validé** ; **cohérence avec le nombre d'épisodes déclaré par `meta.json`** |
| Warning (remonte) | moyenne des notes plausible ; titres présents ; ratio de fichiers orphelins |

3. **En cas d'échec critique** : alerte `critical` (logs, webhook si `ALERT_WEBHOOK_URL`, table
   `alerts`), puis arrêt avec le code de sortie 2. Aucun entraînement n'a lieu sur ce lot.

### Incident contrôlé : lot invalide

```bash
docker compose run --rm pipeline python -m imdb_mlops data --fault ratings_out_of_range
docker compose run --rm pipeline python -m imdb_mlops data --fault truncated
```

Défauts disponibles : `ratings_out_of_range`, `duplicates`, `missing_column`, `truncated`,
`null_keys`. Le lot corrompu est enregistré sous l'identifiant `<sha>-fault-<type>`, ce qui
n'écrase pas le snapshot sain.

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
- **Exécution conteneurisée des traitements** : sur le poste Windows de développement, Smart
  App Control bloque des DLL non signées de manière non déterministe (scipy, pyarrow). Les
  traitements tournent donc dans une image Linux unique, construite depuis `uv.lock`. C'est
  aussi la cible de production. Le `.venv` local sert au lint et aux tests unitaires.
- **`meta.json` fait foi** : le dépôt source ne supprime pas les fichiers des séries sorties du
  top 10 000 (10 185 fichiers pour 9 912 séries déclarées). Ces orphelins sont exclus et
  comptés (contrôle de type warning).
- **Deux niveaux de sévérité** : tout écart ne justifie pas d'arrêter la production. Seuls les
  défauts qui rendraient le modèle faux ou inutilisable bloquent la chaîne.
- **Limites mémoire par conteneur** : Docker ne dispose que d'environ 3,9 Go sur la machine de
  développement. Chaque service est borné pour qu'aucun ne puisse affamer les autres.

## Architecture 🚧

## Démonstration (cas d'usage simulés) 🚧

## Synthèse OPS 🚧
