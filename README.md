# IMDb Episode Rating — projet MLOps

Industrialisation d'un modèle simple qui **prédit la note IMDb d'un épisode de série**.
L'objectif n'est pas la performance du modèle, mais une chaîne opérationnelle complète,
reproductible et supervisée : ingestion → contrôle qualité → entraînement → registre →
API → feedback → supervision.

> Statut : étapes 0 (socle), 1 (ingestion, qualité), 2 (entraînement, registre) et 3 (orchestration) en place. Les sections marquées 🚧 seront
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
GIT_SHA=$(git rev-parse HEAD) docker compose --profile jobs build pipeline   # SHA du code intégré
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

## Entraînement, traçabilité et registre

```bash
docker compose run --rm pipeline python -m imdb_mlops train              # dernier snapshot validé
docker compose run --rm pipeline python -m imdb_mlops train --sha <SHA>  # snapshot précis
docker compose run --rm pipeline python -m imdb_mlops models             # versions et alias
docker compose run --rm pipeline python -m imdb_mlops rollback           # retour au champion précédent
docker compose run --rm pipeline python -m imdb_mlops reproduce <RUN_ID> # rejoue un run
```

**Modèle** : LightGBM (régression de la note). Les features ne décrivent que **l'historique de
la série avant l'épisode** : moyenne et écart-type des notes précédentes, note de l'épisode
précédent, moyenne glissante sur 3 épisodes, tendance, moyenne de la saison en cours, position
dans la série et dans la saison, popularité de la série. Le nombre de votes de l'épisode
lui-même est exclu, car il n'est connu qu'après diffusion. Une seule fonction
([features.py](src/imdb_mlops/features.py)) sert à l'entraînement et à l'API, et un test
vérifie que les deux produisent exactement les mêmes valeurs.

**Évaluation** : split temporel par série. Les 20 % derniers épisodes de chaque série forment le
test, ce qui correspond au cas d'usage réel : prédire la suite d'une série à partir de son passé.
Référence de comparaison (baseline) : la moyenne des épisodes précédents de la série.

**Parallélisation** : Optuna lance `parallel_trials` trials simultanés, chacun avec
`threads_per_trial` threads LightGBM (4 × 3 = 12 cœurs). Le modèle final utilise tous les cœurs.

**Tracé dans MLflow pour chaque run** :

| Catégorie | Contenu |
|---|---|
| Données | SHA du snapshot, URL source au commit, SHA-256 du Parquet, dataset (`log_input`), rapport qualité |
| Code | SHA Git (`GIT_SHA` intégré à l'image), indicateur de modifications non commitées, version du paquet |
| Paramètres | seed, splits, configuration complète (artefact), hyperparamètres retenus (`hp_*`) |
| Métriques | RMSE/MAE/R² en test, baselines, RMSE par segment (premier épisode de saison, longueur d'historique), courbe d'apprentissage, durées de chaque phase |
| Tuning | un run enfant par trial Optuna (paramètres, RMSE de validation, durée) |
| Artefacts | importance des features, prédit vs réel, résidus, historique Optuna, échantillon de prédictions, modèle avec signature |

**Règle de mise en service** ([registry.py](src/imdb_mlops/training/registry.py)) : chaque
nouvelle version reçoit l'alias `challenger`. Elle devient `champion` seulement si elle respecte
trois conditions : RMSE sous le plafond `max_rmse`, meilleure que la baseline, et meilleure que
le champion actuel **réévalué sur le même jeu de test**, d'au moins `min_improvement`.
L'ancien champion reçoit l'alias `previous_champion`, ce qui rend le rollback immédiat. Chaque
décision est enregistrée de trois façons : tag sur la version, table `model_events` et alerte.

**Reproductibilité** : `reproduce <RUN_ID>` relit le SHA des données et les hyperparamètres
d'un run, puis réentraîne avec la même seed. La commande échoue (code 4) si la RMSE diffère.
Note : avec des trials parallèles, l'ordre des trials Optuna n'est pas déterministe. On
reproduit donc le *modèle* à partir des hyperparamètres retenus, pas le chemin de recherche.

### Résultats du run de référence (snapshot `1e02ab7c02`)

| | RMSE test | MAE | R² |
|---|---|---|---|
| Moyenne globale | 1,068 | 0,785 | 0,00 |
| Baseline (moyenne des épisodes précédents) | 0,741 | 0,516 | 0,52 |
| **LightGBM v1 (30 trials)** | **0,570** | **0,393** | **0,71** |

- **RMSE par segment** : premier épisode de saison 0,856 (le plus difficile), épisodes en
  cours de saison 0,556. Historique court (< 10 épisodes) 0,472, historique long (≥ 50) 0,606 :
  les séries longues dérivent davantage.
- **Durées** (10 cœurs) : features 10 s, tuning de 30 trials 269 s, modèle final 13 s, total 332 s.
- **Décisions du registre observées** :
  - v1 promu (aucun champion) ;
  - v2 promu (gain de 0,0014, avec `min_improvement=0`) ;
  - rollback de v2 vers v1 ;
  - `min_improvement` relevé à 0,005, puis v3 rejeté (gain de 0,0015, considéré comme du bruit).
- **Reproduction** du run v1 : écart de RMSE de 0,0.

## Orchestration (Prefect)

Le service `worker` ([serve.py](src/imdb_mlops/orchestration/serve.py)) enregistre deux
déploiements, dont les planifications sont versionnées avec le code. L'interface Prefect
(http://localhost:4200) donne l'historique des runs, les logs, les relances et les durées.

| Déploiement | Déclenchement | Rôle |
|---|---|---|
| `watch-source` | toutes les 30 min (`WATCH_INTERVAL_S`) | Ingestion dynamique : si le dépôt source a un nouveau commit, lance la chaîne complète, sinon ne fait rien |
| `daily-pipeline` | à la demande (UI ou CLI) | Données → qualité → entraînement → décision. Paramètres : `ref`, `fault`, `n_trials`, `train` |

Le déclenchement par surveillance (pull) plutôt que par une heure fixe est voulu : le dépôt
source annonce une mise à jour à 05:00 UTC, mais ses commits ont été observés vers 12:00–12:40
UTC. Les deux déploiements ont `concurrency_limit=1`, pour éviter deux entraînements simultanés.

**Gestion des échecs** ([flows.py](src/imdb_mlops/orchestration/flows.py)) :

| Type d'erreur | Exemples | Comportement |
|---|---|---|
| Transitoire | réseau, MLflow ou GitHub injoignable | 2 relances (30 s puis 120 s pour les données, 60 s puis 180 s pour l'entraînement) |
| Définitive | `DataQualityError`, `TrainingBlockedError` | aucune relance (inutile), échec immédiat avec le motif |
| Tout échec de flow | | hook `on_failure`/`on_crashed` qui envoie une alerte critique (logs, webhook, table `alerts`) |

Chaque étape est aussi tracée dans la table `pipeline_steps` (durée, statut, erreur), pour
Grafana. Côté worker, le client MLflow échoue rapidement (`MLFLOW_HTTP_REQUEST_MAX_RETRIES=2`),
ce qui laisse à Prefect la politique de relance, visible et paramétrée.

```bash
docker compose up -d --wait                          # inclut le worker
docker compose exec worker prefect deployment run watch-source/watch-source
docker compose exec worker prefect deployment run daily-pipeline/daily-pipeline -p n_trials=6
```

### Incidents contrôlés (résultats observés)

| Scénario | Commande | Résultat |
|---|---|---|
| Lot invalide | `... daily-pipeline -p fault=duplicates` | Tâche des données en `Failed`, **aucune relance**, entraînement non lancé, 2 alertes critiques (qualité et flow) |
| Dépendance indisponible avec reprise | `docker compose stop mlflow`, puis lancement du flow, puis `docker compose start mlflow` pendant le délai de relance | Échec de l'entraînement (`MlflowException`), `AwaitingRetry`, relance 60 s plus tard réussie, flow en `Completed` |
| Dépendance durablement indisponible | `docker compose stop mlflow`, puis lancement du flow | *Attendu, pas encore exécuté* : 3 tentatives (environ 5 min), puis flow en `Failed` et alerte critique |

## Qualité et tests

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -m "not integration"   # tests unitaires (poste Linux/macOS, CI)
docker compose run --rm pipeline pytest -m "not integration"   # mêmes tests, dans l'image
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
- **Le tuning plafonné à 10 cœurs sur 12** : à 12 cœurs, le serveur MLflow, privé de CPU,
  coupait des connexions (*connection reset*) pendant le tuning. Le conteneur `pipeline` est
  limité à 10 CPU (5 trials × 2 threads), le keep-alive d'uvicorn côté MLflow passe à 120 s, et
  le traçage d'un trial est retenté puis ignoré en cas d'échec : un problème de traçage ne fait
  pas échouer un entraînement.
- **Seuil de promotion à 0,005** : deux tunings sur les mêmes données diffèrent d'environ
  0,0015 de RMSE. Promouvoir sur un tel écart remplacerait le modèle en production sans gain réel.
- **Prefect plutôt qu'Airflow** : déploiement léger (un serveur et un worker, environ 400 Mo),
  flows écrits en Python pur et testables, relances conditionnelles natives
  (`retry_condition_fn`). Airflow aurait demandé un scheduler, un webserver et des workers,
  incompatibles avec un budget Docker de 3,9 Go.
- **Features vectorisées** : la moyenne glissante calculée série par série prenait 40 à 50 s
  dans le worker. En version vectorisée, il faut 3,5 s pour 421 000 épisodes.
- **Limites mémoire par conteneur** : Docker ne dispose que d'environ 3,9 Go sur la machine de
  développement. Chaque service est borné pour qu'aucun ne puisse affamer les autres.

## Architecture 🚧

## Démonstration (cas d'usage simulés) 🚧

## Synthèse OPS 🚧
