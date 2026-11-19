# Documentation technique

Fonctionnement détaillé de chaque brique : chaîne de données, entraînement, orchestration,
API, suivi en production, supervision, tests de charge et console de démonstration.
La présentation du projet, l'architecture et le démarrage sont dans le [README](../README.md).

Sommaire :
1. [Chaîne données : ingestion et contrôles qualité](#chaîne-données--ingestion-et-contrôles-qualité)
2. [Entraînement, traçabilité et registre](#entraînement-traçabilité-et-registre)
3. [Orchestration (Prefect)](#orchestration-prefect)
4. [API de prédiction](#api-de-prédiction)
5. [Suivi en production : prédiction vs note réelle (J+1)](#suivi-en-production--prédiction-vs-note-réelle-j1)
6. [Supervision (Prometheus + Grafana)](#supervision-prometheus--grafana)
7. [Tests de charge et dimensionnement](#tests-de-charge-et-dimensionnement)
8. [Console de démonstration (présentation uniquement)](#console-de-démonstration-présentation-uniquement)
9. [Qualité et tests](#qualité-et-tests)
10. [Structure du dépôt](#structure-du-dépôt)

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
([features.py](../src/imdb_mlops/features.py)) sert à l'entraînement et à l'API, et un test
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

**Règle de mise en service** ([registry.py](../src/imdb_mlops/training/registry.py)) : chaque
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

Le service `worker` ([serve.py](../src/imdb_mlops/orchestration/serve.py)) enregistre deux
déploiements, dont les planifications sont versionnées avec le code. L'interface Prefect
(http://localhost:4200) donne l'historique des runs, les logs, les relances et les durées.

| Déploiement | Déclenchement | Rôle |
|---|---|---|
| `watch-source` | toutes les 30 min (`WATCH_INTERVAL_S`) | Ingestion dynamique : si le dépôt source a un nouveau commit, lance la chaîne complète, sinon ne fait rien |
| `daily-pipeline` | à la demande (UI ou CLI) | Données → qualité → **évaluation J+1** → entraînement → décision. Paramètres : `ref`, `fault`, `n_trials`, `train` |

Le déclenchement par surveillance (pull) plutôt que par une heure fixe est voulu : le dépôt
source annonce une mise à jour à 05:00 UTC, mais ses commits ont été observés vers 12:00–12:40
UTC. Les deux déploiements ont `concurrency_limit=1`, pour éviter deux entraînements simultanés.

**Gestion des échecs** ([flows.py](../src/imdb_mlops/orchestration/flows.py)) :

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

## API de prédiction

FastAPI ([app.py](../src/imdb_mlops/api/app.py)), servie par uvicorn. Le nombre de workers, les
CPU, la mémoire et le cache se règlent par variables d'environnement (`API_WORKERS`,
`API_CPUS`, `API_MEMORY`, `CACHE_ENABLED`).

| Endpoint | Rôle |
|---|---|
| `POST /predict` | Note d'un épisode (existant ou futur) d'une série connue |
| `POST /predict/batch` | 1 à 1000 épisodes ; **succès partiel** (éléments en erreur listés à part) |
| `POST /feedback` | Note observée pour une `prediction_id` (évaluation en production) |
| `GET /model` | Version servie, run MLflow, SHA des données, RMSE |
| `GET /health`, `GET /ready` | Liveness ; readiness (modèle chargé, état du cache) |
| `POST /admin/reload` | Force la relecture du champion (sinon automatique toutes les 30 s). Exige l'en-tête `X-Admin-Token` (variable `ADMIN_TOKEN` de `.env`) ; désactivé sans jeton |
| `GET /metrics` | Métriques Prometheus (multi-workers) |

```bash
curl -s localhost:8000/predict -H 'Content-Type: application/json'   -d '{"series_id": "tt0903747", "season": 5, "episode": 16}'
# {"prediction_id": "...", "predicted_rating": 9.354, "model_version": "1", "cached": false, ...}
```

- **Version identifiée** : chaque réponse contient `model_version` et l'en-tête
  `X-Model-Version`. Chaque requête reçoit un `X-Request-ID`, repris dans les erreurs et les
  traces.
- **Validation et erreurs** : schémas pydantic stricts (format `tt…`, bornes, champs inconnus
  refusés) et format d'erreur unique `{error, message, request_id, details}`. Codes :
  `invalid_input` (422), `unknown_series` (404), `prediction_not_found` (404),
  `model_unavailable` (503), `storage_unavailable` (503), `internal_error` (500).
- **Mise en service à chaud** : l'API sert toujours l'alias `champion`. Une promotion ou un
  rollback est pris en compte en 30 s au plus, sans redémarrage.
- **Mode dégradé** : si MLflow est injoignable au démarrage, l'API démarre quand même,
  `/ready` et `/predict` répondent 503, et le chargement est retenté en arrière-plan.
- **Cache Redis** : la clé contient la version du modèle et le SHA des données, ce qui invalide
  le cache automatiquement à chaque changement. Redis est optionnel : timeouts de 50 ms,
  coupe-circuit et sonde en arrière-plan.
- **Traces** (PostgreSQL, écriture asynchrone par lots, sans latence ajoutée) :
  - `predictions` : entrée, note, version, run, données, cache, latence, statut, erreur ;
  - `api_requests` : chemin, code HTTP, durée, erreur, version, worker ;
  - `feedback`.

### Vérifications faites sur l'API réelle

| Cas | Résultat |
|---|---|
| Breaking Bad S5E16 (note IMDb : 9,9) | 9,354, 218 ms au 1er appel, **0,7 ms** au 2e (cache) |
| Appels non cachés | 44 à 70 ms (calcul des features avec pandas) |
| Lot de 3 dont 1 série inconnue | 2 prédictions et 1 erreur `unknown_series`, HTTP 200 |
| Rollback v1 → v2 dans le registre | Version 2 servie, ancien cache ignoré |
| Redis arrêté | Toutes les requêtes en 200 et 50 à 60 ms, `/ready` indique `cache: down`, réactivation automatique au redémarrage |

## Suivi en production : prédiction vs note réelle (J+1)

À chaque nouveau snapshot validé, la tâche `evaluate-predictions` du flow (ou
`python -m imdb_mlops evaluate`) recherche dans ce snapshot la note réelle de chaque
prédiction servie sur un snapshot plus ancien
([outcomes.py](../src/imdb_mlops/monitoring/outcomes.py)) :

- les résultats sont enregistrés dans `prediction_outcomes` (prédit, réel, erreur, version du
  modèle, snapshots) ;
- chaque prédiction est classée **nouvel épisode** (absent du snapshot qui a servi à prédire,
  donc vraie prédiction à l'aveugle) ou **épisode connu** ;
- les épisodes pas encore publiés restent en attente, 30 jours au plus ;
- les métriques live (RMSE, MAE, biais, par version et par type d'épisode, sur 7 jours) sont
  tracées dans l'expérience MLflow `imdb-episode-rating-production` ;
- une alerte critique de **dérive** part si la RMSE live dépasse 0,8 sur au moins 20 prédictions.

Les feedbacks utilisateurs (`/feedback`) offrent une seconde source de vérité terrain.

**Origine du trafic** : chaque prédiction enregistre l'en-tête `X-Traffic-Source` (`user` par
défaut, `simulation` pour le simulateur, `loadtest` pour Locust). Le trafic de charge est
**exclu** de l'évaluation J+1. Sans cette séparation, les 458 566 prédictions des tests de
charge auraient constitué 99,8 % des évaluations du 17/11.

### Rejeu réel J → J+1 (snapshots du 14/11 `514edeb` et du 15/11 `1e02ab7`)

```bash
R="docker compose run --rm pipeline python -m imdb_mlops"
J0=514edebea2f29b258470d743f3161adca556a9c0 ; J1=1e02ab7c027c48274aca22eb04244a849f909ae7
$R data --ref $J0 && $R data --ref $J1                       # les deux snapshots
SERVE_SNAPSHOT_SHA=$J0 docker compose up -d --wait api       # API épinglée sur le jour J
$R simulate -n 400 --api-url http://api:8000 --served-sha $J0 --target-sha $J1
$R evaluate --sha $J1                                        # comparaison au jour J+1
docker compose up -d --wait api                              # retour au dernier snapshot
```

Le « dernier snapshot » est le plus récent par **date des données** (date du commit source),
et non par date d'ingestion : ré-ingérer un ancien commit ne change pas ce que servent l'API
et l'entraînement. Pour servir un snapshot précis, on l'épingle avec `SERVE_SNAPSHOT_SHA`.

| | n | MAE | RMSE | Biais |
|---|---|---|---|---|
| Épisodes connus | 251 | 0,338 | 0,453 | +0,008 |
| **Nouveaux épisodes** | **174** | **0,572** | **0,722** | **−0,073** |
| Ensemble (live) | 425 | 0,434 | 0,578 | −0,025 |
| Rappel : test hors ligne | 88 001 | 0,393 | 0,570 | |

- 82 prédictions portent sur des épisodes futurs et restent en attente. Les feedbacks (62)
  montrent une MAE de 0,451.
- **Lecture** : la RMSE live globale (0,578) est en ligne avec le test (0,570), il n'y a donc
  pas de dérive. En revanche, les nouveaux épisodes sont nettement plus difficiles et
  sous-estimés. Les plus grosses erreurs concernent des épisodes tout juste sortis, notés
  9 à 9,5 par les premiers votants. C'est une limite connue : la note d'un épisode récent est
  instable pendant ses premiers jours.
- Les 37 requêtes en 404 correspondent à des séries entrées dans le top 10 000 le 15/11,
  inconnues de l'API servie avec J. C'est le comportement attendu.

## Supervision (Prometheus + Grafana)

Grafana (http://localhost:3000, consultation sans compte ; admin : `GRAFANA_ADMIN_PASSWORD`)
charge au démarrage trois dashboards versionnés, générés par
[build_dashboards.py](../docker/grafana/build_dashboards.py) dans le dossier « IMDb MLOps ».

| Dashboard | Source | Contenu |
|---|---|---|
| **API : utilisation et performances** | Prometheus (relevé toutes les 5 s) | débit, latence p50/p95/p99 par endpoint, taux d'erreurs 4xx/5xx, codes d'erreur, prédictions par version et cache, taux de hit, note moyenne prédite, rechargements du modèle, file et pertes des traces, alertes actives |
| **Traitements : données, pipeline, modèles** | PostgreSQL | dernier snapshot validé, champion, durée et statut des étapes Prefect, journal des exécutions avec erreurs, snapshots et contrôles qualité en échec, RMSE candidat/champion/baseline, décisions du registre, journal des alertes |
| **Performance en production** | PostgreSQL | RMSE/MAE/biais live, nouveaux épisodes vs connus, par version, distribution des erreurs, plus grosses erreurs, prédictions par statut, durées et codes HTTP (p95), feedbacks |

- **Lecture seule** : Grafana se connecte à PostgreSQL avec un utilisateur `grafana_ro`, qui a
  le droit `SELECT` uniquement ([grafana-readonly.sh](../docker/postgres/grafana-readonly.sh)).
- **Règles d'alerte Prometheus** ([alerts.yml](../docker/prometheus/alerts.yml)), visibles dans
  Grafana et sur `/alerts` :

| Alerte | Condition |
|---|---|
| `ApiDown` | API injoignable depuis 30 s |
| `HighErrorRate` | plus de 5 % de réponses 5xx |
| `HighPredictLatencyP95` | p95 de `/predict` au-dessus de 500 ms |
| `ModelNotLoaded` | aucun modèle chargé |
| `CacheUnavailable` | cache contourné |
| `TracesDropped` | traces non écrites en base |

- **Alertes de la chaîne** : celles du pipeline (qualité, échec de flow, dérive, décisions du
  registre) sont dans la table `alerts`, affichée par le dashboard « Traitements », et
  envoyées au webhook s'il est configuré.

> Pour une base déjà initialisée avant l'ajout de Grafana, créer l'utilisateur en lecture seule :
> `MSYS_NO_PATHCONV=1 docker compose exec postgres sh /docker-entrypoint-initdb.d/20-grafana-readonly.sh`

## Tests de charge et dimensionnement

Outils : [locustfile.py](../loadtests/locustfile.py) (scénario),
[run_benchmarks.py](../loadtests/run_benchmarks.py) (un scénario = redéploiement de l'API avec
une configuration, échauffement, mesure, relevé CPU et RAM),
[campaign.sh](../loadtests/campaign.sh) (campagne complète, environ 45 min) et
[report.py](../loadtests/report.py), qui produit **[RESULTS.md](../reports/loadtests/RESULTS.md)**.

```bash
bash loadtests/campaign.sh && python loadtests/report.py
# ou un scénario seul :
python loadtests/run_benchmarks.py --name essai --workers 2 --cpus 2 --cache false --users 30
```

**Scénario** :
- 85 % de `/predict`, avec des séries tirées selon leur popularité (votes), ce qui donne un
  taux de répétition réaliste pour le cache ;
- 5 % de lots de 20 épisodes, 5 % de `/model`, 5 % d'entrées invalides (422 attendu) ;
- utilisateurs sans pause, pour mesurer la capacité ;
- Locust en 4 processus avec `FastHttpUser`, dans un conteneur du même réseau Docker.

### Démarche : chaque optimisation a été motivée par une mesure

| # | Constat (mesure) | Cause identifiée | Action | Effet mesuré |
|---|---|---|---|---|
| 1 | 7,5 req/s, `/predict` p50 1,6 s (dashboard puis Locust) | features recalculées avec pandas à chaque requête (178 ms) et prédiction via le wrapper scikit-learn (24 ms) | features d'inférence en numpy sur un historique pré-indexé, booster LightGBM appelé directement | features 178 ms → **0,29 ms**, prédiction 24 ms → **0,94 ms**, même conditions : 7,5 → **61,6 req/s** (×8) ; équivalence entraînement/inférence vérifiée par test sur chaque ligne |
| 2 | Conteneur bridé (`nr_throttled`), pics p99 de 5 s | LightGBM ouvre 12 threads OpenMP (cœurs visibles) dans un conteneur limité à 1-2 CPU | `num_threads=1`, `OMP_NUM_THREADS=1` (le parallélisme passe par les workers) | suppression des threads concurrents |
| 3 | 2 ou 4 workers n'apportent rien ; logs d'accès toujours présents | commande YAML repliée : `--workers` et `--no-access-log` **ignorés**, un seul worker effectif | commande en forme liste, vérification des processus dans le conteneur | 1 worker sans log d'accès : 61 → 113 req/s ; mesures antérieures rectifiées dans `results.json` |
| 4 | 1 worker ≈ 2 workers ≈ 110 req/s | le générateur (Locust, 1 processus, client `requests`) plafonnait | `FastHttpUser` et 4 processus Locust | 2 workers : jusqu'à 245 req/s |
| 5 | à 2 workers : connexions coupées, workers « died » | le parent uvicorn tue un worker qui ne répond pas à son ping en 5 s quand il est saturé | `--timeout-worker-healthcheck 30` | 0 échec sur plus de 140 000 requêtes de la campagne |
| 6 | 4 workers : 503 `model_unavailable` pendant 30 s | 56 s par worker pour indexer les 9 912 historiques, et téléchargements simultanés depuis MLflow | indexation vectorisée (tri unique), 5 tentatives de chargement au démarrage | indexation 56 s → **2,2 s**, 0 échec à 4 workers |
| 7 | cache Redis : débit −13 %, p95 +20 % | 2 allers-retours réseau (environ 1,1 ms chacun ici) pour 16 % de hits, alors que le calcul ne coûte plus que 1,5 ms | cache **LRU local** par worker en premier niveau, **Redis désactivé par défaut** | Redis seul : 116 req/s contre 134 sans cache ; LRU local : pas de différence significative (variabilité) |

### Résultats de la campagne (médiane de 3 runs, détail dans RESULTS.md)

| Configuration | Débit médian | /predict p50 | p95 | RAM max |
|---|---|---|---|---|
| 1 worker, 1 CPU | 146 req/s | 160 ms | 490 ms | 465 Mio |
| 2 workers, 2 CPU | 134 req/s | 140 ms | 400 ms | 727 Mio |
| 4 workers, 4 CPU | 179 req/s | 115 ms | 365 ms | 1 275 Mio |
| 2 workers, 2 CPU + Redis | 116 req/s | 190 ms | 490 ms | 850 Mio |

Les latences ci-dessus sont celles d'une saturation volontaire (30 clients sans pause). Pour un
débit réel modéré, la montée en charge (2 workers) donne : **p50 5 ms à 1 client**, p95 82 ms à
5 clients (140 req/s), p95 140 ms à 20 clients (245 req/s), et l'objectif p95 ≤ 300 ms est
dépassé à partir d'environ 40 clients simultanés.

### Recommandation

Hypothèses : un utilisateur actif fait une prédiction toutes les 10 s (0,1 req/s) ; objectif
p95 ≤ 300 ms ; on n'exploite que 70 % de la capacité mesurée.

| Utilisateurs actifs | Charge | Configuration recommandée | RAM à réserver |
|---|---|---|---|
| jusqu'à **500** | ≤ 50 req/s | 1 worker, 1 CPU | 0,75 Go |
| jusqu'à **1 000** | ≤ 100 req/s | **2 workers, 2 CPU** (défaut, et un worker peut tomber sans interrompre le service) | 1,5 Go |
| 1 000 à 1 250 | ≤ 125 req/s | 4 workers, 4 CPU (gain mesuré de seulement +23 % pour 2 fois plus de CPU) | 2 Go |
| au-delà | > 100 req/s | **mise à l'échelle horizontale** : N réplicas de 2 workers / 2 CPU derrière un répartiteur, avec N = ⌈utilisateurs / 1 000⌉ ; PostgreSQL et MLflow partagés | 1,5 Go par réplica |

- **Cache** : Redis ne devient rentable que si une prédiction coûte nettement plus qu'un
  aller-retour réseau (modèle plus lourd, features calculées à la volée). C'était le cas avant
  l'optimisation 1, plus après. Le cache LRU local reste actif : son coût est négligeable
  (environ 10 Mio) et il sert aussi pendant une panne de Redis.
- **Limites de ces mesures** :
  - générateur et API partagent le même poste, sous Docker Desktop/WSL2, où un simple aller-retour
    réseau coûte 1 à 3 ms ;
  - la variabilité d'un run à l'autre atteint ±30 %, d'où les 3 répétitions et l'usage de la
    médiane ;
  - un run sur 3 de la configuration 4 workers a échoué au déploiement ;
  - environ 2 000 traces d'utilisation (≈ 0,3 %) ont été perdues pendant la campagne, sans que la
    cause puisse être établie, car les journaux ont été effacés par les redéploiements ;
  - à re-mesurer sur l'infrastructure cible avant une mise en production.

## Console de démonstration (présentation uniquement)

> ⚠ **Fonctionnalité réservée à la présentation du projet.** Elle ne fait pas partie du
> produit et ne doit jamais être déployée en production. Pour pouvoir arrêter et redémarrer
> des conteneurs, elle reçoit le socket Docker, ce qui lui donne de fait le contrôle de
> l'hôte. Elle n'est démarrée qu'avec le profil Compose `demo` et n'écoute que sur
> `127.0.0.1`.

```bash
docker compose --profile demo up -d demo-console     # http://localhost:8080
docker compose --profile demo stop demo-console      # à arrêter après la présentation
```

Une page web ([demo_console/](../src/imdb_mlops/demo_console/)) lance d'un clic les incidents
contrôlés du [guide de démonstration](DEMO.md). Pour chacun, elle affiche le déroulé en direct
(états Prefect, contrôles bloquants, alertes émises) et des liens vers l'interface où
constater l'effet.

| Scénario | Ce qu'il fait | Résultat attendu |
|---|---|---|
| Lot de données invalide | défaut injecté (choix parmi 5), puis chaîne complète | ingestion en échec, aucune relance, aucun entraînement, 2 alertes critiques |
| MLflow indisponible, puis rétabli | arrêt de MLflow, chaîne lancée, redémarrage pendant le délai de relance | `AwaitingRetry`, puis succès |
| MLflow durablement indisponible | MLflow arrêté pendant tout le flow | 3 tentatives, flow `Failed`, alerte critique |
| Redis indisponible | arrêt de Redis et 10 prédictions | toutes en 200 (Redis n'est pas critique) |
| API arrêtée | arrêt de l'API jusqu'à l'alerte | alerte Prometheus `ApiDown` déclenchée, puis API rétablie |
| Rollback à chaud | `previous_champion` remis en service, API rechargée | autre version servie sans redémarrage (relancer pour revenir) |
| Trafic simulé | 300 requêtes réalistes | dashboards alimentés |

Garde-fous :
- un seul scénario à la fois ; si un `daily-pipeline` réel est déjà en cours (par exemple l'entraînement d'un nouveau commit source), le scénario l'indique dans le journal et attend sa fin ;
- seuls `mlflow`, `redis` et `api` peuvent être arrêtés, et uniquement dans le projet Compose
  courant ;
- chaque service arrêté est redémarré à la fin du scénario, même en cas d'erreur ;
- un bouton « Rétablir les services » permet de tout relancer en cas de besoin ;
- les paramètres sont validés contre une liste fermée.

## Qualité et tests

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -m "not integration"   # tests unitaires (poste Linux/macOS, CI)
docker compose run --rm pipeline pytest -m "not integration"   # mêmes tests, dans l'image
uv run pytest -m integration         # nécessite docker compose up
```

## Structure du dépôt

```
configs/config.yaml          paramètres métier (données, qualité, entraînement, promotion, suivi)
src/imdb_mlops/
  data/                      ingestion, contrôles qualité, injection de défauts, étape données
  features.py                features (entraînement vectorisé + inférence numpy)
  training/                  entraînement traçable, registre et règles de mise en service
  orchestration/             flows et déploiements Prefect
  api/                       FastAPI, chargement du champion, cache, télémétrie
  monitoring/                évaluation J+1, simulateur de trafic
  demo_console/              console de démonstration (présentation uniquement)
  db.py, alerts.py           persistance (PostgreSQL), alertes
  __main__.py                CLI : data, train, models, rollback, reproduce, evaluate, simulate
tests/                       unitaires + intégration (marqueur integration)
loadtests/                   Locust, banc de mesure, campagne, rapport
docker/                      images, init PostgreSQL, Prometheus, Grafana (dashboards générés)
reports/loadtests/           résultats des tests de charge (RESULTS.md, results.json, CSV)
docs/                        documentation technique, guide de démonstration, synthèse OPS, diagramme
```

Les décisions techniques et leur justification, les preuves d'intégration et les problèmes
rencontrés sont dans **[SYNTHESE_OPS.md](SYNTHESE_OPS.md)**.
