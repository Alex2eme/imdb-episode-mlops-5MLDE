# Guide de démonstration

Ce guide permet de rejouer le **fonctionnement nominal** et les **incidents contrôlés**, de
bout en bout, depuis un clone du dépôt. Les commandes sont écrites pour bash (Linux, macOS,
Git Bash sous Windows) et se lancent depuis la racine du dépôt.

Durée totale : environ 45 minutes, dont environ 15 de construction et de premier entraînement.

| Interface | URL | Ce qu'on y regarde |
|---|---|---|
| Grafana | http://localhost:3000 (sans compte) | Dossier « IMDb MLOps » : dashboards API, Traitements, Performance en production |
| Prefect | http://localhost:4200 | Runs des flows, tâches, relances, logs |
| MLflow | http://localhost:5000 | Expériences, runs (paramètres, métriques, artefacts), onglet Models (alias) |
| API | http://localhost:8000/docs | Documentation interactive, essais |
| Prometheus | http://localhost:9090/alerts | Règles d'alerte et leur état |

Raccourci utilisé dans ce guide :

```bash
R="docker compose run --rm pipeline python -m imdb_mlops"
```

---

## 0. Installation et démarrage

Prérequis : Docker (avec Compose v2), Git. Prévoir environ 4 Go de RAM pour Docker.

```bash
git clone <url-du-depot> imdb-mlops && cd imdb-mlops
cp .env.example .env            # puis remplacer les mots de passe « change-me »
export GIT_SHA=$(git rev-parse HEAD)
docker compose --profile jobs build        # image applicative (environ 10 min la 1re fois)
docker compose up -d --wait                # 8 services
docker compose ps                          # tous « healthy » ou « running »
```

L'API démarre en mode dégradé (`/ready` en 503) tant qu'aucun modèle n'est en service. C'est
voulu : elle n'échoue pas, elle attend.

## 1. Nominal : données, entraînement, mise en service

```bash
$R data          # ingestion du dernier commit source + contrôles qualité (environ 1 min)
$R train         # entraînement avec 30 trials Optuna en parallèle + décision (environ 5 min)
$R models        # registre : la version 1 porte l'alias « champion »
```

Ce qu'on observe :
- **Terminal** : `19/19 contrôles OK`, puis `PROMU : Aucun champion : RMSE 0.57…`.
- **MLflow, expérience `imdb-episode-rating`** : un run avec 30 runs enfants (un par trial) ;
  - tags `data.sha`, `code.git_sha` et `seed` ;
  - artefacts `plots/` (importance des features, prédit vs réel, résidus, historique Optuna),
    `evaluation/`, `config/` et `data/quality_report.json` ;
  - modèle enregistré avec l'alias `champion`.
- **API** : l'alias est pris en compte en 30 s au plus.

```bash
curl -s localhost:8000/ready
curl -s localhost:8000/model
```

## 2. Orchestration automatique (Prefect)

Le service `worker` surveille le dépôt source toutes les 30 min (`watch-source`) et lance la
chaîne complète à chaque nouveau commit. Pour la déclencher à la main :

```bash
docker compose exec worker prefect deployment run watch-source/watch-source
# « Commit … déjà traité : rien à faire » si aucun nouveau commit

docker compose exec worker prefect deployment run daily-pipeline/daily-pipeline -p n_trials=6
```

Dans **Prefect**, le flow `daily-pipeline` enchaîne trois tâches : `ingest-and-validate`, puis
`evaluate-predictions`, puis `train-and-promote`. Dans **Grafana « Traitements »**, on retrouve
la durée et le statut des étapes et la décision du registre. En général, la nouvelle version
est **non promue** : son gain est inférieur au seuil de 0,005.

## 3. API de prédiction

```bash
J='-H Content-Type:application/json'
# Breaking Bad S5E16 (note IMDb réelle : 9,9) ; 2e appel servi par le cache local
curl -s $J localhost:8000/predict -d '{"series_id":"tt0903747","season":5,"episode":16}'
curl -s $J localhost:8000/predict -d '{"series_id":"tt0903747","season":5,"episode":16}'
# Épisode futur
curl -s $J localhost:8000/predict -d '{"series_id":"tt0903747","season":6,"episode":1}'
# Lot avec un élément en erreur (succès partiel)
curl -s $J localhost:8000/predict/batch -d '{"items":[{"series_id":"tt0944947","season":8,"episode":6},{"series_id":"tt0000000","season":1,"episode":1}]}'
# Erreurs : entrée invalide (422, détail par champ), série inconnue (404)
curl -s $J localhost:8000/predict -d '{"series_id":"breaking-bad","season":-1,"episode":1}'
curl -s $J localhost:8000/predict -d '{"series_id":"tt0000000","season":1,"episode":1}'
# Feedback (remplacer par un prediction_id renvoyé ci-dessus)
curl -s $J localhost:8000/feedback -d '{"prediction_id":"<id>","rating":9.9,"comment":"démo"}'
```

Chaque réponse porte `model_version`, ainsi que les en-têtes `X-Model-Version` et
`X-Request-ID`. Pour alimenter les dashboards avec un trafic réaliste (épisodes nouveaux,
connus et futurs, erreurs, lots, feedbacks) :

```bash
docker compose run --rm pipeline python -m imdb_mlops simulate -n 500 --api-url http://api:8000
```

Ensuite, dans **Grafana « API »**, on voit le débit, la latence p50/p95/p99, les erreurs 4xx,
la version servie et le cache.

## 4. Incident contrôlé n° 1 : lot de données invalide

```bash
$R data --fault ratings_out_of_range ; echo "code retour : $?"     # 2
docker compose exec worker prefect deployment run daily-pipeline/daily-pipeline -p fault=truncated
```

Attendu :
- `LOT REJETÉ — chaîne interrompue`, avec le motif (`expect_valid_imdb_rating : 21 057 valeurs
  hors plage`, ou `row_count_change… -70 %`) ;
- **aucune relance** et **aucun entraînement** ;
- deux alertes critiques (qualité et échec du flow) dans les logs, la table `alerts` et le
  webhook s'il est configuré.

Où le voir :
- **Prefect** : le run est `Failed`, la tâche `ingest-and-validate` aussi, sans nouvelle
  tentative ;
- **Grafana « Traitements »** : « Lots rejetés », « Contrôles en échec », « Journal des alertes ».

Autres défauts disponibles : `duplicates`, `missing_column`, `null_keys`.

## 5. Incident contrôlé n° 2 : dépendance indisponible (MLflow), avec reprise automatique

```bash
docker compose stop mlflow
docker compose exec worker prefect deployment run daily-pipeline/daily-pipeline -p n_trials=2 &
# attendre « Échec train-and-promote (MlflowException) → relance » :
docker compose logs -f worker | grep --line-buffered -m1 "relance"
docker compose start mlflow                 # pendant le délai de relance (60 s)
wait
```

Attendu : la tâche `train-and-promote` passe à `AwaitingRetry` dans Prefect, puis la relance
réussit et le flow se termine en `Completed`. `pipeline_steps` contient une tentative `failed`
(erreur MLflow) suivie d'une tentative `success`. Si MLflow ne revient pas, le flow échoue
après 3 tentatives et une alerte critique part.

Côté API, pendant la panne de MLflow, **les prédictions continuent** : le modèle est déjà en
mémoire, seul le rechargement périodique échoue (`model_reloads_total{result="error"}`).

## 6. Incident contrôlé n° 3 : cache Redis indisponible

Redis est désactivé par défaut (voir les tests de charge). Pour la démonstration, on l'active :

```bash
CACHE_ENABLED=true docker compose up -d --wait api
docker compose stop redis
for e in 1 2 3; do curl -s -o /dev/null -w "HTTP %{http_code} en %{time_total}s\n" \
  -H Content-Type:application/json localhost:8000/predict \
  -d "{\"series_id\":\"tt0108778\",\"season\":4,\"episode\":$e}"; done
curl -s localhost:8000/ready        # "cache":"down" : service dégradé mais disponible
docker compose start redis          # réactivation automatique en quelques secondes
docker compose up -d --wait api     # retour à la configuration par défaut
```

## 7. Rollback et mise en service à chaud

```bash
$R train --trials 2          # crée une nouvelle version (en général rejetée : gain < 0,005)
$R models
$R rollback --version 2 --reason "démo : mise en service manuelle"
curl -s -X POST localhost:8000/admin/reload    # sinon automatique en 30 s au plus
curl -s localhost:8000/model                    # version 2 servie, sans redémarrage
$R rollback --reason "démo : retour au champion précédent"
```

Chaque décision est enregistrée dans les tags de la version (MLflow), dans la table
`model_events` (Grafana « Traitements », colonnes promoted, rejected et rollback) et dans les
alertes.

## 8. Reproduction d'un run

```bash
$R reproduce <RUN_ID>   # run_id visible dans MLflow ou dans la sortie de « train »
# RMSE … | écart avec l'original : 0.00e+00   (code retour 4 si les résultats diffèrent)
```

Le run de reproduction relit les données au même SHA, les mêmes hyperparamètres et la même
seed. Il est tracé dans MLflow (tag `reproduces_run`) mais **n'est pas enregistré dans le
registre**.

## 9. Comparaison J+1 (prédiction vs note réelle)

Rejeu avec deux snapshots réels successifs du dépôt source. Prendre deux SHA consécutifs dans
l'historique des commits du dépôt source ; ceux d'origine du projet sont indiqués ci-dessous.

```bash
J0=514edebea2f29b258470d743f3161adca556a9c0   # 14/11
J1=1e02ab7c027c48274aca22eb04244a849f909ae7   # 15/11
$R data --ref $J0 && curl -s -X POST localhost:8000/admin/reload
docker compose run --rm pipeline python -m imdb_mlops simulate -n 400 \
  --api-url http://api:8000 --served-sha $J0 --target-sha $J1
$R data --ref $J1
$R evaluate     # n_evaluated, n_new_episodes, live_rmse…
```

Où le voir :
- **Grafana « Performance en production »** : RMSE live, nouveaux épisodes vs connus,
  distribution des erreurs, plus grosses erreurs ;
- **MLflow, expérience `imdb-episode-rating-production`** : métriques `live_*`.

## 10. Test de charge (version courte)

```bash
python loadtests/run_benchmarks.py --name demo --workers 2 --cpus 2 --cache false --users 20 --duration 30s
python loadtests/report.py     # met à jour reports/loadtests/RESULTS.md
```

Pendant le test, le dashboard **Grafana « API »** montre le débit et la latence en temps réel.
Le trafic de charge est marqué (`X-Traffic-Source: loadtest`) et **exclu du suivi en
production**. La campagne complète (environ 45 min) se lance avec `bash loadtests/campaign.sh`.

## 11. Qualité du code et tests

```bash
docker compose run --rm pipeline ruff check .
docker compose run --rm pipeline pytest            # unitaires + intégration (infra démarrée)
```

## Arrêt

```bash
docker compose down          # conserve les volumes (données, runs, modèles)
docker compose down -v       # repart de zéro (supprime aussi les volumes)
```
