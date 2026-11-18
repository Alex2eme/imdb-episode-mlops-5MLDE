# Synthèse des choix et résultats OPS

Projet : prédire la note IMDb d'un épisode de série et industrialiser la chaîne complète, de
l'ingestion des données à la supervision du service. Le modèle est volontairement simple
(LightGBM sur l'historique de la série). Ce document justifie les choix techniques, rassemble
les **preuves d'intégration** observées, les résultats des tests de charge, le dimensionnement
recommandé et les limites connues.

Sommaire :
1. [Décisions techniques](#1-décisions-techniques)
2. [Preuves d'intégration](#2-preuves-dintégration-observées-sur-la-plateforme-réelle)
3. [Résultats du modèle](#3-résultats-du-modèle)
4. [Tests de charge et dimensionnement](#4-tests-de-charge-et-dimensionnement)
5. [Incidents rencontrés pendant la construction](#5-incidents-rencontrés-pendant-la-construction-et-corrections)
6. [Limites connues et pistes](#6-limites-connues-et-pistes)

---

## 1. Décisions techniques

### 1.1 Choix d'outils

| Besoin | Choix | Alternatives écartées | Justification |
|---|---|---|---|
| Dépendances | **uv** + `uv.lock`, Python 3.12 figé | pip + requirements, Poetry | Résolution et installation rapides, lockfile multi-plateforme. Python 3.12 car MLflow, Great Expectations et LightGBM ne garantissent pas encore 3.14 |
| Qualité du code | **ruff** (lint + format), pre-commit, pytest, GitHub Actions | flake8 + black + isort | Un seul outil, très rapide, appliqué en local (pre-commit) et en CI |
| Exécution | **Image Docker unique** construite depuis `uv.lock` (pipeline, worker, API, tests) | venv local | Environnement identique partout, et contournement des blocages de DLL par Windows Smart App Control (voir §5) |
| Configuration | YAML (paramètres métier) + `.env` (infra, secrets) validés par **pydantic** | variables seules, Hydra | Configuration séparée du code et validée au chargement ; les secrets sont hors Git |
| Ingestion | **Archive du dépôt source à un commit précis** (API GitHub + codeload) | fichiers bruts un par un | 1 téléchargement de 5 Mo au lieu d'environ 10 000. Le **SHA** identifie les données de façon immuable : base de la reproductibilité |
| Qualité des données | **Great Expectations 1.x** (expectations personnalisées) + règles métier Python | pandera, contrôles manuels | Standard du domaine, rapport détaillé ; les règles inter-snapshots (variation, cohérence avec `meta.json`) s'expriment mieux en Python |
| Orchestration | **Prefect 3** (serveur + worker) | Airflow, cron | Environ 400 Mo contre plusieurs Go pour Airflow (budget Docker de 3,9 Go). Flows en Python testables, relances conditionnelles natives, interface de suivi |
| Suivi et registre | **MLflow 3** (PostgreSQL + artefacts servis par MLflow) | W&B (payant), DVC seul | Gratuit, local, avec registre et alias (`champion`, `challenger`, `previous_champion`) |
| Stockage d'artefacts | **Volume Docker servi par MLflow** (`--serve-artifacts`) | MinIO | Les images communautaires de MinIO ne sont plus distribuées publiquement ; cette solution compte aussi un service et un secret de moins |
| Modèle | **LightGBM** + **Optuna** (trials en parallèle) | réseau de neurones | Rapide, robuste aux valeurs manquantes, parallélisable ; la performance n'est pas l'objet du projet |
| API | **FastAPI** + uvicorn multi-workers | Flask, BentoML | Validation pydantic native, OpenAPI automatique, asynchrone |
| Cache | **LRU local par worker** + Redis optionnel | Redis seul | Choix fondé sur la mesure : Redis seul faisait perdre 13 % de débit (§4) |
| Persistance | **PostgreSQL 16** (bases `app`, `mlflow`, `prefect`) | SQLite | Imposé par l'énoncé ; requêtable par Grafana ; une instance pour trois usages |
| Supervision | **Prometheus** (métriques, règles d'alerte) + **Grafana** (3 dashboards provisionnés) | Evidently seul, ELK | Standard, léger ; dashboards versionnés et générés par du code |
| Tests de charge | **Locust** (FastHttpUser, multi-processus) | k6, JMeter | Scénario écrit en Python, réutilisant les données réelles |

### 1.2 Décisions de conception

- **Les features ne décrivent que l'historique antérieur de la série** (moyennes, tendance,
  position dans la saison…). Le nombre de votes de l'épisode est exclu, car il n'est connu
  qu'après diffusion. Évaluation sur un **split temporel** : les derniers épisodes de chaque
  série servent de test, ce qui correspond au cas d'usage réel.
- **Une même définition des features pour l'entraînement et l'inférence**. Deux
  implémentations existent (pandas vectorisé pour l'entraînement, numpy pour l'API), et un test
  vérifie leur **égalité sur chaque ligne** d'un jeu aléatoire. Les écarts entre entraînement et
  inférence (*training/serving skew*) sont ainsi exclus.
- **`meta.json` fait foi** : le dépôt source ne supprime pas les fichiers des séries sorties du
  top 10 000 (10 185 fichiers pour 9 912 séries déclarées). Ces orphelins sont exclus et comptés.
- **Deux niveaux de sévérité des contrôles** : seuls les défauts qui rendraient le modèle faux
  ou inutilisable interrompent la chaîne (code 2, alerte critique). Les autres sont remontés en
  warning.
- **Relances uniquement pour les erreurs transitoires** (`retry_condition_fn`). Un lot non
  conforme ou un entraînement refusé échoue immédiatement, car relancer donnerait le même
  résultat.
- **Règle de mise en service** : RMSE ≤ plafond, meilleure que la baseline (moyenne des épisodes
  précédents), et meilleure que le champion **réévalué sur le même jeu de test**, d'au moins
  **0,005**. Ce seuil est fixé d'après la mesure : deux tunings sur les mêmes données diffèrent
  d'environ 0,0015, donc en dessous on ne ferait que remplacer le modèle en production sans gain.
- **L'API sert toujours l'alias `champion`** et le relit toutes les 30 s : une promotion ou un
  rollback s'applique sans redémarrage. Le cache est indexé par version et données, donc il ne
  sert jamais une prédiction d'un autre modèle.
- **Les traces sont écrites hors du chemin de réponse** (file bornée, insertion par lots). Une
  base lente ou en panne ne dégrade pas les temps de réponse ; les pertes sont comptées.
- **Les dépendances non critiques ne font pas tomber le service** : Redis est contourné (sonde
  de fond, coupe-circuit) ; si MLflow est absent au démarrage, l'API démarre en 503 et réessaie
  en arrière-plan ; si PostgreSQL est indisponible, les traces sont perdues et comptées, mais
  les prédictions continuent.
- **Limites mémoire et CPU sur chaque conteneur** : aucun service ne peut affamer les autres
  sur un budget Docker de 3,9 Go. L'entraînement est limité à 10 cœurs sur 12, pour préserver
  MLflow et PostgreSQL.
- **Dashboards et règles d'alerte versionnés** (provisioning, `allowUiUpdates: false`).
- **Lecture seule pour Grafana** : l'utilisateur `grafana_ro` n'a que le droit `SELECT`.

---

## 2. Preuves d'intégration (observées sur la plateforme réelle)

Toutes ces exécutions ont eu lieu sur les données réelles du dépôt source, et chacune est
visible dans au moins une interface. Le [guide de démonstration](DEMO.md) permet de les rejouer.

| # | Scénario | Résultat observé | Où le voir |
|---|---|---|---|
| 1 | **Fonctionnement autonome** : nouveau commit source le 17/11 détecté par `watch-source` dès le redémarrage de la machine (conteneurs relancés par `restart: unless-stopped`) | Ingestion et contrôles OK (421 286 épisodes, 24 s), puis évaluation de 948 prédictions passées (RMSE 0,587, pas de dérive), puis entraînement de la v6 (290 s) **correctement rejetée** (0,5680 contre 0,5699 pour le champion, gain < 0,005). **Aucune intervention** | Prefect, Grafana « Traitements » et « Performance en production », MLflow |
| 2 | Ingestion et contrôles sur les données réelles | 18/18 contrôles OK, 421 210 épisodes, 9 912 séries, 47 s | `data_snapshots`, rapport JSON |
| 3 | **Lot invalide** (notes ×10, doublons, troncature…) | Lot rejeté avec le motif précis (21 057 notes hors plage, −70 % de lignes), **aucune relance**, entraînement non lancé, alertes critiques | Prefect (`Failed`), Grafana (contrôles en échec, alertes) |
| 4 | **MLflow indisponible** pendant le flow | Échec `MlflowException`, `AwaitingRetry`, relance 60 s plus tard **réussie** après redémarrage de MLflow, flow `Completed` | Prefect, `pipeline_steps` (tentative en échec, puis succès) |
| 5 | **Redis arrêté** sous trafic | Toutes les requêtes en 200, 50 à 60 ms, `/ready` indique `cache: down`, réactivation automatique | API, métriques `cache_requests_total{result="bypass"}` |
| 6 | Entraînement et registre | v1 promu (RMSE 0,570 contre 0,741 pour la baseline), v2 promu, v3 à v5 rejetés (gain < 0,005) | MLflow (Models), `model_events` |
| 7 | **Rollback** | `rollback` v2 → v1 ; l'API sert la nouvelle version **sans redémarrage**, cache invalidé | `/model`, en-tête `X-Model-Version`, Grafana |
| 8 | **Reproduction d'un run** | Mêmes SHA de données, hyperparamètres et seed : écart de RMSE **0,0** | MLflow (run `reproduce-*`, tag `reproduces_run`) |
| 9 | **Comparaison J+1** (snapshots réels du 14/11 et du 15/11) | 425 prédictions évaluées, RMSE live 0,578 ; nouveaux épisodes 0,722, épisodes connus 0,453 | Grafana « Performance en production », MLflow `…-production` |
| 10 | Validation des entrées et erreurs de l'API | 422 (détail par champ), 404 `unknown_series`, lot partiellement réussi (200 avec liste d'erreurs) | `api_requests`, Grafana |
| 11 | Charge : plus de 140 000 requêtes en campagne | **0 échec**, jusqu'à 245 req/s (2 workers) | `reports/loadtests/RESULTS.md`, Grafana « API » |

---

## 3. Résultats du modèle

| | RMSE | MAE | R² |
|---|---|---|---|
| Moyenne globale | 1,068 | 0,785 | 0,00 |
| Baseline : moyenne des épisodes précédents | 0,741 | 0,516 | 0,52 |
| **LightGBM (champion v1), test hors ligne** | **0,570** | **0,393** | **0,71** |
| En production, J+1 (425 prédictions) | 0,578 | 0,434 | |
| dont **nouveaux épisodes** (174) | 0,722 | 0,572 | |

- Le premier épisode de chaque saison est le segment le plus difficile (RMSE 0,856, contre 0,556
  en cours de saison).
- **En production, les nouveaux épisodes sont sous-estimés** (biais −0,07). Les plus grosses
  erreurs concernent des épisodes tout juste sortis, notés 9 à 9,5 par les premiers votants. La
  note d'un épisode récent est instable pendant ses premiers jours.
- Pas de dérive observée : la RMSE live est en ligne avec le test hors ligne. Le seuil d'alerte
  est fixé à 0,8.

---

## 4. Tests de charge et dimensionnement

Détail complet : [reports/loadtests/RESULTS.md](../reports/loadtests/RESULTS.md) et la section
« Tests de charge » du [README](../README.md#tests-de-charge-et-dimensionnement).

**Avant / après** :

| | Avant | Après |
|---|---|---|
| Débit (même configuration, mêmes conditions) | 7,5 req/s | 61,6 req/s (×8), puis 130 à 245 req/s une fois les problèmes de la plateforme corrigés |
| Latence `/predict` | p50 1,6 s, p95 3,6 s | p50 **5 ms** à faible charge, p95 140 ms à 245 req/s |
| Features par requête | 178 ms (pandas) | 0,29 ms (numpy) |
| Démarrage d'un worker (historiques) | 56 s | 2,2 s |

**Dimensionnement recommandé** (hypothèses : 1 prédiction toutes les 10 s par utilisateur
actif, p95 ≤ 300 ms, 70 % de la capacité mesurée) :

| Utilisateurs actifs | Configuration API | RAM |
|---|---|---|
| ≤ 500 | 1 worker, 1 CPU | 0,75 Go |
| ≤ 1 000 | **2 workers, 2 CPU** (défaut) | 1,5 Go |
| ≤ 1 250 | 4 workers, 4 CPU (gain mesuré +23 % seulement) | 2 Go |
| > 1 000 | N réplicas de 2 workers / 2 CPU, N = ⌈utilisateurs / 1 000⌉, derrière un répartiteur | 1,5 Go par réplica |

Cache : **Redis désactivé par défaut**. Il faisait perdre 13 % de débit pour 16 % de hits, une
fois une prédiction ramenée à 1,5 ms. Le cache LRU local par worker est conservé.

---

## 5. Incidents rencontrés pendant la construction et corrections

Ces incidents ont été diagnostiqués par la mesure, puis corrigés. Ils documentent la démarche
d'exploitation autant que le résultat.

| Symptôme | Diagnostic | Correction |
|---|---|---|
| DLL scipy/pyarrow bloquées sous Windows | Smart App Control bloque, de façon non déterministe, les binaires non signés | Exécution des traitements et des tests dans l'image Linux |
| Images MinIO introuvables | Images communautaires retirées des registres publics | Artefacts servis par MLflow (volume) |
| MLflow redémarre en boucle | Workers de jobs GenAI : 768 Mo saturés | Jobs désactivés (environ 570 Mo) |
| `Connection reset` vers MLflow pendant le tuning | Tuning sur 12 cœurs : serveur affamé ; keep-alive uvicorn de 5 s | Pipeline limité à 10 CPU, keep-alive de 120 s, relances du traçage |
| Expectation GX en échec sans détail | `column_list` en tuple non supporté ; exception imbriquée par GX | Liste ; extraction de l'exception imbriquée dans le rapport |
| Règle de promotion faussée | Version MLflow renvoyée en `int` ou en `str` selon le backend | Normalisation en `str` (détectée par les tests de bout en bout) |
| Table `model_events` absente | Schéma créé seulement par l'étape données | `init_db` idempotent au démarrage de chaque commande |
| Requêtes bloquées 4 à 30 s, Redis arrêté | Résolution DNS d'un conteneur arrêté, non couverte par les timeouts | Sonde de fond et coupe-circuit : aucune requête n'attend |
| 2 ou 4 workers sans effet ; logs d'accès présents | Commande YAML repliée : options uvicorn ignorées | Commande en forme liste ; vérification des processus |
| Pics p99 de 5 s, conteneur bridé | 12 threads OpenMP LightGBM pour 1 à 2 CPU de quota | `num_threads=1`, `OMP_NUM_THREADS=1` |
| Workers « died » sous charge | Healthcheck des workers uvicorn (5 s) dépassé sous saturation | `--timeout-worker-healthcheck 30` |
| 503 au démarrage à 4 workers | Indexation pandas de 56 s par worker et téléchargements simultanés | Indexation vectorisée (2,2 s) et tentatives de chargement |
| Plafond de 110 req/s quelle que soit l'API | Générateur Locust saturé (1 processus, `requests`) | `FastHttpUser` et 4 processus |
| Évaluation J+1 du 17/11 sur 436 231 prédictions | Les 458 566 prédictions des tests de charge étaient traitées comme du trafic de production et noyaient les métriques live | Origine du trafic tracée (`X-Traffic-Source` : `user`, `simulation`, `loadtest`), trafic de charge exclu de l'évaluation, historique étiqueté d'après les fenêtres horaires et évaluations polluées supprimées |

### Relecture du code (18/11) : constats corrigés

| Constat | Correction |
|---|---|
| Après un rollback, Grafana affichait deux versions « servies » : `Gauge.clear()` n'est pas pris en charge par Prometheus en mode multi-processus | L'ancienne version est remise à 0 et les requêtes filtrent `model_info == 1` |
| Un commit dont l'entraînement avait échoué après ses relances était considéré comme traité, donc jamais réentraîné | Un commit n'est traité que si ses données sont rejetées ou si un modèle a été enregistré dessus |
| `watch-source` lançait la chaîne en sous-flow, donc hors de la limite de concurrence : deux entraînements pouvaient tourner en même temps | Lancement via le déploiement `daily-pipeline` (`run_deployment`), soumis à sa limite de 1 |
| Le « dernier snapshot » était choisi par date de validation : ré-ingérer un ancien commit faisait servir de vieilles données | Tri par date des données (commit source) ; épinglage explicite avec `SERVE_SNAPSHOT_SHA` |
| À l'arrêt de l'API, seul un lot de traces était écrit, et le reste était perdu sans être compté | La file est vidée entièrement et tout reliquat est compté dans `db_traces_dropped_total` |
| Ports publiés sur toutes les interfaces, Redis sans mot de passe ; `/admin/reload` ouvert | Publication sur `127.0.0.1` uniquement ; jeton `X-Admin-Token` exigé |
| Une version entraînée avec d'autres features pouvait être chargée sans contrôle | Comparaison avec le tag `features` de la version : chargement refusé et modèle précédent conservé |
| La CI ne construisait pas l'image Docker | Ajout d'un job : validation de `docker compose config`, `docker build`, tests de fumée dans l'image |
| `boto3` inutile (reliquat de MinIO) et code mort | Supprimés |

---

## 6. Limites connues et pistes

- **Mesures de charge sur un poste unique** (Docker Desktop/WSL2) : le générateur et l'API
  partagent les ressources, un aller-retour réseau coûte 1 à 3 ms, et la variabilité d'un run à
  l'autre atteint ±30 % (atténuée par la médiane de 3 runs). À re-mesurer sur l'infrastructure
  cible.
- **Traces perdues pendant la campagne de charge** : environ 2 000 (≈ 0,3 %), cause non
  établie avec certitude, car les journaux ont été effacés par les redéploiements successifs.
  La relecture du code a trouvé une cause plausible, corrigée depuis : à l'arrêt, l'API ne
  vidait qu'un seul lot de sa file de traces et perdait le reste sans le compter. Le compteur
  `db_traces_dropped_total` et l'alerte `TracesDropped` permettent de les détecter.
- **Origine du trafic déclarative** : un client qui omet `X-Traffic-Source` compte comme un
  utilisateur réel. En production, il faudrait réserver la valeur `loadtest` à des clients
  authentifiés.
- **Snapshots manqués quand la machine est éteinte** : `watch-source` ne traite que le dernier
  commit. Le snapshot du 16/11 n'a pas été ingéré. Piste : rattraper les commits intermédiaires
  via l'API GitHub.
- **Séries entrées dans le top 10 000 le jour J** : inconnues de l'API jusqu'au snapshot suivant
  (404). Piste : un modèle de repli sans historique.
- **Nouveaux épisodes sous-estimés** (biais −0,07, RMSE 0,72) : piste, une feature d'âge de
  l'épisode, ou ne prédire qu'une fois un nombre minimal de votes atteint.
- **Alertes** : envoyées par webhook (Discord/Slack) si `ALERT_WEBHOOK_URL` est configuré, et
  toujours en base et dans les logs. Les règles Prometheus ne sont pas reliées à Alertmanager,
  elles sont seulement visibles dans Prometheus et Grafana.
- **Sécurité** : les ports sont publiés sur `127.0.0.1` uniquement et `/admin/reload` exige un
  jeton. L'API de prédiction, MLflow et Prefect restent sans authentification, ce qui n'est
  acceptable qu'en local. Il faudrait un reverse proxy avec authentification et TLS avant
  toute exposition.
- **Tests du flow Prefect** : le serveur de test embarqué de Prefect 3.8 est incompatible avec
  la version de FastAPI du projet. Le test d'orchestration s'exécute donc contre le vrai serveur
  (test d'intégration).
