# Synthèse des choix et résultats OPS

Projet : prédire la note IMDb d'un épisode de série et industrialiser la chaîne complète, de
l'ingestion des données à la supervision du service. Le modèle est volontairement simple
(LightGBM sur l'historique de la série). Ce document justifie les choix techniques, rassemble
les **preuves d'intégration** observées, les résultats des tests de charge, le dimensionnement
recommandé et les problèmes rencontrés pendant la construction.

Sommaire :
1. [Décisions techniques](#1-décisions-techniques)
2. [Preuves d'intégration](#2-preuves-dintégration-observées-sur-la-plateforme-réelle)
3. [Résultats du modèle](#3-résultats-du-modèle)
4. [Tests de charge et dimensionnement](#4-tests-de-charge-et-dimensionnement)
5. [Problèmes rencontrés](#5-problèmes-rencontrés)

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
| 12 | **Nouveau commit source du 19/11** (`9799ccdde3`) traité seul | Détecté par `watch-source` à 14:26 et lancé via le déploiement `daily-pipeline` : contrôles OK (421 483 épisodes, 15 s), 510 prédictions de la veille évaluées (RMSE live 0,563), entraînement de la v9 (30 min) **rejetée** (0,5695 contre 0,5700, gain < 0,005) | Prefect, Grafana « Traitements », MLflow |
| 13 | **Console de démonstration** : les 7 scénarios lancés depuis la page web | Lot invalide rejeté (doublons, colonne manquante), MLflow rétabli (`Completed`) ou durablement absent (3 tentatives, `Failed`, alerte critique), Redis contourné (10/10 en 200), `ApiDown` en `firing` en 55 s, rollback aller-retour v1 → v2 → v1, 300 requêtes simulées | Console, Prefect, Prometheus, Grafana |

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
« Tests de charge » de la [documentation technique](DOCUMENTATION_TECHNIQUE.md#tests-de-charge-et-dimensionnement).

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

## 5. Problèmes rencontrés

Chaque problème a été diagnostiqué par la mesure ou par les journaux, puis corrigé. Les
derniers ont été trouvés en relisant le code (18/11), en rejouant la démonstration sur un clone
vierge (18/11) et en utilisant la console de démonstration (19/11).

| Problème | Cause | Solution |
|---|---|---|
| DLL scipy/pyarrow bloquées sous Windows | Smart App Control bloque, de façon non déterministe, les binaires non signés | Traitements et tests exécutés dans l'image Linux |
| Images MinIO introuvables | Images communautaires retirées des registres publics | Artefacts servis par MLflow sur un volume |
| MLflow redémarre en boucle | Workers de jobs GenAI : 768 Mo saturés | Jobs désactivés (environ 570 Mo) |
| `Connection reset` vers MLflow pendant le tuning | Serveur affamé par un tuning sur 12 cœurs ; keep-alive uvicorn de 5 s | Pipeline limité à 10 CPU, keep-alive de 120 s, relances du traçage |
| Expectation GX en échec sans détail | `column_list` en tuple non supporté ; exception imbriquée par GX | Liste ; exception imbriquée extraite dans le rapport |
| Règle de promotion faussée | Version MLflow renvoyée en `int` ou en `str` selon le backend | Normalisation en `str` |
| Requêtes bloquées 4 à 30 s quand Redis est arrêté | Résolution DNS d'un conteneur arrêté, non couverte par les timeouts | Sonde de fond et coupe-circuit : aucune requête n'attend |
| 2 ou 4 workers d'API sans effet | Commande YAML repliée : options uvicorn ignorées | Commande en forme de liste |
| Pics p99 de 5 s | 12 threads OpenMP LightGBM pour 1 à 2 CPU de quota | `num_threads=1`, `OMP_NUM_THREADS=1` |
| Workers d'API « died » sous charge | Healthcheck des workers uvicorn (5 s) dépassé sous saturation | `--timeout-worker-healthcheck 30` |
| 503 au démarrage à 4 workers | Indexation pandas de 56 s par worker | Indexation vectorisée (2,2 s) |
| Plafond de 110 req/s quelle que soit l'API | Générateur Locust saturé | `FastHttpUser` et 4 processus |
| Métriques J+1 noyées (436 231 prédictions évaluées) | Le trafic des tests de charge était traité comme du trafic de production | Origine du trafic tracée (`X-Traffic-Source`) et trafic de charge exclu |
| Deux versions « servies » dans Grafana après un rollback | `Gauge.clear()` non pris en charge par Prometheus en multi-processus | Ancienne version remise à 0, filtre `model_info == 1` |
| Un commit dont l'entraînement avait échoué n'était jamais réentraîné | Il était considéré comme traité dès ses données validées | Traité seulement si les données sont rejetées ou si un modèle a été enregistré |
| Deux entraînements pouvaient tourner en même temps | `watch-source` lançait la chaîne en sous-flow, hors de la limite de concurrence | Lancement via le déploiement `daily-pipeline` (limite de 1) |
| Ré-ingérer un ancien commit faisait servir de vieilles données | « Dernier snapshot » choisi par date de validation | Tri par date du commit source ; épinglage avec `SERVE_SNAPSHOT_SHA` |
| Traces perdues à l'arrêt de l'API | Un seul lot de la file était écrit | File vidée entièrement, reliquat compté dans `db_traces_dropped_total` |
| Services exposés sur le réseau, `/admin/reload` ouvert | Ports publiés sur toutes les interfaces | Ports sur `127.0.0.1` ; jeton `X-Admin-Token` |
| Un modèle entraîné avec d'autres features pouvait être chargé | Aucun contrôle au chargement | Comparaison avec le tag `features` : chargement refusé |
| Le banc de charge ne relevait pas le CPU dans un clone | Nom de conteneur propre au poste de développement | Conteneur résolu par `docker compose ps -q api` |
| Requêtes avec accents refusées sous Git Bash Windows | `curl` n'y encode pas les arguments en UTF-8 | Exemples sans accent ; erreurs HTTP au format unique |
| Une tâche d'entraînement « bloquée » 1 h 38 | La machine était en veille (aucun relevé Prometheus ni log PostgreSQL) | Aucune : la relance a fonctionné au réveil |
| `watch-source` arrêté depuis le 18/11 (« *at capacity* ») | Un run resté « Running » depuis un redémarrage de la machine occupait la limite de concurrence | Au démarrage du worker, runs orphelins marqués `Crashed` et exécutions en retard annulées |
| Un scénario de la console semblait bloqué, puis « réussi » à l'état `SCHEDULED` | Il attendait derrière un entraînement réel, sans que la console l'affiche | La console signale l'exécution en cours et attend sa fin ; un dépassement de délai est un échec |
| `watch-source` en échec alors que son entraînement réussissait | Serveur Prefect saturé pendant l'entraînement : réponse `500` passagère | Attente tolérante aux erreurs passagères (`watch-source` et console) |
| Le même commit entraîné deux fois | Pendant l'entraînement, le snapshot validé sans modèle encore enregistré était pris pour un échec à reprendre | `watch-source` ne lance rien tant qu'un `daily-pipeline` est en cours |
