# Résultats des tests de charge

Générés par `loadtests/report.py` à partir de `results.json` (campagne `loadtests/campaign.sh`). Plateforme : Docker Desktop / WSL2, poste 12 cœurs ; Locust (4 processus, FastHttpUser) dans un conteneur du même réseau Docker. Scénario : 85 % `/predict` (séries tirées selon leur popularité), 5 % lots de 20, 5 % `/model`, 5 % entrées invalides ; utilisateurs sans temps de pause (mesure de capacité).

## 1. Comparaison de configurations (30 utilisateurs simultanés, médiane de 3 runs)

| Configuration | Workers | CPU | Cache | Débit médian (req/s) | Débits des runs | /predict p50 | p95 | p99 | Échecs | CPU moyen | RAM max | Hit cache |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w1c1 | 1 | 1 | aucun | **145.5** | 150, 146, 83 | 160 ms | 490 ms | 1 300 ms | 0/18300 | 70 % | 465 Mio | - |
| w2c2 | 2 | 2 | aucun | **134.2** | 134, 178, 131 | 140 ms | 400 ms | 730 ms | 0/22709 | 130 % | 727 Mio | - |
| w2c2-lru | 2 | 2 | LRU local | **98.3** | 182, 98, 86 | 190 ms | 440 ms | 630 ms | 0/18850 | 142 % | 719 Mio | - |
| w2c2-redis | 2 | 2 | Redis | **116.1** | 122, 116, 82 | 190 ms | 490 ms | 1 300 ms | 0/15076 | 128 % | 850 Mio | 16.2 % |
| w4c4 | 4 | 4 | aucun | **179.2** | 195, 163 | 115 ms | 365 ms | 700 ms | 0/17437 | 228 % | 1 275 Mio | - |

## 2. Montée en charge (2 workers, 2 CPU, sans cache)

| Utilisateurs simultanés | Débit (req/s) | /predict p50 | p95 | p99 | CPU API | Échecs | p95 ≤ 300 ms |
|---|---|---|---|---|---|---|---|
| 1 | 120.5 | 5 ms | 19 ms | 39 ms | 57 % | 0 | oui |
| 5 | 139.9 | 24 ms | 82 ms | 150 ms | 124 % | 0 | oui |
| 10 | 221.2 | 36 ms | 87 ms | 120 ms | 161 % | 0 | oui |
| 20 | 245.4 | 67 ms | 140 ms | 190 ms | 162 % | 0 | oui |
| 40 | 204.7 | 150 ms | 380 ms | 830 ms | 152 % | 0 | non |
| 80 | 265.2 | 270 ms | 550 ms | 710 ms | 165 % | 0 | non |

## 3. Recommandation de dimensionnement

Hypothèses : objectif p95 `/predict` ≤ 300 ms ; on n'exploite que 70 % du débit mesuré (marge pour les pics, le GC et les rechargements de modèle) ; un utilisateur actif fait une prédiction toutes les 10 s (0.1 req/s).

- Point de fonctionnement mesuré respectant l'objectif (2 workers / 2 CPU) : **245 req/s** à 20 utilisateurs simultanés (p95 140 ms).

| Configuration API | Débit médian mesuré | Débit utilisable (x 0.7) | Utilisateurs actifs supportés | RAM à réserver (pic x 1,5) |
|---|---|---|---|---|
| 1 worker(s), 1 CPU | 146 req/s | 102 req/s | **≈ 1 018** | 698 Mio |
| 2 worker(s), 2 CPU | 134 req/s | 94 req/s | **≈ 939** | 1 090 Mio |
| 4 worker(s), 4 CPU | 179 req/s | 125 req/s | **≈ 1 254** | 1 912 Mio |
