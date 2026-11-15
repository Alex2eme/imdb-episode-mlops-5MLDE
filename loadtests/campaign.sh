#!/usr/bin/env bash
# Campagne de tests de charge reproductible (≈ 45 min). Infrastructure démarrée au préalable.
#   bash loadtests/campaign.sh && python loadtests/report.py
#
# 1. Comparaison de configurations : 3 répétitions chacune (la variabilité d'un run à l'autre
#    sur Docker Desktop atteint ±40 % : on retient la médiane).
# 2. Montée en charge (config de référence 2 workers / 2 CPU) : courbe débit/latence et point
#    de saturation, base de la recommandation par nombre d'utilisateurs.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY=${PYTHON:-python}
run() { "$PY" loadtests/run_benchmarks.py "$@" 2>&1 | grep -E "req/s|Error" || true; }

for r in 1 2 3; do
  run --name "f-w1c1-r$r"       --workers 1 --cpus 1 --cache false --users 30 --duration 60s --note "campagne"
  run --name "f-w2c2-r$r"       --workers 2 --cpus 2 --cache false --users 30 --duration 60s --note "campagne"
  run --name "f-w4c4-r$r"       --workers 4 --cpus 4 --memory 3g --cache false --users 30 --duration 60s --note "campagne"
  run --name "f-w2c2-redis-r$r" --workers 2 --cpus 2 --cache true  --users 30 --duration 60s --note "campagne"
  run --name "f-w2c2-lru-r$r"   --workers 2 --cpus 2 --cache false --local-cache 50000 --users 30 --duration 60s --note "campagne"
done

for u in 1 5 10 20 40 80; do
  run --name "ramp-w2c2-u$u" --workers 2 --cpus 2 --cache false --users "$u" --duration 45s --note "montée en charge"
done
