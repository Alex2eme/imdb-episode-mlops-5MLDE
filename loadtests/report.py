"""Rapport des tests de charge : reports/loadtests/results.json → RESULTS.md.

- Comparaison de configurations : médiane des répétitions (`f-<config>-rN`).
- Montée en charge (`ramp-...-uN`) : débit et latences selon la concurrence.
- Recommandation : capacité utilisable (marge de 30 %) convertie en utilisateurs réels avec
  une hypothèse explicite de comportement (une prédiction toutes les THINK_TIME_S secondes).

Usage : python loadtests/report.py
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "loadtests" / "results.json"
OUT = ROOT / "reports" / "loadtests" / "RESULTS.md"

SLO_P95_MS = 300  # objectif de service : p95 de /predict ≤ 300 ms
HEADROOM = 0.70  # on n'exploite que 70 % de la capacité mesurée
THINK_TIME_S = 10  # hypothèse : un utilisateur actif fait une prédiction toutes les 10 s


def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def n(value, digits=0, suffix=""):
    if value is None:
        return "-"
    return f"{value:,.{digits}f}{suffix}".replace(",", " ")


def campaign(results):
    groups = defaultdict(list)
    for r in results:
        m = re.fullmatch(r"f-(.+)-r\d+", r["name"])
        if m:
            groups[m.group(1)].append(r)
    rows = {}
    for name, runs in groups.items():
        rows[name] = {
            "config": runs[0]["config"],
            "rps": med([x["all"].get("rps") for x in runs]),
            "rps_all": [x["all"].get("rps") for x in runs],
            "p50": med([x["predict"].get("p50_ms") for x in runs]),
            "p95": med([x["predict"].get("p95_ms") for x in runs]),
            "p99": med([x["predict"].get("p99_ms") for x in runs]),
            "fail": sum(x["all"].get("failures", 0) for x in runs),
            "req": sum(x["all"].get("requests", 0) for x in runs),
            "cpu": med([x["api_cpu_pct"]["avg"] for x in runs]),
            "mem": max((x["api_mem_mib"]["max"] or 0) for x in runs),
            "hit": med([x.get("cache_hit_ratio") for x in runs]),
        }
    return rows


def ramp(results):
    rows = []
    for r in results:
        m = re.fullmatch(r"ramp-(.+)-u(\d+)", r["name"])
        if m:
            rows.append(
                {
                    "users": int(m.group(2)),
                    "rps": r["all"].get("rps"),
                    "p50": r["predict"].get("p50_ms"),
                    "p95": r["predict"].get("p95_ms"),
                    "p99": r["predict"].get("p99_ms"),
                    "cpu": r["api_cpu_pct"]["avg"],
                    "fail": r["all"].get("failures", 0),
                }
            )
    return sorted(rows, key=lambda x: x["users"])


def main() -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    camp, curve = campaign(results), ramp(results)
    lines = [
        "# Résultats des tests de charge",
        "",
        "Générés par `loadtests/report.py` à partir de `results.json` (campagne "
        "`loadtests/campaign.sh`). Plateforme : Docker Desktop / WSL2, poste 12 cœurs ; Locust "
        "(4 processus, FastHttpUser) dans un conteneur du même réseau Docker. Scénario : 85 % "
        "`/predict` (séries tirées selon leur popularité), 5 % lots de 20, 5 % `/model`, 5 % "
        "entrées invalides ; utilisateurs sans temps de pause (mesure de capacité).",
        "",
        "## 1. Comparaison de configurations (30 utilisateurs simultanés, médiane de 3 runs)",
        "",
        "| Configuration | Workers | CPU | Cache | Débit médian (req/s) | Débits des runs | "
        "/predict p50 | p95 | p99 | Échecs | CPU moyen | RAM max | Hit cache |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in sorted(camp.items()):
        c = r["config"]
        cache = ("Redis " if c["cache"] == "true" else "") + (
            "LRU local" if c.get("local_cache") else ""
        )
        hit = None if r["hit"] is None else r["hit"] * 100
        lines.append(
            f"| {name} | {c['workers']} | {c['cpus']:g} | {cache.strip() or 'aucun'} | "
            f"**{n(r['rps'], 1)}** | {', '.join(n(v) for v in r['rps_all'])} | "
            f"{n(r['p50'], 0, ' ms')} | {n(r['p95'], 0, ' ms')} | {n(r['p99'], 0, ' ms')} | "
            f"{r['fail']}/{r['req']} | {n(r['cpu'], 0, ' %')} | {n(r['mem'], 0, ' Mio')} | "
            f"{n(hit, 1, ' %')} |"
        )

    if curve:
        lines += [
            "",
            "## 2. Montée en charge (2 workers, 2 CPU, sans cache)",
            "",
            f"| Utilisateurs simultanés | Débit (req/s) | /predict p50 | p95 | p99 | CPU API | "
            f"Échecs | p95 ≤ {SLO_P95_MS} ms |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in curve:
            ok = r["p95"] is not None and r["p95"] <= SLO_P95_MS
            lines.append(
                f"| {r['users']} | {n(r['rps'], 1)} | {n(r['p50'], 0, ' ms')} | "
                f"{n(r['p95'], 0, ' ms')} | {n(r['p99'], 0, ' ms')} | {n(r['cpu'], 0, ' %')} | "
                f"{r['fail']} | {'oui' if ok else 'non'} |"
            )

    lines += [
        "",
        "## 3. Recommandation de dimensionnement",
        "",
        f"Hypothèses : objectif p95 `/predict` ≤ {SLO_P95_MS} ms ; on n'exploite que "
        f"{int(HEADROOM * 100)} % du débit mesuré (marge pour les pics, le GC et les "
        f"rechargements de modèle) ; un utilisateur actif fait une prédiction toutes les "
        f"{THINK_TIME_S} s ({1 / THINK_TIME_S:.1f} req/s).",
        "",
    ]
    ok_points = [r for r in curve if r["p95"] is not None and r["p95"] <= SLO_P95_MS]
    sustained = max(ok_points, key=lambda r: r["rps"] or 0) if ok_points else None
    if sustained:
        lines.append(
            f"- Point de fonctionnement mesuré respectant l'objectif (2 workers / 2 CPU) : "
            f"**{n(sustained['rps'])} req/s** à {sustained['users']} utilisateurs simultanés "
            f"(p95 {n(sustained['p95'])} ms)."
        )
    lines += [
        "",
        f"| Configuration API | Débit médian mesuré | Débit utilisable (x {HEADROOM:g}) | "
        "Utilisateurs actifs supportés | RAM à réserver (pic x 1,5) |",
        "|---|---|---|---|---|",
    ]
    base = [k for k in camp if "redis" not in k and "lru" not in k]
    for name in sorted(base, key=lambda k: camp[k]["config"]["workers"]):
        r = camp[name]
        usable = (r["rps"] or 0) * HEADROOM
        lines.append(
            f"| {r['config']['workers']} worker(s), {r['config']['cpus']:g} CPU | "
            f"{n(r['rps'])} req/s | {n(usable)} req/s | **≈ {n(usable * THINK_TIME_S)}** | "
            f"{n(r['mem'] * 1.5)} Mio |"
        )
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{OUT} : {len(camp)} configurations, {len(curve)} points de montée en charge")


if __name__ == "__main__":
    main()
