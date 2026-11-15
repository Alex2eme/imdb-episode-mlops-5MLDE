"""Banc de tests de charge : une configuration de l'API par scénario, mesures consolidées.

Pour chaque scénario :
1. redéploie l'API avec la configuration voulue (workers, CPU, cache) ;
2. vide le cache Redis, échauffe l'API (10 s, non mesuré) ;
3. lance Locust (conteneur dédié) pendant la durée voulue, en relevant CPU et mémoire du
   conteneur API toutes les 2 s ;
4. consolide débit, latences (p50/p95/p99), erreurs et ressources dans
   reports/loadtests/results.json (une ligne par scénario ; un même nom est remplacé).

Usage (depuis la racine du dépôt, infrastructure démarrée) :
    python loadtests/run_benchmarks.py --name c2-w2-cache --workers 2 --cpus 2 --cache true \
        --users 20 --duration 60
Uniquement la bibliothèque standard : exécutable depuis l'hôte sans dépendance.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "loadtests"
RESULTS = OUT / "results.json"
ENV = {**os.environ, "MSYS_NO_PATHCONV": "1"}
LOCUST_PROCESSES = 4  # générateur de charge multi-processus (sinon plafond côté Locust)


def sh(cmd: list[str], check: bool = True, capture: bool = False) -> str:
    r = subprocess.run(
        cmd, cwd=ROOT, env=ENV, text=True, encoding="utf-8", capture_output=capture, check=False
    )
    if check and r.returncode != 0:
        raise RuntimeError(f"Échec ({r.returncode}) : {' '.join(cmd)}\n{r.stderr or ''}")
    return r.stdout if capture else ""


def deploy_api(workers: int, cpus: str, cache: str, memory: str, local_cache: int) -> None:
    ENV.update(
        API_WORKERS=str(workers),
        API_CPUS=cpus,
        CACHE_ENABLED=cache,
        API_MEMORY=memory,
        LOCAL_CACHE_SIZE=str(local_cache),
    )
    sh(["docker", "compose", "up", "-d", "--wait", "api"])
    sh(["docker", "compose", "exec", "-T", "redis", "redis-cli", "FLUSHALL"], capture=True)


def locust(name: str, users: int, duration: str, csv_prefix: str | None) -> None:
    cmd = [
        "docker",
        "compose",
        "run",
        "--rm",
        "--no-deps",
        "-v",
        f"{ROOT / 'loadtests'}:/app/loadtests",
        "-v",
        f"{ROOT / 'reports'}:/app/reports",
        "pipeline",
        "locust",
        "-f",
        "loadtests/locustfile.py",
        "--headless",
        "-u",
        str(users),
        "-r",
        str(max(1, users // 2)),
        "-t",
        duration,
        "--host",
        "http://api:8000",
        "--only-summary",
        "--stop-timeout",
        "5",
    ]
    if csv_prefix:
        cmd += ["--csv", csv_prefix, "--html", f"{csv_prefix}.html"]
    # Locust sort en code 1 dès qu'une requête échoue : les échecs sont comptés dans les
    # statistiques, ce n'est pas une erreur du banc de test.
    sh(cmd, check=False, capture=True)


class StatsSampler(threading.Thread):
    """Relève CPU (%) et mémoire (Mio) du conteneur API pendant le test."""

    def __init__(self):
        super().__init__(daemon=True)
        self.cpu: list[float] = []
        self.mem: list[float] = []
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            out = sh(
                ["docker", "stats", "--no-stream", "--format", "{{json .}}", "projet-api-1"],
                check=False,
                capture=True,
            )
            try:
                s = json.loads(out.strip().splitlines()[-1])
                self.cpu.append(float(s["CPUPerc"].rstrip("%")))
                used = s["MemUsage"].split("/")[0].strip()
                factor = {"GiB": 1024, "MiB": 1, "KiB": 1 / 1024}
                unit = next(u for u in factor if used.endswith(u))
                self.mem.append(float(used[: -len(unit)]) * factor[unit])
            except (ValueError, IndexError, KeyError, StopIteration):
                pass
            self._halt.wait(2)

    def stop(self):
        self._halt.set()
        self.join(timeout=10)


def parse(csv_prefix: str) -> dict:
    rows = {
        r["Name"]: r
        for r in csv.DictReader((ROOT / f"{csv_prefix}_stats.csv").open(encoding="utf-8"))
    }

    def pick(name: str) -> dict:
        r = rows.get(name)
        if not r or not int(r["Request Count"]):
            return {}
        return {
            "requests": int(r["Request Count"]),
            "failures": int(r["Failure Count"]),
            "rps": round(float(r["Requests/s"]), 1),
            "p50_ms": float(r["50%"]),
            "p95_ms": float(r["95%"]),
            "p99_ms": float(r["99%"]),
            "avg_ms": round(float(r["Average Response Time"]), 1),
        }

    return {"all": pick("Aggregated"), "predict": pick("/predict"), "batch": pick("/predict/batch")}


def cache_hit_ratio() -> float | None:
    out = sh(
        ["docker", "compose", "exec", "-T", "redis", "redis-cli", "INFO", "stats"],
        check=False,
        capture=True,
    )
    stats = dict(line.split(":", 1) for line in out.splitlines() if ":" in line)
    hits, misses = int(stats.get("keyspace_hits", 0)), int(stats.get("keyspace_misses", 0))
    return round(hits / (hits + misses), 3) if hits + misses else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cpus", default="2")
    ap.add_argument("--memory", default="2g")
    ap.add_argument("--cache", default="true", help="Cache Redis (niveau 2)")
    ap.add_argument(
        "--local-cache",
        type=int,
        default=0,
        help="Taille du cache LRU local par worker (niveau 1), 0 = désactivé",
    )
    ap.add_argument("--users", type=int, default=20)
    ap.add_argument("--duration", default="60s")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(
        f"[{args.name}] déploiement : {args.workers} worker(s), {args.cpus} CPU, cache={args.cache}"
    )
    deploy_api(args.workers, args.cpus, args.cache, args.memory, args.local_cache)
    locust(args.name, max(2, args.users // 4), "10s", None)  # échauffement non mesuré
    sh(
        ["docker", "compose", "exec", "-T", "redis", "redis-cli", "CONFIG", "RESETSTAT"],
        capture=True,
    )

    prefix = f"reports/loadtests/{args.name}"
    sampler = StatsSampler()
    sampler.start()
    t0 = time.time()
    locust(args.name, args.users, args.duration, prefix)
    sampler.stop()

    result = {
        "name": args.name,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "workers": args.workers,
            "cpus": float(args.cpus),
            "cache": args.cache,
            "local_cache": args.local_cache,
            "users": args.users,
            "duration": args.duration,
            "memory": args.memory,
        },
        "note": args.note,
        "wall_s": round(time.time() - t0, 1),
        **parse(prefix),
        "api_cpu_pct": {
            "avg": round(statistics.mean(sampler.cpu), 1) if sampler.cpu else None,
            "max": max(sampler.cpu, default=None),
        },
        "api_mem_mib": {
            "avg": round(statistics.mean(sampler.mem)) if sampler.mem else None,
            "max": round(max(sampler.mem)) if sampler.mem else None,
        },
        "cache_hit_ratio": cache_hit_ratio() if args.cache == "true" else None,
    }
    results = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else []
    results = [r for r in results if r["name"] != args.name] + [result]
    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    a, p = result["all"], result["predict"]
    print(
        f"[{args.name}] {a.get('rps')} req/s | /predict p50 {p.get('p50_ms')} ms, "
        f"p95 {p.get('p95_ms')} ms, p99 {p.get('p99_ms')} ms | échecs "
        f"{a.get('failures')}/{a.get('requests')} | CPU {result['api_cpu_pct']} | "
        f"RAM {result['api_mem_mib']} | cache {result['cache_hit_ratio']}"
    )


if __name__ == "__main__":
    main()
