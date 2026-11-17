"""Simulateur d'utilisation de l'API (démonstration, alimentation des tableaux de bord).

Génère un trafic réaliste à partir des snapshots :
- épisodes **nouveaux** (présents dans le snapshot cible, absents du snapshot servi) ;
- épisodes déjà connus ; épisodes futurs (pas encore publiés) ;
- une part de séries inconnues et d'entrées invalides (erreurs attendues) ;
- quelques appels par lot ; des feedbacks sur une partie des prédictions.
Ce n'est pas un test de charge (voir loadtests/) : le débit est volontairement modéré.
"""

from __future__ import annotations

import logging
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import httpx
import pandas as pd

from imdb_mlops import db
from imdb_mlops.config import ROOT_DIR

log = logging.getLogger(__name__)
KEY = ["series_id", "season", "episode"]
HEADERS = {"X-Traffic-Source": "simulation"}


def _frame(sha: str | None) -> pd.DataFrame:
    snap = db.get_snapshot(sha) if sha else db.last_valid_snapshot()
    if snap is None:
        raise ValueError(f"Snapshot {sha or '(dernier validé)'} introuvable")
    return pd.read_parquet(ROOT_DIR / snap["parquet_path"], columns=[*KEY, "rating"])


def build_scenario(
    served: pd.DataFrame, target: pd.DataFrame | None, n: int, seed: int = 0
) -> list[dict]:
    """Construit la liste des requêtes à envoyer (fonction pure, testée)."""
    rng = random.Random(seed)
    known = served[KEY].drop_duplicates()
    new = pd.DataFrame(columns=KEY)
    if target is not None:
        merged = target[KEY].merge(known, on=KEY, how="left", indicator=True)
        new = merged[merged["_merge"] == "left_only"][KEY]
    last = served.sort_values(KEY).groupby("series_id").tail(1)
    future = last.assign(episode=last["episode"] + 1)[KEY]

    def pick(df: pd.DataFrame) -> dict:
        r = df.iloc[rng.randrange(len(df))]
        return {"series_id": r.series_id, "season": int(r.season), "episode": int(r.episode)}

    pools = [("new", new, 0.45), ("known", known, 0.35), ("future", future, 0.10)]
    pools = [(k, d, w) for k, d, w in pools if len(d)]
    scenario = []
    for _ in range(n):
        u = rng.random()
        if u < 0.05:
            scenario.append(
                {
                    "kind": "unknown_series",
                    "body": {
                        "series_id": f"tt{rng.randrange(10**8, 10**9)}",
                        "season": 1,
                        "episode": 1,
                    },
                }
            )
        elif u < 0.08:
            scenario.append(
                {"kind": "invalid", "body": {"series_id": "not-an-id", "season": -1, "episode": 1}}
            )
        elif u < 0.12:
            kind, df, _ = rng.choice(pools)
            scenario.append(
                {"kind": "batch", "body": {"items": [pick(df) for _ in range(rng.randint(5, 30))]}}
            )
        else:
            kind, df, _ = rng.choices(pools, weights=[w for *_, w in pools])[0]
            scenario.append({"kind": kind, "body": pick(df)})
    return scenario


def simulate(
    api_url: str,
    n: int = 300,
    served_sha: str | None = None,
    target_sha: str | None = None,
    feedback_rate: float = 0.2,
    workers: int = 4,
    seed: int = 0,
) -> dict:
    served = _frame(served_sha)
    target = _frame(target_sha) if target_sha else None
    actual = target.set_index(KEY)["rating"] if target is not None else None
    scenario = build_scenario(served, target, n, seed)
    rng = random.Random(seed + 1)
    stats: Counter = Counter()

    feedbacks: list[tuple[str, float]] = []

    def send(item: dict) -> None:
        path = "/predict/batch" if item["kind"] == "batch" else "/predict"
        with httpx.Client(base_url=api_url, timeout=30, headers=HEADERS) as client:
            r = client.post(path, json=item["body"])
        stats[f"{item['kind']}:{r.status_code}"] += 1
        if r.status_code != 200 or item["kind"] == "batch" or rng.random() >= feedback_rate:
            return
        body = r.json()
        key = (body["series_id"], body["season"], body["episode"])
        if actual is not None and key in actual.index:
            rating = float(actual[key])
        else:
            rating = min(10.0, max(1.0, body["predicted_rating"] + rng.gauss(0, 0.5)))
        feedbacks.append((body["prediction_id"], round(rating, 1)))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(send, scenario))

    # Comme un utilisateur réel, le feedback arrive après coup : les traces de prédiction
    # (écrites en base par lots, < 1 s) sont alors disponibles.
    time.sleep(2)
    with httpx.Client(base_url=api_url, timeout=30, headers=HEADERS) as client:
        for pid, rating in feedbacks:
            fb = client.post("/feedback", json={"prediction_id": pid, "rating": rating})
            stats[f"feedback:{fb.status_code}"] += 1
    log.info("Trafic simulé : %s", dict(stats))
    return dict(sorted(stats.items()))
