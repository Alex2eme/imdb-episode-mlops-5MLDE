"""Scénario de charge de l'API de prédiction (Locust).

Trafic modélisé :
- 85 % de `/predict` : séries tirées proportionnellement à leur popularité (votes), épisode
  existant ou suivant le dernier publié. La popularité crée une répétition naturelle des
  requêtes, d'où un taux de hit du cache réaliste plutôt qu'artificiel ;
- 5 % de `/predict/batch` (20 épisodes) ; 5 % de `/model` ; 5 % d'entrées invalides.

Les utilisateurs n'attendent pas entre deux requêtes (`wait_time` nul) : on mesure la
capacité maximale pour une concurrence donnée. Le passage en nombre d'utilisateurs réels
(avec temps de réflexion) est fait dans l'analyse (voir reports/loadtests/).

FastHttpUser (geventhttpclient) et plusieurs processus Locust : avec HttpUser (requests) dans
un seul processus, le générateur plafonnait vers 110 req/s quelle que soit la taille de l'API.

Variables : SNAPSHOT_PARQUET (défaut : dernier Parquet de data/raw), LOCUST_SEED.
"""

from __future__ import annotations

import glob
import os
import random

import numpy as np
import pandas as pd
from locust import FastHttpUser, constant, events, task

# Identifie le trafic de charge : il est exclu du suivi de performance en production.
HEADERS = {"X-Traffic-Source": "loadtest"}
KEYS: list[tuple[str, int, int]] = []
# Séquence de clés pré-tirée selon la popularité : un tirage pondéré parmi ~430 000 clés à
# chaque requête ferait de Locust lui-même le goulot d'étranglement.
SAMPLE: np.ndarray = np.empty(0, dtype=np.int64)


@events.init.add_listener
def load_keys(environment, **_):
    global SAMPLE  # noqa: PLW0603 (initialisé une fois par le hook Locust)
    path = os.environ.get("SNAPSHOT_PARQUET") or max(
        glob.glob("data/raw/episodes_*.parquet"), key=os.path.getmtime
    )
    df = pd.read_parquet(path, columns=["series_id", "series_votes", "season", "episode"])
    # Épisode suivant le dernier publié de chaque série (prédiction d'un épisode futur).
    last = df.sort_values(["series_id", "season", "episode"]).groupby("series_id").tail(1)
    future = last.assign(episode=last["episode"] + 1)
    keys = pd.concat([df, future], ignore_index=True)
    KEYS.extend(
        zip(keys["series_id"], keys["season"].astype(int), keys["episode"].astype(int), strict=True)
    )
    # Popularité : poids ∝ votes de la série, réparti sur ses épisodes.
    n_eps = keys.groupby("series_id")["series_id"].transform("size")
    w = keys["series_votes"].astype(float) / n_eps
    rng = np.random.default_rng(int(os.environ.get("LOCUST_SEED", "0")))
    SAMPLE = rng.choice(len(KEYS), size=1_000_000, p=(w / w.sum()).to_numpy())
    print(f"Locust : {len(KEYS)} clés chargées depuis {path}")


class ApiUser(FastHttpUser):
    wait_time = constant(0)

    def on_start(self):
        self.pos = random.randrange(len(SAMPLE))

    def _key(self) -> dict:
        self.pos = (self.pos + 1) % len(SAMPLE)
        sid, season, episode = KEYS[SAMPLE[self.pos]]
        return {"series_id": sid, "season": season, "episode": episode}

    @task(85)
    def predict(self):
        self.client.post("/predict", json=self._key(), name="/predict", headers=HEADERS)

    @task(5)
    def predict_batch(self):
        self.client.post(
            "/predict/batch",
            headers=HEADERS,
            json={"items": [self._key() for _ in range(20)]},
            name="/predict/batch",
        )

    @task(5)
    def model(self):
        self.client.get("/model", name="/model", headers=HEADERS)

    @task(5)
    def invalid(self):
        with self.client.post(
            "/predict",
            headers=HEADERS,
            json={"series_id": "bad", "season": -1, "episode": 1},
            name="/predict [invalide]",
            catch_response=True,
        ) as r:
            if r.status_code == 422:
                r.success()  # rejet attendu : ce n'est pas une erreur du service
            else:
                r.failure(f"attendu 422, reçu {r.status_code}")
