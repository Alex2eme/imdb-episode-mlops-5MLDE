"""Injection de défauts dans un lot de données : démonstration d'incident contrôlé.

Simule une livraison amont défectueuse pour montrer que les contrôles qualité bloquent
la chaîne, remontent l'information et alertent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FAULTS = {
    "ratings_out_of_range": "5 % des notes multipliées par 10 (échelle 0-100)",
    "duplicates": "2 % des épisodes dupliqués",
    "missing_column": "colonne `votes` supprimée (rupture de schéma)",
    "truncated": "lot tronqué à 30 % des lignes (livraison partielle)",
    "null_keys": "3 % des numéros d'épisode manquants",
}


def inject_fault(df: pd.DataFrame, fault: str, seed: int = 0) -> pd.DataFrame:
    if fault not in FAULTS:
        raise ValueError(f"Défaut inconnu {fault!r}, choix : {sorted(FAULTS)}")
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)
    if fault == "ratings_out_of_range":
        idx = rng.choice(n, size=max(1, n // 20), replace=False)
        df.loc[idx, "rating"] = df.loc[idx, "rating"] * 10
    elif fault == "duplicates":
        idx = rng.choice(n, size=max(1, n // 50), replace=False)
        df = pd.concat([df, df.iloc[idx]], ignore_index=True)
    elif fault == "missing_column":
        df = df.drop(columns=["votes"])
    elif fault == "truncated":
        df = df.sample(frac=0.3, random_state=seed).reset_index(drop=True)
    elif fault == "null_keys":
        idx = rng.choice(n, size=max(1, 3 * n // 100), replace=False)
        df.loc[idx, "episode"] = pd.NA
    return df
