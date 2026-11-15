"""Features : historique de la série *avant* l'épisode à prédire.

Une seule implémentation sert à l'entraînement (vectorisée sur tout le snapshot) et à
l'inférence (sur l'historique d'une série) : aucune divergence possible entre les deux.
Les features n'utilisent que les épisodes strictement antérieurs (ordre saison, épisode)
afin d'éviter toute fuite de la cible. Le nombre de votes de l'épisode lui-même est exclu :
il n'est connu qu'après diffusion.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

KEY = ["series_id", "season", "episode"]
TARGET = "rating"

FEATURES = [
    "season",
    "episode",
    "is_season_premiere",
    "series_ep_index",
    "season_ep_index",
    "log_series_votes",
    "series_prev_mean",
    "series_prev_std",
    "prev_rating",
    "prev3_mean",
    "season_prev_mean",
    "prev_trend",
    "log_prev_votes",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les features pour chaque ligne (série, saison, épisode) du DataFrame."""
    out = df.sort_values(KEY, kind="stable").reset_index(drop=True).copy()
    # Cible inconnue (NaN) à l'inférence : neutralisée dans les cumuls, dont elle est
    # de toute façon soustraite (seuls les épisodes antérieurs comptent).
    rating = out[TARGET].astype("float64").fillna(0.0)
    out["_r"] = rating
    g = out.groupby("series_id", sort=False)

    n_prev = g.cumcount()
    out["series_ep_index"] = n_prev
    cum = g["_r"].cumsum() - rating
    cum_sq = (rating**2).groupby(out["series_id"], sort=False).cumsum() - rating**2
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = cum / n_prev.replace(0, np.nan)
        var = cum_sq / n_prev.replace(0, np.nan) - mean**2
    out["series_prev_mean"] = mean
    out["series_prev_std"] = np.sqrt(var.clip(lower=0)).where(n_prev > 1)

    shifted = g[TARGET].shift(1)
    out["prev_rating"] = shifted
    # Moyenne des 3 épisodes précédents, vectorisée (un rolling par série coûtait ~40 s
    # sur 10 000 séries) : nanmean des décalages 1, 2 et 3.
    lags = np.column_stack([g[TARGET].shift(k).to_numpy(dtype="float64") for k in (1, 2, 3)])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # lignes sans historique → NaN
        out["prev3_mean"] = np.nanmean(lags, axis=1)
    out["prev_trend"] = out["prev3_mean"] - out["series_prev_mean"]
    out["log_prev_votes"] = np.log1p(g["votes"].shift(1).astype("float64"))

    gs = out.groupby(["series_id", "season"], sort=False)
    n_prev_season = gs.cumcount()
    out["season_ep_index"] = n_prev_season
    season_cum = gs["_r"].cumsum() - rating
    out["season_prev_mean"] = season_cum / n_prev_season.replace(0, np.nan)

    out["is_season_premiere"] = (n_prev_season == 0).astype("int8")
    out["log_series_votes"] = np.log1p(out["series_votes"].astype("float64"))
    out["season"] = out["season"].astype("float64")
    out["episode"] = out["episode"].astype("float64")
    for col in FEATURES:
        out[col] = out[col].astype("float64")
    return out.drop(columns="_r")


def features_for(history: pd.DataFrame, season: int, episode: int) -> pd.DataFrame:
    """Features d'un épisode à partir de l'historique de sa série (inférence).

    Seuls les épisodes strictement antérieurs à (season, episode) sont utilisés, que
    l'épisode existe déjà dans l'historique ou non.
    """
    if history.empty:
        raise ValueError("Historique de série vide")
    before = history[
        (history["season"] < season)
        | ((history["season"] == season) & (history["episode"] < episode))
    ]
    target = {
        "series_id": history["series_id"].iloc[0],
        "series_votes": history["series_votes"].iloc[0],
        "season": season,
        "episode": episode,
        TARGET: np.nan,
        "votes": np.nan,
    }
    frame = pd.concat([before[list(target)], pd.DataFrame([target])], ignore_index=True)
    frame["season"] = frame["season"].astype("int64")
    frame["episode"] = frame["episode"].astype("int64")
    return build_features(frame).iloc[[-1]][FEATURES].reset_index(drop=True)


def temporal_split(feats: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Les derniers épisodes de chaque série forment le test : on évalue la capacité à
    prédire la suite d'une série à partir de son passé (cas d'usage réel)."""
    n = feats.groupby("series_id")["series_ep_index"].transform("size")
    is_test = feats["series_ep_index"] >= np.floor(n * (1 - test_size))
    is_test &= n >= 3  # séries trop courtes : uniquement en entraînement
    return feats[~is_test], feats[is_test]
