import numpy as np
import pandas as pd
import pytest

from imdb_mlops.features import FEATURES, build_features, features_for, temporal_split


@pytest.fixture
def small():
    return pd.DataFrame(
        {
            "series_id": ["tt1"] * 5 + ["tt2"] * 2,
            "series_title": "x",
            "series_votes": [100] * 5 + [50] * 2,
            "season": [1, 1, 1, 2, 2, 1, 1],
            "episode": [1, 2, 3, 1, 2, 1, 2],
            "rating": [8.0, 6.0, 7.0, 9.0, 5.0, 7.5, 8.5],
            "votes": [10, 20, 30, 40, 50, 60, 70],
        }
    )


def test_features_use_only_past_episodes(small):
    f = build_features(small).set_index(["series_id", "season", "episode"], drop=False)
    row = f.loc[("tt1", 2, 1)]
    assert row["series_ep_index"] == 3
    assert row["series_prev_mean"] == pytest.approx(7.0)
    assert row["prev_rating"] == 7.0
    assert row["prev3_mean"] == pytest.approx(7.0)
    assert row["is_season_premiere"] == 1
    assert np.isnan(row["season_prev_mean"])
    # premier épisode : pas d'historique
    assert np.isnan(f.loc[("tt1", 1, 1)]["series_prev_mean"])
    # pas de fuite entre séries
    assert np.isnan(f.loc[("tt2", 1, 1)]["prev_rating"])


def test_changing_target_does_not_change_its_features(small):
    modified = small.copy()
    modified.loc[3, "rating"] = 1.0  # tt1 S2E1
    a = build_features(small).iloc[3][FEATURES]
    b = build_features(modified).iloc[3][FEATURES]
    pd.testing.assert_series_equal(a, b)


def test_serving_matches_training(small):
    train = build_features(small).set_index(["series_id", "season", "episode"], drop=False)
    history = small[small.series_id == "tt1"]
    for season, episode in [(1, 1), (1, 3), (2, 2)]:
        served = features_for(history, season, episode).iloc[0]
        expected = train.loc[("tt1", season, episode)][FEATURES]
        np.testing.assert_allclose(served.to_numpy(float), expected.to_numpy(float), equal_nan=True)


def test_serving_future_episode(small):
    f = features_for(small[small.series_id == "tt1"], 3, 1).iloc[0]
    assert f["series_ep_index"] == 5
    assert f["prev_rating"] == 5.0


def test_temporal_split_keeps_last_episodes_in_test(small):
    feats = build_features(small)
    train, test = temporal_split(feats, 0.4)
    assert set(test["series_id"]) == {"tt1"}  # tt2 trop courte
    assert test["series_ep_index"].min() > train[train.series_id == "tt1"]["series_ep_index"].max()
