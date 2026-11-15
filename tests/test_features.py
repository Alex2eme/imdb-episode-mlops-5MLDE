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


def _random_series(seed: int = 0, n_series: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_series):
        for season in range(int(rng.integers(0, 4)), int(rng.integers(4, 7))):
            for episode in sorted(
                rng.choice(range(0, 25), int(rng.integers(1, 12)), replace=False)
            ):
                rows.append(
                    (
                        f"tt{i:07d}",
                        1000 + i,
                        season,
                        int(episode),
                        round(float(rng.uniform(1, 10)), 1),
                        int(rng.integers(10, 9999)),
                    )
                )
    return pd.DataFrame(
        rows, columns=["series_id", "series_votes", "season", "episode", "rating", "votes"]
    )


def test_fast_serving_features_match_training_on_every_row():
    from imdb_mlops.features import SeriesHistory, serving_features

    df = _random_series()
    train = build_features(df)
    histories = {sid: SeriesHistory.from_frame(g) for sid, g in df.groupby("series_id")}
    fast = np.vstack(
        [serving_features(histories[r.series_id], r.season, r.episode) for r in train.itertuples()]
    )
    np.testing.assert_allclose(
        fast, train[FEATURES].to_numpy(float), rtol=1e-9, atol=1e-9, equal_nan=True
    )


def test_fast_serving_features_match_reference_for_unseen_episodes():
    from imdb_mlops.features import SeriesHistory, serving_features

    df = _random_series(seed=1, n_series=10)
    for sid, g in df.groupby("series_id"):
        h = SeriesHistory.from_frame(g)
        last = g.sort_values(["season", "episode"]).iloc[-1]
        for season, episode in [
            (int(last.season), int(last.episode) + 1),
            (int(last.season) + 1, 1),
            (0, 0),
        ]:
            np.testing.assert_allclose(
                serving_features(h, season, episode),
                features_for(g, season, episode).iloc[0].to_numpy(float),
                rtol=1e-9,
                atol=1e-9,
                equal_nan=True,
                err_msg=f"{sid} S{season}E{episode}",
            )
