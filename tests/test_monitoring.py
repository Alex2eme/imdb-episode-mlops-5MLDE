import pandas as pd
import pytest

from imdb_mlops.monitoring.outcomes import match_outcomes
from imdb_mlops.monitoring.simulate import build_scenario


@pytest.fixture
def served():
    return pd.DataFrame(
        {
            "series_id": ["tt0000001"] * 3 + ["tt0000002"] * 2,
            "season": [1, 1, 1, 1, 1],
            "episode": [1, 2, 3, 1, 2],
            "rating": [8.0, 7.5, 8.2, 6.0, 6.5],
        }
    )


@pytest.fixture
def target(served):
    extra = pd.DataFrame(
        {
            "series_id": ["tt0000001", "tt0000002"],
            "season": [1, 1],
            "episode": [4, 3],
            "rating": [9.0, 6.1],
        }
    )
    return pd.concat([served, extra], ignore_index=True)


def test_match_outcomes_flags_new_episodes(served, target):
    pending = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "created_at": pd.Timestamp("2026-11-14", tz="UTC"),
            "series_id": ["tt0000001", "tt0000001", "tt0000001"],
            "season": [1, 1, 1],
            "episode": [2, 4, 5],  # connu, nouveau, pas encore publié
            "predicted_rating": [7.0, 8.0, 8.0],
            "model_version": "1",
            "data_sha": "J",
        }
    )
    known = {"J": set(zip(served.series_id, served.season, served.episode, strict=True))}
    actual = target.assign(votes=100)
    out = match_outcomes(pending, actual, known).set_index("id")
    assert set(out.index) == {"a", "b"}  # « c » reste en attente
    assert out.loc["a", "error"] == pytest.approx(-0.5)
    assert not out.loc["a", "is_new_episode"]
    assert out.loc["b", "is_new_episode"]
    assert out.loc["b", "abs_error"] == pytest.approx(1.0)


def test_match_outcomes_empty():
    assert match_outcomes(pd.DataFrame(), pd.DataFrame(), {}).empty


def test_build_scenario_mix(served, target):
    scenario = build_scenario(served, target, n=500, seed=1)
    kinds = {s["kind"] for s in scenario}
    assert {"new", "known", "future", "invalid", "unknown_series", "batch"} <= kinds
    new_eps = {
        (s["body"]["series_id"], s["body"]["episode"]) for s in scenario if s["kind"] == "new"
    }
    assert new_eps <= {("tt0000001", 4), ("tt0000002", 3)}
    assert build_scenario(served, target, 50, seed=1) == build_scenario(served, target, 50, seed=1)
