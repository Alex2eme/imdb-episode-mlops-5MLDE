"""Entraînement de bout en bout sur données synthétiques, MLflow local (SQLite).

Vérifie la chaîne complète sans infrastructure : traçabilité, registre, promotion,
rejet d'un challenger moins bon, rollback et reproduction d'un run.
"""

import hashlib

import mlflow
import pytest
from mlflow import MlflowClient

from imdb_mlops.data.ingest import parse_archive
from imdb_mlops.training import registry, train
from tests.conftest import build_archive, make_shows


@pytest.fixture
def env(tmp_path, monkeypatch, cfg):
    df, _ = parse_archive(build_archive(tmp_path / "a.tar.gz", make_shows(n_series=60)))
    parquet = tmp_path / "episodes.parquet"
    df.to_parquet(parquet, index=False)
    snap = {
        "sha": "b" * 40,
        "source_repo": "test/repo",
        "commit_date": "2026-11-15",
        "quality_status": "passed",
        "parquet_path": parquet.name,
        "parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "quality_report": {"success": True},
    }
    monkeypatch.setattr(train.db, "get_snapshot", lambda sha: snap if sha == snap["sha"] else None)
    monkeypatch.setattr(train.db, "last_valid_snapshot", lambda **_: snap)
    monkeypatch.setattr(train, "ROOT_DIR", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "config.yaml").write_text("test: true")
    monkeypatch.setattr(registry, "_record", lambda *a, **k: None)
    monkeypatch.setattr(registry, "send_alert", lambda *a, **k: None)

    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    cfg.training.n_trials = 3
    cfg.training.parallel_trials = 2
    cfg.training.threads_per_trial = 1
    cfg.training.n_estimators = 50
    cfg.promotion.max_rmse = 10.0
    cfg.promotion.must_beat_baseline = False
    yield cfg
    mlflow.set_tracking_uri(None)


def test_train_registers_and_promotes_then_rolls_back(env):
    cfg = env
    first = train.train(cfg)
    assert first.promoted
    client = MlflowClient()
    name = cfg.training.registered_model_name
    assert str(client.get_model_version_by_alias(name, "champion").version) == first.model_version

    run = client.get_run(first.run_id)
    assert run.data.tags["data.sha"] == "b" * 40
    assert "code.git_sha" in run.data.tags
    assert "test_rmse" in run.data.metrics
    assert any(k.startswith("hp_") for k in run.data.params)
    children = client.search_runs(
        [run.info.experiment_id], f"tags.mlflow.parentRunId = '{first.run_id}'"
    )
    assert len(children) == 3  # un run enfant par trial Optuna

    # Un challenger exigeant une amélioration impossible est rejeté.
    cfg.promotion.min_improvement = 100.0
    second = train.train(cfg, params=first.best_params)
    assert not second.promoted
    assert str(client.get_model_version_by_alias(name, "champion").version) == first.model_version

    # Promotion forcée puis rollback vers le champion précédent.
    cfg.promotion.min_improvement = -100.0
    third = train.train(cfg, params=first.best_params)
    assert third.promoted
    assert registry.rollback(cfg) == first.model_version
    assert str(client.get_model_version_by_alias(name, "champion").version) == first.model_version


def test_reproduce_gives_same_rmse(env):
    original = train.train(env)
    _, delta = train.reproduce(env, original.run_id)
    assert delta < 1e-9


def test_training_refused_on_failed_snapshot(env, monkeypatch):
    monkeypatch.setattr(train.db, "last_valid_snapshot", lambda **_: None)
    with pytest.raises(train.TrainingBlockedError):
        train.train(env)
