"""Orchestration : politique de relance et arrêt de la chaîne sur lot non conforme."""

from types import SimpleNamespace

import pytest

from imdb_mlops.orchestration import flows


class _State:
    def __init__(self, exc):
        self.exc = exc

    def result(self):
        raise self.exc


class DataQualityError(Exception):
    pass


class TrainingBlockedError(Exception):
    pass


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ConnectionError("mlflow down"), True),
        (TimeoutError(), True),
        (RuntimeError("réseau"), True),
        (DataQualityError(), False),
        (TrainingBlockedError(), False),
    ],
)
def test_only_transient_errors_are_retried(exc, expected):
    task = SimpleNamespace(name="t")
    assert flows._is_transient(task, None, _State(exc)) is expected


@pytest.fixture
def prefect_api():
    """Serveur Prefect réel (docker compose). Le test harness embarqué de Prefect 3.8 est
    incompatible avec la version de FastAPI du projet : on teste contre le vrai serveur."""
    import os

    if not os.environ.get("PREFECT_API_URL"):
        pytest.skip("PREFECT_API_URL non défini (lancer dans le conteneur pipeline)")


@pytest.mark.integration
def test_invalid_batch_stops_before_training(prefect_api, monkeypatch):
    from imdb_mlops.data import pipeline
    from imdb_mlops.data.quality import DataQualityError as RealDQE
    from imdb_mlops.data.quality import QualityReport

    calls = {"train": 0, "alerts": []}

    def fake_data_step(cfg, ref=None, fault=None):
        raise RealDQE(QualityReport("x" * 40, "now", False, []))

    monkeypatch.setattr(pipeline, "run_data_step", fake_data_step)
    monkeypatch.setattr(flows.db, "record_step", lambda **k: None)
    monkeypatch.setattr(flows, "send_alert", lambda title, *a, **k: calls["alerts"].append(title))
    monkeypatch.setattr(
        flows.train_and_promote, "fn", lambda *a, **k: calls.__setitem__("train", 1)
    )

    state = flows.daily_pipeline(return_state=True)
    assert state.is_failed()
    assert calls["train"] == 0
    assert any("daily-pipeline" in t for t in calls["alerts"])


@pytest.mark.parametrize(
    ("snapshot", "trained", "expected"),
    [
        (None, False, False),  # nouveau commit
        ({"quality_status": "failed"}, False, True),  # lot rejeté : inutile de rejouer
        ({"quality_status": "passed"}, True, True),  # entraîné
        ({"quality_status": "passed"}, False, False),  # entraînement en échec : à reprendre
    ],
)
def test_already_processed(snapshot, trained, expected):
    assert flows.already_processed(snapshot, trained) is expected


def test_plan_recovery_marks_orphans_and_late_runs():
    from datetime import UTC, datetime, timedelta

    from imdb_mlops.orchestration.recovery import plan_recovery

    now = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)

    def run(state, minutes_ago):
        st = SimpleNamespace(type=SimpleNamespace(value=state))
        return SimpleNamespace(state=st, expected_start_time=now - timedelta(minutes=minutes_ago))

    zombie, pending = run("RUNNING", 3 * 24 * 60), run("PENDING", 60)
    late, soon = run("SCHEDULED", 60), run("SCHEDULED", -20)
    queued = run("SCHEDULED", 2)  # en attente de place depuis peu : conservé
    orphans, to_cancel = plan_recovery(
        [zombie, pending, late, soon, queued], now, timedelta(minutes=10)
    )
    assert orphans == [zombie, pending]
    assert to_cancel == [late]
