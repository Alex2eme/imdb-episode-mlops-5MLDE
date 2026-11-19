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


def test_wait_flow_run_tolerates_transient_api_errors(monkeypatch):
    """Une erreur 500 passagère du serveur Prefect ne fait plus échouer watch-source."""
    import contextlib
    import logging

    import prefect.client.orchestration as orch

    answers = iter([RuntimeError("500"), SimpleNamespace(state=None), RuntimeError("500"), "done"])

    class Client:
        def read_flow_run(self, _id):
            a = next(answers)
            if isinstance(a, Exception):
                raise a
            if a == "done":
                return SimpleNamespace(
                    state=SimpleNamespace(is_final=lambda: True, name="Completed")
                )
            return a

    monkeypatch.setattr(orch, "get_client", lambda sync_client: contextlib.nullcontext(Client()))
    monkeypatch.setattr(flows, "get_run_logger", lambda: logging.getLogger("t"))
    assert flows.wait_flow_run("x", poll_s=0) == "Completed"


def test_watch_source_does_not_relaunch_while_pipeline_runs(monkeypatch):
    """Snapshot validé, modèle pas encore enregistré, entraînement en cours : pas de doublon."""
    import logging

    launched = []
    monkeypatch.setattr(flows, "resolve_source_commit", lambda: "a" * 40)
    monkeypatch.setattr(flows.db, "get_snapshot", lambda sha: {"quality_status": "passed"})
    monkeypatch.setattr(flows.db, "model_trained_on", lambda sha: False)
    monkeypatch.setattr(flows, "active_pipeline_runs", lambda: ["lush-buffalo"])
    monkeypatch.setattr(flows, "get_run_logger", lambda: logging.getLogger("t"))
    monkeypatch.setattr(flows, "send_alert", lambda *a, **k: launched.append(a))
    out = flows.watch_source.fn()
    assert out["triggered"] is False and out["busy"] == ["lush-buffalo"]
    assert launched == []
