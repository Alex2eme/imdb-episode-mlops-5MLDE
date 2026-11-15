"""Tests d'intégration de l'infrastructure : `docker compose up -d` doit avoir été lancé."""

import httpx
import pytest
import redis
from sqlalchemy import create_engine, text

from imdb_mlops.config import get_settings

pytestmark = pytest.mark.integration


def test_mlflow_up():
    r = httpx.get(f"{get_settings().mlflow_tracking_uri}/health", timeout=5)
    assert r.status_code == 200


def test_prefect_up():
    r = httpx.get("http://localhost:4200/api/health", timeout=5)
    assert r.status_code == 200


def test_postgres_app_db():
    engine = create_engine(get_settings().database_url)
    with engine.connect() as conn:
        assert conn.execute(text("select 1")).scalar() == 1


def test_redis_up():
    assert redis.Redis.from_url(get_settings().redis_url).ping()
