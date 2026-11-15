"""Chargement de la configuration : YAML (paramètres métier) + environnement (infra/secrets)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"


class ProjectConfig(BaseModel):
    name: str
    seed: int = 42


class DataConfig(BaseModel):
    source_repo: str
    source_branch: str = "main"
    pinned_sha: str = ""
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    min_votes: int = Field(10, ge=0)


class TrainingConfig(BaseModel):
    experiment_name: str
    registered_model_name: str
    test_size: float = Field(0.2, gt=0, lt=1)
    n_trials: int = Field(20, ge=1)
    n_jobs: int = -1


class PromotionConfig(BaseModel):
    metric: str = "rmse"
    min_improvement: float = 0.0
    max_rmse: float = 1.0


class AppConfig(BaseModel):
    project: ProjectConfig
    data: DataConfig
    training: TrainingConfig
    promotion: PromotionConfig


class Settings(BaseSettings):
    """Paramètres d'infrastructure, lus depuis l'environnement ou le fichier .env."""

    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    config_path: Path = DEFAULT_CONFIG_PATH
    mlflow_tracking_uri: str = "http://localhost:5000"
    database_url: str = "postgresql+psycopg://mlops:mlops@localhost:5432/app"
    redis_url: str = "redis://localhost:6379/0"
    alert_webhook_url: str = ""
    github_token: str = ""


def load_config(path: Path | str | None = None) -> AppConfig:
    path = Path(path) if path else Settings().config_path
    with Path(path).open(encoding="utf-8") as f:
        return AppConfig.model_validate(yaml.safe_load(f))


@lru_cache
def get_settings() -> Settings:
    return Settings()
