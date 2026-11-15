import pytest
from pydantic import ValidationError

from imdb_mlops.config import AppConfig, load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg.project.seed == 42
    assert cfg.data.source_repo == "ya0903/imdb-episode-dataset"
    assert 0 < cfg.training.test_size < 1


def test_invalid_config_rejected():
    raw = load_config().model_dump()
    raw["training"]["test_size"] = 1.5
    with pytest.raises(ValidationError):
        AppConfig.model_validate(raw)
