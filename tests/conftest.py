import io
import json
import tarfile

import numpy as np
import pandas as pd
import pytest

from imdb_mlops.config import load_config
from imdb_mlops.data.ingest import SnapshotManifest


def make_shows(n_series: int = 50, n_seasons: int = 3, n_eps: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    shows = []
    for i in range(n_series):
        sid = f"tt{1000000 + i:07d}"
        seasons = {
            str(s): {
                str(e): {
                    "r": round(float(rng.uniform(5, 9.5)), 1),
                    "v": int(rng.integers(10, 5000)),
                }
                for e in range(1, n_eps + 1)
            }
            for s in range(1, n_seasons + 1)
        }
        shows.append({"id": sid, "title": f"Show {i}", "votes": 1000 * (i + 1), "seasons": seasons})
    return shows


def build_archive(path, shows, orphans=(), declared_eps=None):
    """Archive au format du dépôt source (dossier racine préfixé par le SHA)."""
    root = "imdb-episode-dataset-abc123"
    meta = {
        "built": "2026-11-15T12:00:00+00:00",
        "show_count": len(shows),
        "ep_count": declared_eps
        if declared_eps is not None
        else sum(len(eps) for s in shows for eps in s["seasons"].values()),
        "shows": [{"id": s["id"], "title": s["title"]} for s in shows],
    }

    def add(tar, name, obj):
        data = json.dumps(obj).encode()
        info = tarfile.TarInfo(f"{root}/{name}")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    with tarfile.open(path, "w:gz") as tar:
        add(tar, "data/meta.json", meta)
        for s in [*shows, *orphans]:
            add(tar, f"data/shows/{s['id']}.json", s)
    return path


@pytest.fixture
def cfg():
    """Config de test : seuils adaptés à un petit jeu synthétique."""
    base = load_config()
    return base.model_copy(
        update={
            "quality": base.quality.model_copy(
                update={"min_rows": 100, "max_rows": 100_000, "min_series": 10}
            )
        }
    )


@pytest.fixture
def episodes_df(tmp_path):
    from imdb_mlops.data.ingest import parse_archive

    df, _ = parse_archive(build_archive(tmp_path / "a.tar.gz", make_shows()))
    return df


@pytest.fixture
def manifest(episodes_df):
    return SnapshotManifest(
        sha="a" * 40,
        source_repo="test/repo",
        commit_date="2026-11-15T12:00:00Z",
        built_at=None,
        ingested_at="2026-11-15T12:00:00+00:00",
        n_rows=len(episodes_df),
        n_series=int(episodes_df["series_id"].nunique()),
        n_orphan_series=0,
        declared_ep_count=len(episodes_df),
        parquet_path="",
        parquet_sha256="",
    )


@pytest.fixture(autouse=True)
def _reports_in_tmp(tmp_path, monkeypatch):
    """Les rapports qualité des tests ne polluent pas data/reports."""
    import imdb_mlops.data.quality as q

    monkeypatch.setattr(q, "ROOT_DIR", tmp_path)


__all__ = ["build_archive", "make_shows", "pd"]
