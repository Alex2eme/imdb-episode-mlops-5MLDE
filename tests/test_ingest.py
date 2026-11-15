import pytest

from imdb_mlops.data.ingest import EPISODE_COLUMNS, IngestionError, parse_archive
from tests.conftest import build_archive, make_shows


def test_parse_archive_flattens_episodes(tmp_path):
    shows = make_shows(n_series=4, n_seasons=2, n_eps=5)
    df, info = parse_archive(build_archive(tmp_path / "a.tar.gz", shows))
    assert list(df.columns) == EPISODE_COLUMNS
    assert len(df) == 4 * 2 * 5
    assert df["season"].dtype.name == "Int64"
    assert info["declared_ep_count"] == 40
    assert not df.duplicated(["series_id", "season", "episode"]).any()


def test_orphan_series_are_ignored_and_counted(tmp_path):
    shows = make_shows(n_series=3)
    orphans = make_shows(n_series=2, seed=1)
    for i, o in enumerate(orphans):
        o["id"] = f"tt900000{i}"
    df, info = parse_archive(build_archive(tmp_path / "a.tar.gz", shows, orphans=orphans))
    assert df["series_id"].nunique() == 3
    assert info["n_orphan_series"] == 2


def test_corrupted_archive_raises(tmp_path):
    bad = tmp_path / "bad.tar.gz"
    bad.write_bytes(b"pas une archive")
    with pytest.raises(IngestionError):
        parse_archive(bad)


def test_archive_without_meta_raises(tmp_path):
    import tarfile

    path = tmp_path / "empty.tar.gz"
    with tarfile.open(path, "w:gz"):
        pass
    with pytest.raises(IngestionError, match=r"meta.json"):
        parse_archive(path)
