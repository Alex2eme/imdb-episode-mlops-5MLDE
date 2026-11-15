"""Ingestion reproductible du dataset IMDb épisodes.

Principe : on résout un commit du dépôt source (dernier commit de la branche, ou SHA figé
dans la config), on télécharge l'archive *à ce commit*, puis on l'aplatit en un Parquet
d'une ligne par épisode. Le SHA identifie de façon immuable les données utilisées.
"""

from __future__ import annotations

import hashlib
import json
import logging
import tarfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from imdb_mlops.config import ROOT_DIR, AppConfig, get_settings

log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
CODELOAD = "https://codeload.github.com"

EPISODE_COLUMNS = [
    "series_id",
    "series_title",
    "series_votes",
    "season",
    "episode",
    "rating",
    "votes",
]


class IngestionError(RuntimeError):
    """Échec de récupération ou de lecture de la source (réseau, archive corrompue...)."""


@dataclass
class SnapshotManifest:
    """Métadonnées d'un snapshot ingéré, tracées dans MLflow et PostgreSQL."""

    sha: str
    source_repo: str
    commit_date: str
    built_at: str | None
    ingested_at: str
    n_rows: int
    n_series: int
    n_orphan_series: int
    declared_ep_count: int | None
    parquet_path: str
    parquet_sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "imdb-mlops"}
    if token := get_settings().github_token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def resolve_commit(repo: str, ref: str, timeout: float = 15.0) -> tuple[str, str]:
    """Retourne (sha, date du commit) pour une branche ou un SHA."""
    url = f"{GITHUB_API}/repos/{repo}/commits/{ref}"
    try:
        r = httpx.get(url, headers=_headers(), timeout=timeout)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        raise IngestionError(f"Impossible de résoudre {repo}@{ref} : {exc}") from exc
    data = r.json()
    return data["sha"], data["commit"]["committer"]["date"]


def download_archive(repo: str, sha: str, dest_dir: Path, timeout: float = 120.0) -> Path:
    """Télécharge l'archive du dépôt au commit donné (mise en cache par SHA)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / f"{sha}.tar.gz"
    if target.exists() and target.stat().st_size > 0:
        log.info("Archive déjà en cache : %s", target)
        return target
    url = f"{CODELOAD}/{repo}/tar.gz/{sha}"
    tmp = target.with_suffix(".part")
    try:
        with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    except httpx.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise IngestionError(f"Téléchargement échoué ({url}) : {exc}") from exc
    tmp.replace(target)
    return target


def parse_archive(archive: Path) -> tuple[pd.DataFrame, dict]:
    """Aplatit l'archive en DataFrame (une ligne par épisode).

    `data/meta.json` fait foi pour la liste des séries : le dépôt source ne supprime pas
    les fichiers des séries sorties du top, ces fichiers « orphelins » sont ignorés.
    Le parsing est volontairement tolérant (pas de cast strict) : la conformité est
    vérifiée ensuite par les contrôles qualité, qui décident de bloquer ou non.
    """
    meta: dict | None = None
    shows: dict[str, dict] = {}
    try:
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                parts = member.name.split("/")
                fh = tar.extractfile(member)
                if fh is None:
                    continue
                if parts[-2:] == ["data", "meta.json"]:
                    meta = json.load(fh)
                elif len(parts) >= 3 and parts[-3:-1] == ["data", "shows"]:
                    show = json.load(fh)
                    shows[show.get("id", parts[-1].removesuffix(".json"))] = show
    except (tarfile.TarError, json.JSONDecodeError, OSError) as exc:
        raise IngestionError(f"Archive illisible {archive} : {exc}") from exc

    if meta is None:
        raise IngestionError("data/meta.json absent de l'archive")

    listed = {s["id"] for s in meta.get("shows", [])}
    rows = []
    for sid in listed & shows.keys():
        show = shows[sid]
        for season, episodes in (show.get("seasons") or {}).items():
            for episode, values in (episodes or {}).items():
                rows.append(
                    (
                        sid,
                        show.get("title"),
                        show.get("votes"),
                        season,
                        episode,
                        values.get("r"),
                        values.get("v"),
                    )
                )
    df = pd.DataFrame(rows, columns=EPISODE_COLUMNS)
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["episode"] = pd.to_numeric(df["episode"], errors="coerce").astype("Int64")
    for col in ("rating",):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in ("votes", "series_votes"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = df.sort_values(["series_id", "season", "episode"], kind="stable").reset_index(drop=True)

    info = {
        "built_at": meta.get("built"),
        "declared_show_count": meta.get("show_count"),
        "declared_ep_count": meta.get("ep_count"),
        "n_orphan_series": len(shows.keys() - listed),
        "n_missing_series": len(listed - shows.keys()),
    }
    return df, info


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ingest(cfg: AppConfig, ref: str | None = None) -> tuple[pd.DataFrame, SnapshotManifest]:
    """Ingestion complète : résolution du commit → archive → Parquet + manifeste."""
    ref = ref or cfg.data.pinned_sha or cfg.data.source_branch
    sha, commit_date = resolve_commit(cfg.data.source_repo, ref)
    log.info("Snapshot source : %s@%s (%s)", cfg.data.source_repo, sha[:10], commit_date)

    raw_dir = ROOT_DIR / cfg.data.raw_dir
    archive = download_archive(cfg.data.source_repo, sha, raw_dir / "archives")
    df, info = parse_archive(archive)

    parquet = raw_dir / f"episodes_{sha}.parquet"
    df.to_parquet(parquet, index=False)
    manifest = SnapshotManifest(
        sha=sha,
        source_repo=cfg.data.source_repo,
        commit_date=commit_date,
        built_at=info["built_at"],
        ingested_at=datetime.now(UTC).isoformat(),
        n_rows=len(df),
        n_series=int(df["series_id"].nunique()),
        n_orphan_series=info["n_orphan_series"],
        declared_ep_count=info["declared_ep_count"],
        parquet_path=str(parquet.relative_to(ROOT_DIR)),
        parquet_sha256=_sha256(parquet),
    )
    parquet.with_suffix(".manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
    )
    log.info("%d épisodes, %d séries → %s", manifest.n_rows, manifest.n_series, parquet)
    return df, manifest


def load_snapshot(path: Path | str) -> tuple[pd.DataFrame, SnapshotManifest]:
    """Recharge un snapshot déjà ingéré (Parquet + manifeste)."""
    path = Path(path)
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    return pd.read_parquet(path), SnapshotManifest(**manifest)
