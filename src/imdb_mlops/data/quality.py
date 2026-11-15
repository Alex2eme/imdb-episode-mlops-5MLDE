"""Contrôles qualité des données avant prétraitement.

Deux niveaux :
- **critique** : un échec lève `DataQualityError`, ce qui interrompt la chaîne (pas
  d'entraînement sur des données non conformes) ;
- **warning** : remonté dans le rapport et en alerte, sans bloquer.

Les contrôles combinent Great Expectations (expectations natives et personnalisées) et
des règles métier propres au projet (comparaison au snapshot précédent, cohérence avec
le manifeste source), que GX n'exprime pas directement.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from imdb_mlops.config import ROOT_DIR, AppConfig
from imdb_mlops.data.ingest import EPISODE_COLUMNS, SnapshotManifest

os.environ.setdefault("GX_ANALYTICS_ENABLED", "False")
import great_expectations as gx
import great_expectations.expectations as gxe

log = logging.getLogger(__name__)


# --- Expectations personnalisées (règles métier encapsulées et nommées) ---------------


class ExpectValidImdbRating(gxe.ExpectColumnValuesToBeBetween):
    """Une note IMDb est comprise entre 1 et 10."""

    column: str = "rating"
    min_value: float = 1.0
    max_value: float = 10.0
    description: str = "Note IMDb comprise entre 1 et 10"


class ExpectImdbTconst(gxe.ExpectColumnValuesToMatchRegex):
    """Identifiant IMDb de série au format tt + chiffres."""

    column: str = "series_id"
    regex: str = r"^tt\d{7,9}$"
    description: str = "Identifiant de série au format IMDb (tt0000000)"


class ExpectUniqueEpisodeKey(gxe.ExpectCompoundColumnsToBeUnique):
    """Un épisode est identifié de façon unique par (série, saison, numéro)."""

    # Une liste (et non un tuple) : GX normalise les noms de colonnes un par un.
    column_list: list[str] = ["series_id", "season", "episode"]  # noqa: RUF012
    description: str = "Clé (series_id, season, episode) unique"


class ExpectMinimumVotes(gxe.ExpectColumnValuesToBeBetween):
    """Le dataset source garantit un nombre minimal de votes par épisode."""

    column: str = "votes"
    description: str = "Nombre de votes au moins égal au seuil du dataset source"


# --- Rapport --------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    success: bool
    severity: str  # "critical" | "warning"
    details: dict = field(default_factory=dict)


@dataclass
class QualityReport:
    snapshot_sha: str
    validated_at: str
    success: bool
    checks: list[CheckResult]

    @property
    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.success and c.severity == "critical"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.success and c.severity == "warning"]

    def summary(self) -> str:
        n_ok = sum(c.success for c in self.checks)
        lines = [f"{n_ok}/{len(self.checks)} contrôles OK — snapshot {self.snapshot_sha[:10]}"]
        lines += [f"  ✗ [CRITIQUE] {c.name} : {c.details}" for c in self.failed]
        lines += [f"  ⚠ [WARNING]  {c.name} : {c.details}" for c in self.warnings]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["n_checks"] = len(self.checks)
        d["n_failed"] = len(self.failed)
        d["n_warnings"] = len(self.warnings)
        return d


class DataQualityError(RuntimeError):
    """Données non conformes : la chaîne doit s'arrêter."""

    def __init__(self, report: QualityReport):
        self.report = report
        super().__init__("Contrôle qualité en échec\n" + report.summary())


# --- Contrôles --------------------------------------------------------------------------


def _build_suite(cfg: AppConfig) -> tuple[list, list]:
    q = cfg.quality
    critical = [
        gxe.ExpectTableColumnsToMatchSet(column_set=EPISODE_COLUMNS, exact_match=True),
        *[
            gxe.ExpectColumnValuesToNotBeNull(column=c)
            for c in ("series_id", "season", "episode", "rating", "votes")
        ],
        ExpectValidImdbRating(),
        ExpectImdbTconst(),
        ExpectUniqueEpisodeKey(),
        ExpectMinimumVotes(min_value=cfg.data.min_votes),
        gxe.ExpectColumnValuesToBeBetween(column="season", min_value=0, max_value=q.max_season),
        gxe.ExpectColumnValuesToBeBetween(column="episode", min_value=0, max_value=q.max_episode),
        gxe.ExpectTableRowCountToBeBetween(min_value=q.min_rows, max_value=q.max_rows),
        gxe.ExpectColumnUniqueValueCountToBeBetween(column="series_id", min_value=q.min_series),
    ]
    warning = [
        gxe.ExpectColumnMeanToBeBetween(
            column="rating", min_value=q.rating_mean_range[0], max_value=q.rating_mean_range[1]
        ),
        gxe.ExpectColumnValuesToNotBeNull(column="series_title"),
    ]
    return critical, warning


def _run_gx(df: pd.DataFrame, expectations: list, severity: str) -> list[CheckResult]:
    context = gx.get_context(mode="ephemeral")
    with contextlib.suppress(Exception):  # option purement cosmétique
        context.variables.progress_bars = {"globally": False}
    batch_def = (
        context.data_sources.add_pandas(f"src_{severity}")
        .add_dataframe_asset("episodes")
        .add_batch_definition_whole_dataframe("snapshot")
    )
    suite = context.suites.add(gx.ExpectationSuite(name=f"episodes_{severity}"))
    for exp in expectations:
        suite.add_expectation(exp)
    result = batch_def.get_batch(batch_parameters={"dataframe": df}).validate(suite)

    checks = []
    for r in result.results:
        cfg = r.expectation_config
        column = cfg.kwargs.get("column") or cfg.kwargs.get("column_list") or ""
        details = {
            k: v
            for k, v in (r.result or {}).items()
            if k in ("observed_value", "unexpected_count", "unexpected_percent")
        }
        if not r.success and (sample := (r.result or {}).get("partial_unexpected_list")):
            details["sample"] = [str(x) for x in sample[:5]]
        if message := _exception_message(r.exception_info):
            details["exception"] = message
        name = f"{cfg.type}({column})" if column else cfg.type
        checks.append(
            CheckResult(
                name=name, success=bool(r.success), severity=severity, details=_jsonable(details)
            )
        )
    return checks


def _business_checks(
    df: pd.DataFrame,
    manifest: SnapshotManifest,
    cfg: AppConfig,
    previous_n_rows: int | None,
    declared_ep_count: int | None,
) -> list[CheckResult]:
    """Règles métier hors Great Expectations."""
    q = cfg.quality
    checks = []

    if previous_n_rows:
        change = (len(df) - previous_n_rows) / previous_n_rows
        checks.append(
            CheckResult(
                "row_count_change_vs_previous_snapshot",
                abs(change) <= q.max_row_change,
                "critical",
                {
                    "previous": previous_n_rows,
                    "current": len(df),
                    "change": round(change, 4),
                    "max_change": q.max_row_change,
                },
            )
        )

    if declared_ep_count:
        gap = abs(len(df) - declared_ep_count) / declared_ep_count
        checks.append(
            CheckResult(
                "row_count_matches_source_manifest",
                gap <= q.max_declared_gap,
                "critical",
                {"declared": declared_ep_count, "parsed": len(df), "gap": round(gap, 4)},
            )
        )

    n_total = manifest.n_series + manifest.n_orphan_series
    orphan_ratio = manifest.n_orphan_series / n_total if n_total else 0.0
    checks.append(
        CheckResult(
            "orphan_series_files_ratio",
            orphan_ratio <= q.max_orphan_ratio,
            "warning",
            {"orphans": manifest.n_orphan_series, "ratio": round(orphan_ratio, 4)},
        )
    )
    return checks


def _exception_message(info: dict | None) -> str | None:
    """GX imbrique parfois l'erreur sous un identifiant de métrique."""
    if not info:
        return None
    if info.get("raised_exception"):
        return info.get("exception_message")
    for value in info.values():
        if isinstance(value, dict) and value.get("raised_exception"):
            return value.get("exception_message")
    return None


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=str))


def validate(
    df: pd.DataFrame,
    manifest: SnapshotManifest,
    cfg: AppConfig,
    previous_n_rows: int | None = None,
    declared_ep_count: int | None = None,
    raise_on_failure: bool = True,
) -> QualityReport:
    """Exécute tous les contrôles, écrit le rapport JSON et lève en cas d'échec critique."""
    critical, warning = _build_suite(cfg)
    checks = _run_gx(df, critical, "critical") + _run_gx(df, warning, "warning")
    checks += _business_checks(df, manifest, cfg, previous_n_rows, declared_ep_count)

    report = QualityReport(
        snapshot_sha=manifest.sha,
        validated_at=datetime.now(UTC).isoformat(),
        success=not any(not c.success and c.severity == "critical" for c in checks),
        checks=checks,
    )
    path = write_report(report, cfg)
    log.info("Rapport qualité : %s\n%s", path, report.summary())
    if raise_on_failure and not report.success:
        raise DataQualityError(report)
    return report


def write_report(report: QualityReport, cfg: AppConfig) -> Path:
    reports_dir = ROOT_DIR / cfg.data.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"quality_{report.snapshot_sha}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path
