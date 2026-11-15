"""Génère les dashboards Grafana provisionnés (JSON) à partir de définitions Python.

Usage : python docker/grafana/build_dashboards.py
Les JSON produits dans docker/grafana/dashboards/ sont commités et chargés au démarrage
de Grafana (provisioning) : les tableaux de bord sont versionnés avec le code.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).parent / "dashboards"
PROM = {"type": "prometheus", "uid": "prometheus"}
PG = {"type": "grafana-postgresql-datasource", "uid": "postgres"}


class Layout:
    """Place les panneaux de gauche à droite sur une grille de 24 colonnes."""

    def __init__(self):
        self.x = self.y = self.row_h = 0
        self.panels: list[dict] = []
        self.next_id = 1

    def add(self, panel: dict, w: int, h: int) -> None:
        if self.x + w > 24:
            self.x, self.y, self.row_h = 0, self.y + self.row_h, 0
        panel |= {"id": self.next_id, "gridPos": {"x": self.x, "y": self.y, "w": w, "h": h}}
        self.next_id += 1
        self.panels.append(panel)
        self.x += w
        self.row_h = max(self.row_h, h)

    def row(self, title: str) -> None:
        self.x, self.y = 0, self.y + self.row_h
        self.row_h = 0
        self.add({"type": "row", "title": title, "collapsed": False, "panels": []}, 24, 1)
        self.x, self.y, self.row_h = 0, self.y + 1, 0


def prom(expr: str, legend: str = "", ref: str = "A") -> dict:
    return {"refId": ref, "datasource": PROM, "expr": expr, "legendFormat": legend, "range": True}


def sql(query: str, fmt: str = "table", ref: str = "A") -> dict:
    return {
        "refId": ref,
        "datasource": PG,
        "rawSql": " ".join(query.split()),
        "format": fmt,
        "rawQuery": True,
        "editorMode": "code",
    }


def stat(
    title,
    targets,
    unit="none",
    decimals=None,
    thresholds=None,
    text_mode="auto",
    desc="",
    color_mode="value",
):
    steps = thresholds or [{"color": "green", "value": None}]
    defaults = {
        "unit": unit,
        "thresholds": {"mode": "absolute", "steps": steps},
        "color": {"mode": "thresholds"},
    }
    if decimals is not None:
        defaults["decimals"] = decimals
    return {
        "type": "stat",
        "title": title,
        "description": desc,
        "targets": targets,
        "datasource": targets[0]["datasource"],
        "fieldConfig": {"defaults": defaults, "overrides": []},
        # Champs texte (SHA, version) : Grafana ne les affiche que si on les sélectionne.
        "options": {
            "reduceOptions": {
                "calcs": ["lastNotNull"],
                "fields": "/.*/" if text_mode == "value" else "",
                "values": False,
            },
            "textMode": text_mode,
            "colorMode": color_mode,
            "graphMode": "none",
        },
    }


def ts(title, targets, unit="none", desc="", stack=False, bars=False, decimals=None):
    custom = {
        "drawStyle": "bars" if bars else "line",
        "lineWidth": 2,
        "fillOpacity": 15,
        "showPoints": "auto",
        "spanNulls": True,
        "stacking": {"mode": "normal" if stack else "none", "group": "A"},
    }
    defaults = {"unit": unit, "custom": custom}
    if decimals is not None:
        defaults["decimals"] = decimals
    return {
        "type": "timeseries",
        "title": title,
        "description": desc,
        "targets": targets,
        "datasource": targets[0]["datasource"],
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {
            "legend": {
                "displayMode": "table",
                "placement": "right",
                "calcs": ["lastNotNull", "max"],
            },
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
    }


def table(title, targets, desc="", overrides=None):
    return {
        "type": "table",
        "title": title,
        "description": desc,
        "targets": targets,
        "datasource": targets[0]["datasource"],
        "fieldConfig": {
            "defaults": {"custom": {"align": "auto", "cellOptions": {"type": "auto"}}},
            "overrides": overrides or [],
        },
        "options": {
            "showHeader": True,
            "cellHeight": "sm",
            "footer": {"show": False, "reducer": ["sum"]},
        },
    }


def colored(field: str, mapping: dict[str, str]) -> dict:
    """Cellule colorée selon sa valeur (statuts)."""
    return {
        "matcher": {"id": "byName", "options": field},
        "properties": [
            {"id": "custom.cellOptions", "value": {"type": "color-background"}},
            {
                "id": "mappings",
                "value": [
                    {
                        "type": "value",
                        "options": {
                            k: {"color": c, "index": i} for i, (k, c) in enumerate(mapping.items())
                        },
                    }
                ],
            },
        ],
    }


def dashboard(uid, title, panels, time_from="now-6h", refresh="10s", desc=""):
    return {
        "uid": uid,
        "title": title,
        "description": desc,
        "tags": ["imdb-mlops"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "editable": False,
        "refresh": refresh,
        "time": {"from": time_from, "to": "now"},
        "panels": panels,
        "templating": {"list": []},
        "annotations": {"list": []},
    }


RED_GREEN = [
    {"color": "green", "value": None},
    {"color": "orange", "value": 1},
    {"color": "red", "value": 5},
]


# --- 1. Utilisation et performances de l'API (Prometheus) -----------------------------


def api_dashboard() -> dict:
    L = Layout()
    no_health = 'endpoint!="/health"'
    L.add(
        stat(
            "Débit",
            [prom(f"sum(rate(api_requests_total{{{no_health}}}[1m]))")],
            unit="reqps",
            decimals=1,
        ),
        4,
        4,
    )
    L.add(
        stat(
            "Latence p95 /predict",
            [
                prom(
                    "histogram_quantile(0.95, sum by (le) (rate(api_request_duration_seconds_bucket"
                    '{endpoint="/predict"}[1m])))'
                )
            ],
            unit="s",
            decimals=3,
            thresholds=[
                {"color": "green", "value": None},
                {"color": "orange", "value": 0.2},
                {"color": "red", "value": 0.5},
            ],
        ),
        4,
        4,
    )
    L.add(
        stat(
            "Erreurs 5xx",
            [
                prom(
                    '100 * (sum(rate(api_requests_total{status=~"5.."}[5m])) or vector(0)) / '
                    f"clamp_min(sum(rate(api_requests_total{{{no_health}}}[5m])), 1e-9)"
                )
            ],
            unit="percent",
            decimals=2,
            thresholds=RED_GREEN,
        ),
        4,
        4,
    )
    L.add(
        stat(
            "Erreurs 4xx (entrées)",
            [
                prom(
                    '100 * (sum(rate(api_requests_total{status=~"4.."}[5m])) or vector(0)) / '
                    f"clamp_min(sum(rate(api_requests_total{{{no_health}}}[5m])), 1e-9)"
                )
            ],
            unit="percent",
            decimals=1,
            desc="Entrées invalides et séries inconnues : erreurs côté client, attendues",
        ),
        4,
        4,
    )
    L.add(
        stat(
            "Taux de hit du cache",
            [
                prom(
                    'sum(rate(cache_requests_total{result="hit"}[5m])) / '
                    'clamp_min(sum(rate(cache_requests_total{result=~"hit|miss|bypass"}[5m])), 1e-9)'
                )
            ],
            unit="percentunit",
            decimals=1,
        ),
        4,
        4,
    )
    L.add(
        stat(
            "Modèle servi",
            [prom("count by (version) (model_info)", "v{{version}}")],
            text_mode="name",
            desc="Alias champion du registre MLflow chargé par l'API",
        ),
        4,
        4,
    )

    L.row("Trafic et latence")
    L.add(
        ts(
            "Requêtes par endpoint et statut",
            [
                prom(
                    f"sum by (endpoint, status) (rate(api_requests_total{{{no_health}}}[1m]))",
                    "{{endpoint}} {{status}}",
                )
            ],
            unit="reqps",
            stack=True,
        ),
        12,
        8,
    )
    L.add(
        ts(
            "Latence /predict (p50, p95, p99)",
            [
                prom(
                    "histogram_quantile(0.50, sum by (le) (rate(api_request_duration_seconds_bucket"
                    '{endpoint="/predict"}[1m])))',
                    "p50",
                    "A",
                ),
                prom(
                    "histogram_quantile(0.95, sum by (le) (rate(api_request_duration_seconds_bucket"
                    '{endpoint="/predict"}[1m])))',
                    "p95",
                    "B",
                ),
                prom(
                    "histogram_quantile(0.99, sum by (le) (rate(api_request_duration_seconds_bucket"
                    '{endpoint="/predict"}[1m])))',
                    "p99",
                    "C",
                ),
            ],
            unit="s",
        ),
        12,
        8,
    )
    L.add(
        ts(
            "Latence p95 par endpoint",
            [
                prom(
                    "histogram_quantile(0.95, sum by (le, endpoint) "
                    f"(rate(api_request_duration_seconds_bucket{{{no_health}}}[1m])))",
                    "{{endpoint}}",
                )
            ],
            unit="s",
        ),
        12,
        8,
    )
    L.add(
        ts(
            "Codes d'erreur",
            [
                prom(
                    'sum by (endpoint, status) (rate(api_requests_total{status=~"4..|5.."}[1m]))',
                    "{{endpoint}} {{status}}",
                ),
                prom(
                    "sum by (error) (rate(prediction_errors_total[1m]))",
                    "prédiction : {{error}}",
                    "B",
                ),
            ],
            unit="reqps",
        ),
        12,
        8,
    )

    L.row("Prédictions, cache et modèle")
    L.add(
        ts(
            "Prédictions servies (par version, cache)",
            [
                prom(
                    "sum by (model_version, cached) (rate(predictions_total[1m]))",
                    "v{{model_version}} cache={{cached}}",
                )
            ],
            unit="ops",
            stack=True,
        ),
        12,
        8,
    )
    L.add(
        ts(
            "Accès au cache",
            [prom("sum by (result) (rate(cache_requests_total[1m]))", "{{result}}")],
            unit="ops",
            stack=True,
            desc="bypass/error : Redis indisponible, l'API répond sans cache",
        ),
        12,
        8,
    )
    L.add(
        ts(
            "Note moyenne prédite",
            [
                prom(
                    "sum(rate(predicted_rating_sum[5m])) / clamp_min(sum(rate(predicted_rating_count[5m])),"
                    " 1e-9)",
                    "moyenne",
                )
            ],
            decimals=2,
        ),
        8,
        7,
    )
    L.add(
        ts(
            "Rechargements du modèle",
            [prom("sum by (result) (increase(model_reloads_total[5m]))", "{{result}}")],
            bars=True,
        ),
        8,
        7,
    )
    L.add(
        ts(
            "Traces en base : file et pertes",
            [
                prom("sum(db_traces_queue_size)", "en attente", "A"),
                prom(
                    "sum by (table) (increase(db_traces_dropped_total[5m]))",
                    "perdues {{table}}",
                    "B",
                ),
            ],
        ),
        8,
        7,
    )
    L.add(
        {
            "type": "alertlist",
            "title": "Alertes Prometheus actives",
            "options": {
                "viewMode": "list",
                "groupMode": "default",
                "maxItems": 20,
                "sortOrder": 1,
                "stateFilter": {"firing": True, "pending": True},
                "alertInstanceLabelFilter": "",
                "datasource": "Prometheus",
            },
        },
        24,
        6,
    )
    return dashboard(
        "imdb-api",
        "IMDb MLOps — API (utilisation et performances)",
        L.panels,
        time_from="now-30m",
        refresh="5s",
        desc="Trafic, latence, erreurs, cache et version du modèle servi.",
    )


# --- 2. Traitements : pipeline, données, modèles (PostgreSQL) -------------------------


def pipeline_dashboard() -> dict:
    L = Layout()
    L.add(
        stat(
            "Dernier snapshot validé",
            [
                sql(
                    "SELECT left(sha, 10) AS sha FROM data_snapshots WHERE quality_status = 'passed' "
                    "ORDER BY validated_at DESC LIMIT 1"
                )
            ],
            text_mode="value",
        ),
        4,
        4,
    )
    L.add(
        stat(
            "Épisodes (dernier snapshot)",
            [
                sql(
                    "SELECT n_rows FROM data_snapshots WHERE quality_status = 'passed' "
                    "ORDER BY validated_at DESC LIMIT 1"
                )
            ],
        ),
        4,
        4,
    )
    L.add(
        stat(
            "Champion actuel",
            [
                sql(
                    "SELECT 'v' || version AS champion FROM model_events "
                    "WHERE event IN ('promoted', 'rollback') ORDER BY created_at DESC LIMIT 1"
                )
            ],
            text_mode="value",
        ),
        4,
        4,
    )
    L.add(
        stat(
            "Étapes en échec (période)",
            [
                sql(
                    "SELECT count(*) FROM pipeline_steps WHERE status = 'failed' "
                    "AND $__timeFilter(started_at)"
                )
            ],
            thresholds=RED_GREEN,
        ),
        4,
        4,
    )
    L.add(
        stat(
            "Lots rejetés (période)",
            [
                sql(
                    "SELECT count(*) FROM data_snapshots WHERE quality_status = 'failed' "
                    "AND $__timeFilter(validated_at)"
                )
            ],
            thresholds=[{"color": "green", "value": None}, {"color": "orange", "value": 1}],
        ),
        4,
        4,
    )
    L.add(
        stat(
            "Alertes critiques (période)",
            [
                sql(
                    "SELECT count(*) FROM alerts WHERE severity = 'critical' "
                    "AND $__timeFilter(created_at)"
                )
            ],
            thresholds=RED_GREEN,
        ),
        4,
        4,
    )

    L.row("Exécutions de la chaîne (Prefect)")
    L.add(
        ts(
            "Durée des étapes",
            [
                sql(
                    "SELECT started_at AS time, step AS metric, duration_s AS value FROM pipeline_steps "
                    "WHERE $__timeFilter(started_at) ORDER BY 1",
                    "time_series",
                )
            ],
            unit="s",
            bars=True,
        ),
        12,
        8,
    )
    L.add(
        ts(
            "Étapes réussies / échouées",
            [
                sql(
                    "SELECT $__timeGroupAlias(started_at, '1h'), status AS metric, count(*) AS value "
                    "FROM pipeline_steps WHERE $__timeFilter(started_at) GROUP BY 1, 2 ORDER BY 1",
                    "time_series",
                )
            ],
            bars=True,
            stack=True,
        ),
        12,
        8,
    )
    L.add(
        table(
            "Dernières exécutions",
            [
                sql(
                    'SELECT started_at AS "début", flow, step AS "étape", status AS statut, '
                    'round(duration_s::numeric, 1) AS "durée (s)", left(run_id, 8) AS run, '
                    "coalesce(details->>'error', details->>'sha', details->>'version', '') AS \"détail\" "
                    "FROM pipeline_steps WHERE $__timeFilter(started_at) "
                    "ORDER BY started_at DESC LIMIT 50"
                )
            ],
            overrides=[colored("statut", {"success": "green", "failed": "red"})],
        ),
        24,
        9,
    )

    L.row("Données : snapshots et contrôles qualité")
    L.add(
        table(
            "Snapshots ingérés",
            [
                sql(
                    'SELECT validated_at AS "validé le", commit_date AS "commit source", '
                    'left(sha, 24) AS sha, n_rows AS "épisodes", n_series AS "séries", '
                    'quality_status AS statut, n_checks AS "contrôles", n_failed AS "échecs", '
                    "n_warnings AS warnings FROM data_snapshots ORDER BY validated_at DESC LIMIT 30"
                )
            ],
            overrides=[colored("statut", {"passed": "green", "failed": "red"})],
        ),
        14,
        9,
    )
    L.add(
        table(
            "Contrôles en échec (lots rejetés)",
            [
                sql(
                    'SELECT s.validated_at AS "date", left(s.sha, 24) AS sha, c->>\'name\' AS "contrôle", '
                    "c->>'details' AS \"détail\" FROM data_snapshots s, "
                    "json_array_elements(s.quality_report->'checks') c "
                    "WHERE (c->>'success')::boolean = false ORDER BY s.validated_at DESC LIMIT 30"
                )
            ],
        ),
        10,
        9,
    )

    L.row("Modèles : registre et décisions de mise en service")
    L.add(
        ts(
            "RMSE des versions entraînées vs champion",
            [
                sql(
                    "SELECT created_at AS time, 'candidat' AS metric, rmse AS value FROM model_events "
                    "WHERE event = 'registered' AND $__timeFilter(created_at) UNION ALL "
                    "SELECT created_at, 'champion (même test)', champion_rmse FROM model_events "
                    "WHERE event = 'registered' AND champion_rmse IS NOT NULL "
                    "AND $__timeFilter(created_at) UNION ALL "
                    "SELECT created_at, 'baseline', baseline_rmse FROM model_events "
                    "WHERE event = 'registered' AND $__timeFilter(created_at) ORDER BY 1",
                    "time_series",
                )
            ],
            decimals=4,
        ),
        10,
        9,
    )
    L.add(
        table(
            "Décisions du registre",
            [
                sql(
                    'SELECT created_at AS "date", event AS "événement", \'v\' || version AS version, '
                    "round(rmse::numeric, 4) AS rmse, round(champion_rmse::numeric, 4) AS champion, "
                    "reason AS motif FROM model_events WHERE event <> 'registered' "
                    "ORDER BY created_at DESC LIMIT 30"
                )
            ],
            overrides=[
                colored(
                    "événement", {"promoted": "green", "rejected": "orange", "rollback": "purple"}
                )
            ],
        ),
        14,
        9,
    )

    L.row("Alertes")
    L.add(
        table(
            "Journal des alertes",
            [
                sql(
                    'SELECT created_at AS "date", severity AS "sévérité", source, title AS titre, '
                    'left(message, 200) AS message, delivered AS "webhook" FROM alerts '
                    "WHERE $__timeFilter(created_at) ORDER BY created_at DESC LIMIT 100"
                )
            ],
            overrides=[
                colored("sévérité", {"critical": "red", "warning": "orange", "info": "blue"})
            ],
        ),
        24,
        10,
    )
    return dashboard(
        "imdb-pipeline",
        "IMDb MLOps — Traitements (données, pipeline, modèles)",
        L.panels,
        time_from="now-7d",
        refresh="30s",
        desc="Exécutions Prefect, contrôles qualité, décisions du registre, alertes.",
    )


# --- 3. Performance en production : J+1 et feedbacks (PostgreSQL) ----------------------


def live_dashboard() -> dict:
    L = Layout()
    window = "evaluated_at > now() - interval '7 days'"
    L.add(
        stat(
            "RMSE live (7 j)",
            [sql(f"SELECT sqrt(avg(error * error)) FROM prediction_outcomes WHERE {window}")],
            decimals=3,
            thresholds=[
                {"color": "green", "value": None},
                {"color": "orange", "value": 0.7},
                {"color": "red", "value": 0.8},
            ],
            desc="Seuil d'alerte de dérive : 0,8 (configs/config.yaml)",
        ),
        4,
        4,
    )
    L.add(
        stat(
            "RMSE nouveaux épisodes",
            [
                sql(
                    f"SELECT sqrt(avg(error * error)) FROM prediction_outcomes WHERE {window} "
                    "AND is_new_episode"
                )
            ],
            decimals=3,
        ),
        4,
        4,
    )
    L.add(
        stat(
            "MAE live (7 j)",
            [sql(f"SELECT avg(abs_error) FROM prediction_outcomes WHERE {window}")],
            decimals=3,
        ),
        4,
        4,
    )
    L.add(
        stat(
            "Biais (prédit - réel)",
            [sql(f"SELECT avg(error) FROM prediction_outcomes WHERE {window}")],
            decimals=3,
        ),
        4,
        4,
    )
    L.add(
        stat(
            "Prédictions évaluées (7 j)",
            [sql(f"SELECT count(*) FROM prediction_outcomes WHERE {window}")],
        ),
        4,
        4,
    )
    L.add(
        stat(
            "MAE des feedbacks",
            [
                sql(
                    "SELECT avg(abs(f.rating - p.predicted_rating)) FROM feedback f "
                    "JOIN predictions p ON p.id = f.prediction_id WHERE $__timeFilter(f.created_at)"
                )
            ],
            decimals=3,
        ),
        4,
        4,
    )

    L.row("Prédit vs réel (snapshot J+1)")
    L.add(
        ts(
            "RMSE live par type d'épisode",
            [
                sql(
                    "SELECT $__timeGroupAlias(evaluated_at, '1h'), CASE WHEN is_new_episode THEN "
                    "'nouvel épisode' ELSE 'épisode connu' END AS metric, sqrt(avg(error * error)) AS value "
                    "FROM prediction_outcomes WHERE $__timeFilter(evaluated_at) GROUP BY 1, 2 ORDER BY 1",
                    "time_series",
                )
            ],
            decimals=3,
            bars=True,
        ),
        12,
        8,
    )
    L.add(
        table(
            "Par version du modèle et type d'épisode",
            [
                sql(
                    "SELECT 'v' || model_version AS version, CASE WHEN is_new_episode THEN 'nouveau' "
                    "ELSE 'connu' END AS \"épisode\", count(*) AS n, "
                    "round(avg(abs_error)::numeric, 3) AS mae, "
                    "round(sqrt(avg(error * error))::numeric, 3) AS rmse, "
                    "round(avg(error)::numeric, 3) AS biais FROM prediction_outcomes "
                    "WHERE $__timeFilter(evaluated_at) GROUP BY 1, 2 ORDER BY 1, 2"
                )
            ],
        ),
        12,
        8,
    )
    L.add(
        {
            "type": "histogram",
            "title": "Distribution des erreurs (prédit - réel)",
            "datasource": PG,
            "targets": [
                sql(
                    'SELECT error AS "erreur" FROM prediction_outcomes '
                    "WHERE $__timeFilter(evaluated_at)"
                )
            ],
            "options": {"bucketSize": 0.1, "combine": False, "legend": {"showLegend": False}},
            "fieldConfig": {"defaults": {"custom": {"fillOpacity": 70}}, "overrides": []},
        },
        12,
        8,
    )
    L.add(
        table(
            "Plus grosses erreurs",
            [
                sql(
                    'SELECT evaluated_at AS "évalué le", series_id AS "série", season AS saison, '
                    'episode AS "épisode", round(predicted_rating::numeric, 2) AS "prédit", '
                    'actual_rating AS "réel", round(error::numeric, 2) AS erreur, '
                    "is_new_episode AS nouveau, 'v' || model_version AS version FROM prediction_outcomes "
                    "WHERE $__timeFilter(evaluated_at) ORDER BY abs_error DESC LIMIT 25"
                )
            ],
        ),
        12,
        8,
    )

    L.row("Utilisation du service (traces en base)")
    L.add(
        ts(
            "Prédictions par statut",
            [
                sql(
                    "SELECT $__timeGroupAlias(created_at, '5m'), status || coalesce(' : ' || error, '') "
                    "AS metric, count(*) AS value FROM predictions WHERE $__timeFilter(created_at) "
                    "GROUP BY 1, 2 ORDER BY 1",
                    "time_series",
                )
            ],
            bars=True,
            stack=True,
        ),
        12,
        8,
    )
    L.add(
        ts(
            "Durée moyenne des requêtes",
            [
                sql(
                    "SELECT $__timeGroupAlias(created_at, '5m'), path AS metric, avg(duration_ms) AS value "
                    "FROM api_requests WHERE $__timeFilter(created_at) GROUP BY 1, 2 ORDER BY 1",
                    "time_series",
                )
            ],
            unit="ms",
        ),
        12,
        8,
    )
    L.add(
        table(
            "Requêtes par endpoint et code",
            [
                sql(
                    "SELECT path AS endpoint, status_code AS code, coalesce(error, '') AS erreur, "
                    'count(*) AS n, round(avg(duration_ms)::numeric, 1) AS "moy. (ms)", '
                    "round((percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms))::numeric, 1) "
                    'AS "p95 (ms)" FROM api_requests WHERE $__timeFilter(created_at) '
                    "GROUP BY 1, 2, 3 ORDER BY n DESC"
                )
            ],
        ),
        12,
        8,
    )
    L.add(
        table(
            "Derniers feedbacks",
            [
                sql(
                    'SELECT f.created_at AS "date", p.series_id AS "série", p.season AS saison, '
                    'p.episode AS "épisode", round(p.predicted_rating::numeric, 2) AS "prédit", '
                    "f.rating AS \"feedback\", 'v' || p.model_version AS version, f.comment AS commentaire "
                    "FROM feedback f JOIN predictions p ON p.id = f.prediction_id "
                    "WHERE $__timeFilter(f.created_at) ORDER BY f.created_at DESC LIMIT 25"
                )
            ],
        ),
        12,
        8,
    )
    return dashboard(
        "imdb-live",
        "IMDb MLOps — Performance en production (J+1, feedbacks)",
        L.panels,
        time_from="now-7d",
        refresh="30s",
        desc="Comparaison des prédictions aux notes publiées le lendemain.",
    )


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, build in [
        ("api", api_dashboard),
        ("pipeline", pipeline_dashboard),
        ("live", live_dashboard),
    ]:
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(build(), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{path} : {len(build()['panels'])} panneaux")
