"""Entraînement traçable : snapshot validé → features → tuning parallèle → MLflow → registre."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import matplotlib
import mlflow
import numpy as np
import optuna
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities import Metric, Param
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from imdb_mlops import __version__, db
from imdb_mlops.config import ROOT_DIR, AppConfig
from imdb_mlops.features import FEATURES, TARGET, build_features, temporal_split
from imdb_mlops.training import registry

matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
INT_PARAMS = {"num_leaves", "min_child_samples"}
LOG_PARAMS = {"learning_rate"}


class TrainingBlockedError(RuntimeError):
    """Entraînement refusé : pas de snapshot validé ou intégrité compromise."""


@dataclass
class TrainResult:
    run_id: str
    model_version: str
    data_sha: str
    rmse: float
    baseline_rmse: float
    champion_rmse: float | None
    promoted: bool
    reason: str
    best_params: dict


# --- Provenance -----------------------------------------------------------------------


def code_version() -> dict[str, str]:
    """SHA du code : variable GIT_SHA (image Docker) sinon dépôt Git local."""
    sha, dirty = os.environ.get("GIT_SHA", ""), "unknown"
    if not sha or sha == "unknown":
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True, stderr=subprocess.DEVNULL
            ).strip()
            dirty = str(
                bool(
                    subprocess.check_output(
                        ["git", "status", "--porcelain"], cwd=ROOT_DIR, text=True
                    ).strip()
                )
            ).lower()
        except (OSError, subprocess.CalledProcessError):
            sha = "unknown"
    return {"git_sha": sha, "git_dirty": dirty, "package_version": __version__}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_validated_snapshot(sha: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Charge un snapshot ayant passé les contrôles qualité, intégrité vérifiée."""
    snap = db.get_snapshot(sha) if sha else db.last_valid_snapshot()
    if snap is None:
        raise TrainingBlockedError(
            f"Snapshot {sha or '(dernier validé)'} introuvable : lancer d'abord l'étape data"
        )
    if snap["quality_status"] != "passed":
        raise TrainingBlockedError(
            f"Snapshot {snap['sha'][:10]} non conforme : entraînement refusé"
        )
    path = ROOT_DIR / snap["parquet_path"]
    if not path.exists():
        raise TrainingBlockedError(f"Parquet absent : {path} (relancer l'ingestion de ce SHA)")
    if _sha256(path) != snap["parquet_sha256"]:
        raise TrainingBlockedError(f"Empreinte du Parquet modifiée : {path}")
    return pd.read_parquet(path), snap


# --- Modèle ---------------------------------------------------------------------------


def _fit(params, X_tr, y_tr, X_val, y_val, n_threads: int, cfg: AppConfig):
    model = lgb.LGBMRegressor(
        n_estimators=cfg.training.n_estimators,
        random_state=cfg.project.seed,
        n_jobs=n_threads,
        subsample_freq=1,
        verbose=-1,
        **params,
    )
    evals: dict = {}
    model.fit(
        X_tr,
        y_tr,
        eval_X=(X_val,),
        eval_y=(y_val,),
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(cfg.training.early_stopping_rounds, verbose=False),
            lgb.record_evaluation(evals),
        ],
    )
    return model, evals["valid_0"]["rmse"]


def _suggest(trial: optuna.Trial, space: dict[str, tuple[float, float]]) -> dict:
    params = {}
    for name, (low, high) in space.items():
        if name in INT_PARAMS:
            params[name] = trial.suggest_int(name, int(low), int(high), log=name == "num_leaves")
        else:
            params[name] = trial.suggest_float(name, low, high, log=name in LOG_PARAMS)
    return params


def _log_trial(client, exp_id, parent_id, trial, params, val_rmse, duration, attempts=3):
    """Run enfant MLflow pour un trial, avec nouvelles tentatives (réseau, serveur chargé)."""
    for attempt in range(1, attempts + 1):
        try:
            child = client.create_run(
                exp_id,
                tags={
                    "mlflow.parentRunId": parent_id,
                    "mlflow.runName": f"trial-{trial.number:03d}",
                    "trial_number": str(trial.number),
                },
            )
            ts = int(time.time() * 1000)
            client.log_batch(
                child.info.run_id,
                metrics=[
                    Metric("val_rmse", val_rmse, ts, 0),
                    Metric("best_iteration", trial.user_attrs["best_iteration"], ts, 0),
                    Metric("duration_s", duration, ts, 0),
                ],
                params=[Param(k, str(v)) for k, v in params.items()],
            )
            client.set_terminated(child.info.run_id)
            return
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(2 * attempt)


def _tune(
    cfg, X_tr, y_tr, X_val, y_val, client: MlflowClient, parent_run
) -> tuple[dict, pd.DataFrame]:
    """Recherche Optuna : `parallel_trials` trials simultanés, `threads_per_trial` threads chacun.

    Chaque trial est tracé comme run enfant MLflow (via le client, sûr en multi-thread).
    """
    t = cfg.training
    exp_id = parent_run.info.experiment_id

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, t.search_space)
        start = time.time()
        model, curve = _fit(params, X_tr, y_tr, X_val, y_val, t.threads_per_trial, cfg)
        val_rmse = float(min(curve))
        trial.set_user_attr("best_iteration", int(model.best_iteration_ or len(curve)))
        try:
            _log_trial(
                client, exp_id, parent_run.info.run_id, trial, params, val_rmse, time.time() - start
            )
        except Exception as exc:  # le traçage ne doit pas faire échouer le tuning
            log.warning("Trial %d non tracé dans MLflow : %s", trial.number, exc)
        return val_rmse

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.project.seed),
        study_name=f"tuning-{parent_run.info.run_id[:8]}",
    )
    study.optimize(objective, n_trials=t.n_trials, n_jobs=t.parallel_trials)
    history = study.trials_dataframe(attrs=("number", "value", "params", "duration", "state"))
    return study.best_params, history


# --- Évaluation -----------------------------------------------------------------------


def _metrics(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    return {
        f"{prefix}rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}r2": float(r2_score(y_true, y_pred)),
    }


def _segment_metrics(test: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    """RMSE par segment : où le modèle se trompe-t-il ?"""
    err = (test[TARGET].to_numpy() - pred) ** 2
    hist = test["series_ep_index"]
    segments = {
        "season_premiere": test["is_season_premiere"] == 1,
        "mid_season": test["is_season_premiere"] == 0,
        "history_lt_10": hist < 10,
        "history_10_50": (hist >= 10) & (hist < 50),
        "history_ge_50": hist >= 50,
    }
    out = {}
    for name, mask in segments.items():
        if mask.any():
            out[f"test_rmse_{name}"] = float(np.sqrt(err[mask.to_numpy()].mean()))
            out[f"test_n_{name}"] = int(mask.sum())
    return out


def _figures(model, test, pred, history: pd.DataFrame | None) -> dict[str, plt.Figure]:
    figs = {}
    imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot.barh(ax=ax, title="Importance des features (gain)")
    figs["feature_importance.png"] = fig

    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(test[TARGET], pred, gridsize=40, mincnt=1, bins="log", cmap="viridis")
    ax.plot([1, 10], [1, 10], "r--", lw=1)
    ax.set(xlabel="Note réelle", ylabel="Note prédite", title="Prédiction vs réel (test)")
    fig.colorbar(hb, ax=ax)
    figs["pred_vs_actual.png"] = fig

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pred - test[TARGET], bins=80)
    ax.set(xlabel="Résidu (prédit - réel)", title="Distribution des résidus (test)")
    figs["residuals.png"] = fig

    if history is not None and not history.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(history["number"], history["value"], s=12, label="trial")
        ax.plot(history["number"], history["value"].cummin(), "r-", label="meilleur")
        ax.set(xlabel="Trial", ylabel="RMSE validation", title="Recherche Optuna")
        ax.legend()
        figs["optuna_history.png"] = fig
    return figs


def _champion_rmse(cfg, X_test, y_test) -> float | None:
    """Champion actuel réévalué sur le *même* jeu de test que le challenger."""
    client = MlflowClient()
    champ = registry.get_alias_version(
        client, cfg.training.registered_model_name, registry.CHAMPION
    )
    if champ is None:
        return None
    try:
        model = mlflow.lightgbm.load_model(
            f"models:/{cfg.training.registered_model_name}/{champ.version}"
        )
        return float(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))
    except Exception as exc:
        log.warning("Réévaluation du champion impossible (%s), RMSE enregistrée utilisée", exc)
        rmse = champ.tags.get("rmse")
        return float(rmse) if rmse else None


# --- Orchestration de l'entraînement --------------------------------------------------


def train(
    cfg: AppConfig,
    snapshot_sha: str | None = None,
    params: dict | None = None,
    run_name: str | None = None,
    trigger: str = "manual",
    extra_tags: dict | None = None,
    register: bool = True,
) -> TrainResult:
    t0 = time.time()
    df, snap = load_validated_snapshot(snapshot_sha)
    sha = snap["sha"]
    feats = build_features(df)
    train_df, test_df = temporal_split(feats, cfg.training.test_size)
    rng = np.random.default_rng(cfg.project.seed)
    is_val = rng.random(len(train_df)) < cfg.training.val_size
    fit_df, val_df = train_df[~is_val], train_df[is_val]
    X_fit, y_fit = fit_df[FEATURES], fit_df[TARGET]
    X_val, y_val = val_df[FEATURES], val_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]
    t_features = time.time() - t0

    mlflow.set_experiment(cfg.training.experiment_name)
    client = MlflowClient()
    version = code_version()
    source_url = f"https://github.com/{snap['source_repo']}/tree/{sha}"

    with mlflow.start_run(run_name=run_name or f"train-{sha[:8]}") as run:
        mlflow.set_tags(
            {
                "data.sha": sha,
                "data.source": source_url,
                "data.parquet_sha256": snap["parquet_sha256"],
                "data.commit_date": str(snap["commit_date"]),
                "data.quality_status": snap["quality_status"],
                "code.git_sha": version["git_sha"],
                "code.git_dirty": version["git_dirty"],
                "code.package_version": version["package_version"],
                "trigger": trigger,
                "seed": str(cfg.project.seed),
                **(extra_tags or {}),
            }
        )
        mlflow.log_input(
            mlflow.data.from_pandas(
                df, source=source_url, targets=TARGET, name=f"imdb-episodes@{sha[:10]}"
            ),
            context="training",
        )
        mlflow.log_params(
            {
                "seed": cfg.project.seed,
                "test_size": cfg.training.test_size,
                "val_size": cfg.training.val_size,
                "split": "temporel par série (derniers épisodes en test)",
                "n_features": len(FEATURES),
                "features": ",".join(FEATURES),
                "n_trials": cfg.training.n_trials if params is None else 0,
                "parallel_trials": cfg.training.parallel_trials,
                "threads_per_trial": cfg.training.threads_per_trial,
                "n_estimators_max": cfg.training.n_estimators,
                "early_stopping_rounds": cfg.training.early_stopping_rounds,
            }
        )
        mlflow.log_metrics(
            {
                "n_rows_total": len(df),
                "n_rows_fit": len(fit_df),
                "n_rows_val": len(val_df),
                "n_rows_test": len(test_df),
                "n_series": int(df["series_id"].nunique()),
                "feature_build_s": t_features,
            }
        )
        mlflow.log_artifact(str(ROOT_DIR / "configs" / "config.yaml"), "config")
        if snap.get("quality_report"):
            mlflow.log_dict(snap["quality_report"], "data/quality_report.json")

        # Baselines : la moyenne globale et la moyenne des épisodes précédents de la série.
        global_mean = float(y_fit.mean())
        baseline_pred = test_df["series_prev_mean"].fillna(global_mean).to_numpy()
        baseline = _metrics(y_test, baseline_pred, "baseline_")
        mlflow.log_metrics(
            baseline | _metrics(y_test, np.full(len(y_test), global_mean), "global_mean_")
        )

        # Tuning (ou paramètres imposés, pour reproduire un run).
        t1 = time.time()
        history = None
        if params is None and cfg.training.n_trials > 0:
            best_params, history = _tune(cfg, X_fit, y_fit, X_val, y_val, client, run)
        else:
            best_params = params or {}
        mlflow.log_metric("tuning_duration_s", time.time() - t1)
        mlflow.log_params({f"hp_{k}": v for k, v in best_params.items()})

        # Modèle final : tous les cœurs, meilleurs hyperparamètres.
        t2 = time.time()
        n_threads = cfg.training.parallel_trials * cfg.training.threads_per_trial
        model, curve = _fit(best_params, X_fit, y_fit, X_val, y_val, n_threads, cfg)
        mlflow.log_metric("training_duration_s", time.time() - t2)
        mlflow.log_metric("best_iteration", model.best_iteration_ or len(curve))
        ts = int(time.time() * 1000)
        curve_metrics = [Metric("val_rmse_curve", float(v), ts, i) for i, v in enumerate(curve)]
        for i in range(0, len(curve_metrics), 1000):
            client.log_batch(run.info.run_id, metrics=curve_metrics[i : i + 1000])

        pred = model.predict(X_test)
        test_metrics = _metrics(y_test, pred, "test_")
        mlflow.log_metrics(test_metrics | _segment_metrics(test_df, pred))
        mlflow.log_metric(
            "test_rmse_gain_vs_baseline", baseline["baseline_rmse"] - test_metrics["test_rmse"]
        )

        for name, fig in _figures(model, test_df, pred, history).items():
            mlflow.log_figure(fig, f"plots/{name}")
            plt.close(fig)
        with tempfile.TemporaryDirectory() as tmp:
            sample = test_df[["series_id", "season", "episode", TARGET]].assign(prediction=pred)
            sample.sample(min(2000, len(sample)), random_state=cfg.project.seed).to_csv(
                Path(tmp) / "test_predictions_sample.csv", index=False
            )
            pd.Series(model.booster_.feature_importance("gain"), index=FEATURES).to_csv(
                Path(tmp) / "feature_importance.csv", header=["gain"]
            )
            if history is not None:
                history.to_csv(Path(tmp) / "optuna_trials.csv", index=False)
            mlflow.log_artifacts(tmp, "evaluation")

        champion_rmse = _champion_rmse(cfg, X_test, y_test)
        if champion_rmse is not None:
            mlflow.log_metric("champion_test_rmse", champion_rmse)

        model_info = mlflow.lightgbm.log_model(
            model,
            name="model",
            signature=infer_signature(X_test.head(100), pred[:100]),
            input_example=X_test.head(3),
            registered_model_name=cfg.training.registered_model_name if register else None,
        )
        if register:
            mv = str(model_info.registered_model_version)
            name = cfg.training.registered_model_name
            for key, value in {
                "rmse": f"{test_metrics['test_rmse']:.4f}",
                "baseline_rmse": f"{baseline['baseline_rmse']:.4f}",
                "data_sha": sha,
                "git_sha": version["git_sha"],
                "run_id": run.info.run_id,
                "features": ",".join(FEATURES),
            }.items():
                client.set_model_version_tag(name, mv, key, value)
            decision = registry.apply_decision(
                cfg,
                mv,
                run.info.run_id,
                test_metrics["test_rmse"],
                baseline["baseline_rmse"],
                champion_rmse,
                sha,
            )
        else:  # reproduction : on vérifie, on ne met rien en service
            mv = ""
            decision = registry.PromotionDecision(False, "Run de vérification, non enregistré")
        mlflow.set_tags(
            {
                "model.version": mv or "none",
                "decision": "promoted"
                if decision.promote
                else "not_registered"
                if not register
                else "rejected",
                "decision.reason": decision.reason,
            }
        )
        mlflow.log_metric("total_duration_s", time.time() - t0)

    result = TrainResult(
        run_id=run.info.run_id,
        model_version=mv,
        data_sha=sha,
        rmse=test_metrics["test_rmse"],
        baseline_rmse=baseline["baseline_rmse"],
        champion_rmse=champion_rmse,
        promoted=decision.promote,
        reason=decision.reason,
        best_params=best_params,
    )
    log.info("Run %s terminé : %s", run.info.run_id, json.dumps(asdict(result), default=str))
    return result


def reproduce(cfg: AppConfig, run_id: str) -> tuple[TrainResult, float]:
    """Rejoue un run : mêmes données (SHA), mêmes hyperparamètres, même seed.

    Retourne le nouveau résultat et l'écart de RMSE avec le run d'origine.
    """
    original = MlflowClient().get_run(run_id)
    tags, run_params = original.data.tags, original.data.params
    if tags.get("seed") and int(tags["seed"]) != cfg.project.seed:
        log.warning("Seed de config (%s) ≠ seed d'origine (%s)", cfg.project.seed, tags["seed"])
    params = {}
    for key, value in run_params.items():
        if key.startswith("hp_"):
            name = key[3:]
            params[name] = int(value) if name in INT_PARAMS else float(value)
    result = train(
        cfg,
        snapshot_sha=tags["data.sha"],
        params=params,
        run_name=f"reproduce-{run_id[:8]}",
        trigger="reproduce",
        extra_tags={"reproduces_run": run_id},
        register=False,
    )
    delta = abs(result.rmse - original.data.metrics["test_rmse"])
    return result, delta
