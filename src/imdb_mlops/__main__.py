"""Point d'entrée en ligne de commande : `python -m imdb_mlops <commande>`."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from imdb_mlops.config import load_config


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    for noisy in ("httpx", "great_expectations", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911, PLR0912 (une branche par commande)
    parser = argparse.ArgumentParser(prog="imdb_mlops")
    parser.add_argument("--config", help="Chemin du fichier de configuration YAML")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="Crée les tables de la base applicative")

    p_data = sub.add_parser("data", help="Ingestion + contrôles qualité d'un snapshot")
    p_data.add_argument("--ref", help="Branche ou SHA du dépôt source (défaut : config)")
    p_data.add_argument(
        "--fault",
        help="Injecte un défaut pour démontrer le blocage (ratings_out_of_range, "
        "duplicates, missing_column, truncated, null_keys)",
    )

    p_train = sub.add_parser(
        "train", help="Entraîne sur un snapshot validé et applique la règle de promotion"
    )
    p_train.add_argument("--sha", help="Snapshot à utiliser (défaut : dernier validé)")
    p_train.add_argument("--trials", type=int, help="Surcharge du nombre de trials Optuna")

    p_repro = sub.add_parser("reproduce", help="Rejoue un run MLflow (données, params, seed)")
    p_repro.add_argument("run_id")

    sub.add_parser("models", help="Liste les versions du registre")

    p_rb = sub.add_parser("rollback", help="Remet en service une version précédente")
    p_rb.add_argument("--version", help="Version cible (défaut : previous_champion)")
    p_rb.add_argument("--reason", default="", help="Motif, tracé dans le registre")

    p_eval = sub.add_parser(
        "evaluate",
        help="Compare les prédictions passées aux notes "
        "publiées dans un snapshot plus récent (J+1)",
    )
    p_eval.add_argument("--sha", help="Snapshot de référence (défaut : dernier validé)")

    p_sim = sub.add_parser("simulate", help="Simule du trafic sur l'API (démonstration)")
    p_sim.add_argument("-n", type=int, default=300, help="Nombre de requêtes")
    p_sim.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost:8000"))
    p_sim.add_argument("--served-sha", help="Snapshot servi par l'API (défaut : dernier validé)")
    p_sim.add_argument("--target-sha", help="Snapshot J+1 : fournit les nouveaux épisodes")
    p_sim.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    cfg = load_config(args.config)

    if args.command in {"train", "reproduce", "rollback", "evaluate"}:
        from imdb_mlops import db

        try:
            db.init_db()  # idempotent : crée les tables manquantes (ex. model_events)
        except Exception as exc:
            logging.getLogger(__name__).warning("Base applicative indisponible : %s", exc)

    if args.command == "init-db":
        from imdb_mlops import db

        db.init_db()
        print("Tables créées.")
        return 0

    if args.command == "data":
        from imdb_mlops.data.pipeline import run_data_step
        from imdb_mlops.data.quality import DataQualityError

        try:
            _, manifest, report = run_data_step(cfg, ref=args.ref, fault=args.fault)
        except DataQualityError as exc:
            print(f"\nLOT REJETÉ — chaîne interrompue.\n{exc.report.summary()}", file=sys.stderr)
            return 2
        print(f"\nLot conforme : {manifest.n_rows} épisodes ({manifest.sha[:10]}).")
        print(report.summary())
        return 0

    if args.command == "train":
        from imdb_mlops.training.train import TrainingBlockedError, train

        if args.trials is not None:
            cfg.training.n_trials = args.trials
        try:
            r = train(cfg, snapshot_sha=args.sha)
        except TrainingBlockedError as exc:
            print(f"ENTRAÎNEMENT REFUSÉ : {exc}", file=sys.stderr)
            return 3
        print(f"\nRun {r.run_id} → modèle v{r.model_version}")
        print(
            f"RMSE test {r.rmse:.4f} | baseline {r.baseline_rmse:.4f} | champion "
            f"{'-' if r.champion_rmse is None else f'{r.champion_rmse:.4f}'}"
        )
        print(f"{'PROMU' if r.promoted else 'NON PROMU'} : {r.reason}")
        return 0

    if args.command == "reproduce":
        from imdb_mlops.training.train import reproduce

        r, delta = reproduce(cfg, args.run_id)
        print(f"\nReproduction de {args.run_id} → run {r.run_id}")
        print(f"RMSE {r.rmse:.6f} | écart avec l'original : {delta:.2e}")
        return 0 if delta < 1e-6 else 4

    if args.command == "models":
        from imdb_mlops.training.registry import list_versions

        rows = list_versions(cfg)
        if not rows:
            print("Registre vide.")
        for r in rows:
            print(
                f"v{r['version']:<3} {r['aliases']:<30} {r['status']:<11} rmse={r['rmse']:<7} "
                f"data={r['data_sha']} code={r['git_sha']} {r['created']}\n"
                f"     {r['reason']}"
            )
        return 0

    if args.command == "evaluate":
        from imdb_mlops.monitoring.outcomes import evaluate_predictions

        summary = evaluate_predictions(cfg, args.sha)
        for k, v in summary.items():
            print(f"{k:18} {v}")
        return 0

    if args.command == "simulate":
        from imdb_mlops.monitoring.simulate import simulate

        stats = simulate(args.api_url, args.n, args.served_sha, args.target_sha, seed=args.seed)
        for k, v in stats.items():
            print(f"{k:24} {v}")
        return 0

    if args.command == "rollback":
        from imdb_mlops.training.registry import rollback

        version = rollback(cfg, args.version, args.reason)
        print(f"Champion → v{version}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
