"""Point d'entrée en ligne de commande : `python -m imdb_mlops <commande>`."""

from __future__ import annotations

import argparse
import logging
import sys

from imdb_mlops.config import load_config


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    for noisy in ("httpx", "great_expectations", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
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

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    cfg = load_config(args.config)

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
    return 1


if __name__ == "__main__":
    sys.exit(main())
