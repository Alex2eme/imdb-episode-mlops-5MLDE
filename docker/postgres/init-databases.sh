#!/bin/sh
# Crée une base par service : séparation des responsabilités sur une seule instance.
set -e
for db in mlflow prefect app; do
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
    -c "CREATE DATABASE $db OWNER $POSTGRES_USER;"
done
