#!/bin/sh
# Utilisateur en lecture seule pour Grafana sur la base applicative (idempotent).
# Exécuté à l'initialisation du volume ; sur un volume existant :
#   docker compose exec postgres sh /docker-entrypoint-initdb.d/20-grafana-readonly.sh
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname app <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'grafana_ro') THEN
    CREATE ROLE grafana_ro LOGIN PASSWORD '${GRAFANA_DB_PASSWORD}';
  ELSE
    ALTER ROLE grafana_ro WITH LOGIN PASSWORD '${GRAFANA_DB_PASSWORD}';
  END IF;
END
\$\$;
GRANT CONNECT ON DATABASE app TO grafana_ro;
GRANT USAGE ON SCHEMA public TO grafana_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana_ro;
ALTER DEFAULT PRIVILEGES FOR ROLE ${POSTGRES_USER} IN SCHEMA public
  GRANT SELECT ON TABLES TO grafana_ro;
SQL
