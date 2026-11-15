FROM python:3.12-slim

# Version alignée sur uv.lock pour garantir la compatibilité client/serveur.
RUN pip install --no-cache-dir mlflow==3.18.0 "psycopg[binary]==3.3.9"

EXPOSE 5000
