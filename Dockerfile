# Image applicative unique : pipeline (ingestion, qualité, entraînement), worker Prefect,
# API et tests. Les dépendances viennent de uv.lock : environnement identique partout.
FROM python:3.12-slim

# libgomp1 : runtime OpenMP requis par LightGBM (entraînement multi-thread).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    GX_ANALYTICS_ENABLED=False

WORKDIR /app

# Couche dépendances (mise en cache tant que uv.lock ne change pas).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project

COPY src ./src
COPY configs ./configs
COPY tests ./tests
RUN uv sync --locked

# Version du code embarquée dans l'image, remontée dans MLflow et l'API.
ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA

CMD ["python", "-m", "imdb_mlops", "--help"]
