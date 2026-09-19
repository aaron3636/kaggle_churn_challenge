FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install dependencies first so they're cached separately from app code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src ./src
COPY models ./models
RUN uv sync --frozen --no-dev

# The processed feature table isn't shipped in the image (derived from Kaggle
# competition data that shouldn't be redistributed) — mount it at runtime, see README.
VOLUME ["/app/data/processed"]

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "src/churn/app.py", "--server.address=0.0.0.0"]
