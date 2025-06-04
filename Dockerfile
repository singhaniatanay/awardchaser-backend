# syntax=docker/dockerfile:1

# Stage 1: install dependencies
FROM python:3.11-slim AS builder

# Install Poetry
ENV POETRY_VERSION=1.8.2 \
    POETRY_HOME=/opt/poetry
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s $POETRY_HOME/bin/poetry /usr/local/bin/poetry && \
    apt-get purge -y curl && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app

# Copy lock file and pyproject to install deps
COPY pyproject.toml ./
COPY pyproject.lock* ./

RUN poetry install --only main --no-root --no-interaction --no-ansi

# Stage 2: copy application
FROM python:3.11-slim

# Copy installed deps from builder
COPY --from=builder /opt/app /opt/app

WORKDIR /opt/app

# Copy source code
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

