FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY . /app

RUN groupadd --gid 1000 nexus \
    && useradd --uid 1000 --gid nexus --shell /bin/bash --create-home nexus \
    && mkdir -p /app/data/raw /app/data/processed /app/data/cache /app/logs \
    && chown -R nexus:nexus /app

USER nexus

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["python", "main.py"]
