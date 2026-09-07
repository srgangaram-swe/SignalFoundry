# AlphaForge — research image with the C++ execution core built in.
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md LICENSE CHANGELOG.md ./
COPY alphaforge ./alphaforge
COPY cpp ./cpp
COPY scripts ./scripts
COPY apps ./apps
COPY configs ./configs

RUN pip install --no-cache-dir "uv==0.11.28" \
    && uv sync --locked --extra dev \
    && uv run python scripts/build_native.py

# Default: run the no-network synthetic demo pipeline end to end.
CMD ["sh", "-c", "python scripts/run_walk_forward.py --synthetic && python scripts/run_backtest.py --latest && python scripts/generate_report.py"]
