# Multi-stage Dockerfile for SentinelTrack
# Base image: Python 3.10 slim
FROM python:3.10-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies for OpenCV and PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and configuration
COPY src/ ./src/
COPY migrations/ ./migrations/
COPY scripts/ ./scripts/
COPY config.example.json ./config.json
COPY alembic.ini .
COPY pyproject.toml .

# Create required directories
RUN mkdir -p data models logs/entries logs/exits docs

# Default entrypoint
CMD ["python", "-m", "src.main"]
