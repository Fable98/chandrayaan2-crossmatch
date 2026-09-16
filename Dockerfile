# SIH PS 26166 Backend Dockerfile
# Python 3.14 to match requirements.txt's 2026-era pins (no 3.11 wheels exist
# for e.g. tifffile==2026.8.23) and the CI runner. Do not downgrade without
# re-pinning every dependency.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    GLOBAL_SEED=42 \
    PYTHONPATH=/app/backend:/app

# Install system dependencies required by OpenCV, GDAL/Rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libgdal-dev \
    gdal-bin \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend and ML engine modules.
# data_preprocessing_pipeline/processed_triplets is the backend's primary
# triplet source (see backend/data/loader.py PROCESSED_TRIPLETS_DIR); without
# it the container boots with zero triplets. Triplet tiles are PNG/JSON so
# the *.tif/*.pt .dockerignore rules do not strip them.
COPY ML_model/ ML_model/
COPY backend/ backend/
COPY utils/ utils/
COPY data/ data/
COPY data_preprocessing_pipeline/processed_triplets/ data_preprocessing_pipeline/processed_triplets/
COPY run_demo.py .

# Non-root runtime (Phase 0.1/Phase 6): the writable trees (dynamic run
# outputs, reports, uploads staging) are created here and chown'd so the
# named-volume mounts in docker-compose.yml stay writable as appuser.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data_preprocessing_pipeline/dynamic_runs /app/reports /app/backend/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=20s \
  CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
