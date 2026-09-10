# SIH PS 26166 Backend Dockerfile
# Python 3.14 to match requirements.txt's 2026-era pins (no 3.11 wheels exist
# for e.g. tifffile==2026.8.23) and the CI runner. Do not downgrade without
# re-pinning every dependency.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    GLOBAL_SEED=42

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

# Copy backend and ML engine modules
COPY ML_model/ ML_model/
COPY backend/ backend/
COPY utils/ utils/
COPY data/ data/
COPY run_demo.py .

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
