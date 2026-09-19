# ── FaceMatch AI — Multi-stage Docker Build ──────────────────
# Stage 1: Build dependencies
# Stage 2: Slim production image
# ──────────────────────────────────────────────────────────────

FROM python:3.11-slim AS builder

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies for dlib, OpenCV, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Production Image ─────────────────────────────────────────
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application code
COPY app/ ./app/
COPY static/ ./static/
COPY config.yaml .
COPY run.py .

# Create data directories
RUN mkdir -p data/people data/index data/db data/uploads

# Expose the default port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/admin/health')" || exit 1

# Run the server
CMD ["python", "run.py"]
