# ==============================================================================
# RetailPulse Multi-Stage Production Dockerfile
# Stage 1: Build & Dependencies
# Stage 2: Minimal Distroless / Slim Non-Root Runtime Image
# ==============================================================================

# ----------------- Stage 1: Build Dependencies -----------------
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install prophet fastapi uvicorn httpx pyyaml optuna shap

# ----------------- Stage 2: Production Non-Root Runtime -----------------
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8501

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create dedicated non-root application user
RUN groupadd -g 1001 retailpulse && \
    useradd -u 1001 -g retailpulse -s /bin/bash -m retailpulse

# Copy application source code and models
COPY --chown=retailpulse:retailpulse . /app

# Switch to non-root user
USER retailpulse

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/_stcore/health || exit 1

EXPOSE 8501 8000 5000

# Default command launches Streamlit application
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
