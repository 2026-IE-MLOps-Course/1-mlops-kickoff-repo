FROM continuumio/miniconda3:latest

WORKDIR /app

# Force Python to stream logs instantly (no buffering)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Copy the lockfile first (Docker caches this layer)
COPY conda-lock.yml .

# Install conda-lock, create env from lockfile, clean up
RUN conda install -c conda-forge conda-lock -y && \
    conda-lock install -n mlops conda-lock.yml && \
    apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    conda clean -afy

# Use the lockfile environment's Python
ENV PATH=/opt/conda/envs/mlops/bin:$PATH

# Copy application code (respects .dockerignore)
COPY . .

# Render assigns PORT dynamically; default to 8000 locally
EXPOSE 8000

# Health check for Render
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8000}/health || exit 1

# Start the API server
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
