# Multi-stage Dockerfile for RAG pipeline

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Download models (if needed)
# Note: Models will be downloaded at runtime, but you can pre-download here
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Stage 2: Runtime
FROM python:3.11-slim as runtime

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 raguser && \
    chown -R raguser:raguser /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/raguser/.local

# Copy application code
COPY --chown=raguser:raguser . .

# Set PATH for user-installed packages
ENV PATH=/home/raguser/.local/bin:$PATH
ENV PYTHONPATH=/app

# Switch to non-root user
USER raguser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/v1/health', timeout=5)" || exit 1

# Expose API port
EXPOSE 8000

# Default command (can be overridden)
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
