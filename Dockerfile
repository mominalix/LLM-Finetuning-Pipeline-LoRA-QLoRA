# Multi-stage Dockerfile for LLM Fine-tuning Pipeline
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash llmuser && \
    echo "llmuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip3 install -e .

# Create necessary directories
RUN mkdir -p /app/data /app/cache /app/logs /app/models /app/outputs && \
    chown -R llmuser:llmuser /app

# Switch to non-root user
USER llmuser

# Expose ports for MLflow, Streamlit, and Flask
EXPOSE 5000 8501 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/health || exit 1

# Default command
CMD ["python3", "-m", "src.gui.app"]

# Development stage
FROM base as development

USER root

# Install development dependencies
RUN pip3 install jupyter jupyterlab ipywidgets

# Install additional development tools
RUN apt-get update && apt-get install -y \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

USER llmuser

# Production stage
FROM base as production

# Remove unnecessary packages and clean up
USER root
RUN apt-get autoremove -y && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER llmuser

# Set production environment
ENV ENVIRONMENT=production
ENV PYTHONOPTIMIZE=1

# GPU-specific stage
FROM base as gpu

# Install additional GPU libraries
USER root
RUN pip3 install \
    flash-attn \
    triton \
    xformers

USER llmuser

# CPU-only stage for lighter deployments
FROM python:3.10-slim as cpu

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install requirements (CPU-specific)
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

COPY . .
RUN pip install -e .

RUN useradd -m -s /bin/bash llmuser && \
    mkdir -p /app/data /app/cache /app/logs /app/models /app/outputs && \
    chown -R llmuser:llmuser /app

USER llmuser

EXPOSE 5000 8501 8080

CMD ["python3", "-m", "src.gui.app"]