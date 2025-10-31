# Base Dockerfile for SpectraBench-Vision (Improved Version)
# Implements 3-stage dependency system: 잠금(Lock) → 검증(Verification) → 프로모션(Promotion)
# Improvements: Reproducibility, Security, Performance, Runtime Quality

# ============================================================================
# Build Arguments
# ============================================================================
ARG VLMEVALKIT_COMMIT=main
ARG PYTHON_VERSION=3.10

# ============================================================================
# Stage 1: Builder (Multi-stage for smaller final image)
# ============================================================================
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

LABEL maintainer="KISTI Large-scale AI Research Center"
LABEL description="SpectraBench-Vision Base Image - Builder Stage"

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Locale settings (improvement: prevent locale issues)
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install build dependencies (improvement: --no-install-recommends)
# Note: Removed apt cache mounts to avoid lock conflicts
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3=${PYTHON_VERSION}* \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    git-lfs \
    wget \
    curl \
    unzip \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libgl1-mesa-dev \
    libglib2.0-0 \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Initialize git-lfs (improvement: for large model downloads)
RUN git lfs install

# Create workspace
WORKDIR /workspace

# Create virtual environment for isolation (improvement: venv strategy)
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Install pip-tools for lock file management
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir pip-tools==7.3.0

# Copy requirements files
COPY docker/requirements/base-requirements.in /workspace/requirements/base-requirements.in
COPY docker/requirements/base-requirements.lock /workspace/requirements/base-requirements.lock

# Stage 1: 잠금(Lock) - Use locked dependencies with hash verification
# Improvement: Use BuildKit cache mount for faster rebuilds
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps --require-hashes \
    -r /workspace/requirements/base-requirements.lock

# Copy verification script
COPY docker/scripts/verify_base.py /tmp/verify_base.py

# Stage 2: 검증(Verification) - Build-time smoke test for base dependencies
RUN python /tmp/verify_base.py && rm /tmp/verify_base.py

# Copy VLMEvalKit patches (applied at build time)
COPY patches/ /workspace/patches/

# Install VLMEvalKit base
# Improvement: Pin commit SHA for reproducibility (default: main = latest)
ARG VLMEVALKIT_COMMIT=main
RUN git clone https://github.com/open-compass/VLMEvalKit.git /workspace/VLMEvalKit && \
    cd /workspace/VLMEvalKit && \
    if [ "${VLMEVALKIT_COMMIT}" != "main" ]; then \
        git checkout ${VLMEVALKIT_COMMIT}; \
    fi

# Apply patches for HuggingFace token support
RUN cd /workspace/VLMEvalKit && \
    for patch in /workspace/patches/*.patch; do \
        if [ -f "$patch" ]; then \
            git apply "$patch" || echo "Warning: Patch $patch failed to apply"; \
        fi \
    done

# Install VLMEvalKit
# Note: Keep -e (editable) for development flexibility
# For production, consider: pip install --no-cache-dir . (non-editable)
RUN cd /workspace/VLMEvalKit && \
    python -m pip install --no-cache-dir -e .

# Track VLMEvalKit commit for reproducibility
RUN cd /workspace/VLMEvalKit && \
    git rev-parse HEAD > /workspace/VLMEVALKIT_COMMIT.txt

# ============================================================================
# Stage 2: Runtime (Devel for DeepSpeed/Flash-Attention compilation support)
# ============================================================================
# NOTE: Using devel instead of runtime to support CUDA compilation in transformer images
# DeepSpeed and other packages require nvcc for compilation
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS runtime

# VLMEvalKit commit tracking (passed from build args)
ARG VLMEVALKIT_COMMIT=main

LABEL maintainer="KISTI Large-scale AI Research Center"
LABEL description="SpectraBench-Vision Base Image - Runtime Stage"
LABEL org.opencontainers.image.source="https://github.com/gwleee/SpectraBench-Vision"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.vlmevalkit.revision="${VLMEVALKIT_COMMIT}"

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Locale settings
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install runtime + build dependencies (devel image includes build tools for DeepSpeed/Flash-Attention)
# Note: Removed apt cache mounts to avoid lock conflicts
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    git-lfs \
    ca-certificates \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libjpeg-turbo8 \
    libpng16-16 \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Create non-root user (improvement: security isolation)
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd -g ${USER_GID} spectravision && \
    useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash spectravision && \
    mkdir -p /workspace /workspace/outputs /workspace/data && \
    chown -R spectravision:spectravision /workspace

# Copy Python virtual environment from builder (improvement: venv isolation)
COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Copy VLMEvalKit
COPY --from=builder --chown=spectravision:spectravision /workspace/VLMEvalKit /workspace/VLMEvalKit

# Note: Patches not copied to runtime - only needed at build time

# Set working directory
WORKDIR /workspace

# Switch to non-root user (improvement: run as non-root by default)
# Note: Can be overridden at runtime with --user if root access needed
USER spectravision

# Stage 3: 프로모션(Promotion) - Ready for transformer-specific builds
# This base image will be promoted to transformer-specific versions

# Health check (improvement: minimal imports for critical dependencies)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests, PIL, numpy; import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python", "-c", "print('SpectraBench-Vision Base Image Ready')"]
