# SpectraBench-Vision Integrated System
# Single image containing DockerOrchestrator with Docker-in-Docker capability
# Users can download once and automatically manage all transformer version containers

FROM ubuntu:22.04

# Avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Docker
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    && curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update \
    && apt-get install -y docker-ce docker-ce-cli containerd.io

# Install Python and system dependencies
RUN apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    bash \
    wget \
    build-essential

# Set up Python environment
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy SpectraVision source code
COPY . /workspace
WORKDIR /workspace

# Note: Integrated system uses individual transformer version containers
# Each container has its own version-specific VLMEvalKit with appropriate patches
# This orchestrator container only manages Docker-in-Docker functionality

# Install Python dependencies for orchestrator
RUN pip3 install \
    pyyaml \
    docker \
    requests \
    psutil \
    nvidia-ml-py \
    tqdm

# Copy Docker Compose files for container management  
COPY docker/docker-compose.prod.yml /workspace/docker-compose.prod.yml
COPY configs/ /workspace/configs/

# Copy startup scripts
COPY docker/integrated/start_spectrabench.sh /workspace/start_spectrabench.sh
COPY docker/integrated/docker_main.py /workspace/scripts/docker_main.py

RUN chmod +x /workspace/start_spectrabench.sh && \
    chmod +x /workspace/scripts/docker_main.py

# Set environment variables
ENV PYTHONPATH=/workspace
ENV DOCKER_HOST=unix:///var/run/docker.sock
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Expose Docker daemon port (optional, for remote access)
EXPOSE 2376

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD docker info > /dev/null 2>&1 || exit 1

# Default startup
ENTRYPOINT ["/workspace/start_spectrabench.sh"]
CMD []