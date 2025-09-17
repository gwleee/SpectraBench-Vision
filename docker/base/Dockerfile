# SpectraVision Base Docker Image
# Contains common dependencies and SpectraVision codebase
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies first including CUDA development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    git \
    wget \
    curl \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Add NVIDIA CUDA repository and install CUDA development tools for flash-attn compilation
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    cuda-nvcc-12-1 \
    cuda-cudart-dev-12-1 \
    cuda-nvml-dev-12-1 \
    libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/* \
    && rm cuda-keyring_1.0-1_all.deb

# Set CUDA environment variables for compilation
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install Python 3.9 from deadsnakes PPA with proper error handling
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3.9-distutils \
    && rm -rf /var/lib/apt/lists/*

# Set Python aliases to make python3.9 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Install pip for Python 3.9 using ensurepip (more stable)
RUN python3.9 -m ensurepip --upgrade \
    || (curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.9 get-pip.py --no-cache-dir \
    && rm get-pip.py)

# Upgrade pip with explicit version constraints
RUN python3.9 -m pip install --no-cache-dir --upgrade \
    "pip>=23.0" \
    "setuptools>=65.0" \
    "wheel>=0.38"

# Create workspace directory
WORKDIR /workspace

# Base image only contains system-level dependencies
# Python packages will be installed by individual transformer containers

# Copy SpectraVision codebase
COPY spectravision/ /workspace/spectravision/
COPY configs/ /workspace/configs/
COPY scripts/ /workspace/scripts/
COPY patches/ /workspace/patches/

# Note: VLMEvalKit will be copied by individual transformer version containers
# This base image focuses on common dependencies only

# VLMEvalKit and all Python dependencies will be installed by transformer containers

# VLMEvalKit installation will be handled by individual transformer version containers

# Create output directories
WORKDIR /workspace
RUN mkdir -p /workspace/outputs/{logs,reports,results,vlmevalkit_results}
RUN mkdir -p /workspace/data

# Set environment variables
ENV PYTHONPATH="/workspace:/workspace/VLMEvalKit:${PYTHONPATH}"
ENV CUDA_VISIBLE_DEVICES="0"

# Default command
CMD ["/bin/bash"]