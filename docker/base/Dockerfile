# SpectraVision Base Docker Image
# Contains common dependencies and SpectraVision codebase
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
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

# Set Python aliases
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Create workspace directory
WORKDIR /workspace

# Install common Python dependencies (version pinned for reproducibility)
RUN pip install \
    torch==2.1.2 \
    torchvision==0.16.2 \
    numpy==1.26.4 \
    pillow==10.0.1 \
    requests==2.31.0 \
    tqdm==4.66.1 \
    pandas==2.1.4 \
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    opencv-python==4.8.1.78 \
    accelerate==0.25.0 \
    bitsandbytes==0.41.3 \
    peft==0.7.1 \
    validators==0.22.0

# Copy SpectraVision codebase
COPY spectravision/ /workspace/spectravision/
COPY configs/ /workspace/configs/
COPY scripts/ /workspace/scripts/
COPY patches/ /workspace/patches/

# Note: VLMEvalKit will be copied by individual transformer version containers
# This base image focuses on common dependencies only

# Install VLMEvalKit required dependencies (avoiding duplicates with common deps)
RUN pip install \
    python-dotenv==1.0.0 \
    rich==13.7.0 \
    tabulate==0.9.0 \
    PyYAML==6.0.1 \
    scikit-learn==1.2.2 \
    scipy==1.11.4 \
    datasets==2.15.0 \
    huggingface-hub==0.19.4 \
    safetensors==0.4.1 \
    timm==0.9.12 \
    omegaconf==2.3.0 \
    nltk==3.8.1 \
    imageio==2.33.1 \
    openpyxl==3.1.2 \
    portalocker==2.8.2 \
    timeout-decorator==0.5.0 \
    xlsxwriter==3.1.9 \
    openai==1.6.1 \
    json-repair==0.7.0 \
    ipdb==0.13.13 \
    sty==1.0.6 \
    qwen-vl-utils==0.0.3

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