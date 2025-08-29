# SpectraVision Base Docker Image
# Contains common dependencies and SpectraVision codebase
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

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

# Install common Python dependencies (version-agnostic)
RUN pip install \
    torch==2.1.2 \
    torchvision==0.16.2 \
    numpy==1.26.4 \
    pillow \
    requests \
    tqdm \
    pandas \
    matplotlib \
    seaborn \
    opencv-python \
    accelerate \
    bitsandbytes \
    peft \
    validators

# Copy SpectraVision codebase
COPY spectravision/ /workspace/spectravision/
COPY configs/ /workspace/configs/
COPY scripts/ /workspace/scripts/

# Copy VLMEvalKit (mounted as volume in dev, copied to image for prod)
COPY VLMEvalKit/ /workspace/VLMEvalKit/

# Create output directories
RUN mkdir -p /workspace/outputs/{logs,reports,results,vlmevalkit_results}
RUN mkdir -p /workspace/data

# Set environment variables
ENV PYTHONPATH="/workspace:/workspace/VLMEvalKit:${PYTHONPATH}"
ENV CUDA_VISIBLE_DEVICES="0"

# Default command
CMD ["/bin/bash"]