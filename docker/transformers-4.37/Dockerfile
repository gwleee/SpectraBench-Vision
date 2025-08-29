# SpectraVision Container for transformers==4.37.2
# Supports current SpectraVision models: InternVL2, LLaVA, CogVLM, etc.
FROM spectravision-base:latest

# Install transformers 4.37.2 and compatible dependencies (current setup)
RUN pip install \
    transformers==4.37.2 \
    tokenizers==0.15.1 \
    timm==0.6.13 \
    sentencepiece==0.1.99 \
    protobuf==3.20.3

# Install LLaVA required dependencies first
RUN pip install \
    einops==0.6.1 \
    einops-exts==0.0.4 \
    markdown2[all] \
    shortuuid \
    accelerate==0.21.0 \
    gradio==4.16.0 \
    gradio-client==0.8.1 \
    httpx==0.24.0

# Install LLaVA
RUN pip install --no-deps git+https://github.com/haotian-liu/LLaVA.git

# Install compatible xformers (without forcing torch upgrade)
RUN pip install xformers --no-deps

# Ensure torch versions are correct
RUN pip install --force-reinstall \
    torch==2.1.2 \
    torchvision==0.16.2

# Fix NumPy compatibility issues (downgrade to 1.x after torch install)
RUN pip install --force-reinstall numpy==1.26.4

# CogVLM dependencies
RUN pip install \
    icetk

# mPLUG-Owl2 dependencies
RUN pip install \
    decord \
    opencv-python

# Set container identifier
ENV CONTAINER_VERSION="transformers-4.37"
ENV CONTAINER_MODELS="InternVL2-2B,llava_v1.5_7b,cogvlm-chat,XComposer2,mPLUG-Owl2"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

WORKDIR /workspace
CMD ["/bin/bash"]