# SpectraVision Container for transformers==4.49.0
# Supports latest models: SmolVLM, Qwen2.5-VL, Pixtral, etc.
FROM spectravision-base:latest

# Copy base VLMEvalKit and patches for runtime setup
COPY VLMEvalKit/ /workspace/VLMEvalKit-base/
COPY patches/ /workspace/patches/
COPY scripts/apply_patches.sh /workspace/

# Setup version-specific VLMEvalKit with patches
RUN cp -r /workspace/VLMEvalKit-base /workspace/VLMEvalKit && \
    cd /workspace/VLMEvalKit && \
    PATCHES_DIR=/workspace/patches /workspace/apply_patches.sh && \
    rm -rf /workspace/VLMEvalKit-base /workspace/patches /workspace/apply_patches.sh

# Copy and install requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Install VLMEvalKit with transformers 4.49 compatibility
WORKDIR /workspace/VLMEvalKit
RUN pip install -e .
WORKDIR /workspace

# Fix NumPy compatibility issues (ensure 1.x compatibility)
RUN pip install --force-reinstall numpy==1.26.4

# Set container identifier
ENV CONTAINER_VERSION="transformers-4.49"
ENV CONTAINER_MODELS="SmolVLM-256M,SmolVLM-500M,SmolVLM,Qwen2-VL-2B-Instruct,Qwen2.5-VL-3B-Instruct,Aria,Qwen2.5-VL-7B-Instruct,Pixtral-12B,Qwen2.5-VL-32B-Instruct,Qwen2.5-VL-72B-Instruct"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

WORKDIR /workspace
CMD ["/bin/bash"]