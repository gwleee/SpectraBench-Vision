# SpectraVision Container for transformers==4.49.0
# Supports latest models: SmolVLM, Qwen2.5-VL, Pixtral, etc.
FROM spectravision-base:latest

# Install latest transformers and compatible dependencies
RUN pip install \
    transformers==4.49.0 \
    tokenizers==0.21.0 \
    timm==1.0.9 \
    sentencepiece==0.2.0 \
    protobuf==5.28.2

# Install essential dependencies for latest models
RUN pip install \
    einops \
    tiktoken

# Additional utilities (no compilation needed)
RUN pip install \
    diffusers \
    optimum

# Fix NumPy compatibility issues (ensure 1.x compatibility)
RUN pip install --force-reinstall numpy==1.26.4

# Set container identifier
ENV CONTAINER_VERSION="transformers-4.49"
ENV CONTAINER_MODELS="SmolVLM-256M,SmolVLM-500M,SmolVLM,Qwen2.5-VL-3B-Instruct,Aria,Qwen2.5-VL-7B-Instruct,Pixtral-12B,Qwen2.5-VL-32B-Instruct,Qwen2.5-VL-72B-Instruct"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

WORKDIR /workspace
CMD ["/bin/bash"]