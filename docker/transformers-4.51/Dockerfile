# SpectraVision Container for transformers==4.51.0
# Supports cutting-edge models: Latest updates and experimental features
FROM spectravision-base:latest

# Install latest transformers and compatible dependencies
RUN pip install \
    transformers==4.51.0 \
    tokenizers==0.21.4 \
    timm==1.0.11 \
    sentencepiece==0.2.0 \
    protobuf==5.28.3

# Install essential dependencies for latest models
RUN pip install \
    einops \
    tiktoken \
    accelerate

# Additional utilities for cutting-edge models
RUN pip install \
    diffusers \
    optimum

# Fix NumPy compatibility issues (ensure 1.x compatibility)
RUN pip install --force-reinstall numpy==1.26.4

# Set container identifier
ENV CONTAINER_VERSION="transformers-4.51"
ENV CONTAINER_MODELS="Phi-4-Vision,Llama-4-Scout-17B-16E-Instruct"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

WORKDIR /workspace
CMD ["/bin/bash"]