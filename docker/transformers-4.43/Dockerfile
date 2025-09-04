# SpectraVision Container for transformers==4.43.0
# Supports mid-range models: Phi-3.5-Vision, updated InternVL2, etc.
FROM spectravision-base:latest

# Install transformers 4.43.0 and compatible dependencies
RUN pip install \
    transformers==4.43.0 \
    tokenizers==0.19.1 \
    timm==0.9.12 \
    sentencepiece==0.1.99 \
    protobuf==4.25.1

# Install essential model dependencies
RUN pip install \
    einops \
    tiktoken

# Phi-3.5-Vision specific dependencies
RUN pip install \
    flash-attn \
    || echo "flash-attn installation failed, continuing without it"

# Fix NumPy compatibility issues (ensure 1.x after package installs)
RUN pip install --force-reinstall numpy==1.26.4

# Set container identifier
ENV CONTAINER_VERSION="transformers-4.43"
ENV CONTAINER_MODELS="Phi-3.5-Vision,Moondream2"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

WORKDIR /workspace
CMD ["/bin/bash"]