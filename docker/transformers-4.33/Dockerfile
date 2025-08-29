# SpectraVision Container for transformers==4.33.0
# Supports legacy models: Qwen, VisualGLM, early mPLUG-Owl2, etc.
FROM spectravision-base:latest

# Install transformers 4.33.0 and compatible dependencies
RUN pip install \
    transformers==4.33.0 \
    tokenizers==0.13.3 \
    timm==0.6.13 \
    sentencepiece==0.1.99 \
    protobuf==3.20.3

# Install model-specific dependencies
RUN pip install \
    einops \
    xformers

# Qwen-VL specific dependencies
RUN pip install \
    tiktoken \
    charset-normalizer

# VisualGLM specific dependencies
RUN pip install \
    icetk

# Set container identifier
ENV CONTAINER_VERSION="transformers-4.33"
ENV CONTAINER_MODELS="qwen_chat,visualglm,monkey,instructblip_13b,pandagpt_13b"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

WORKDIR /workspace
CMD ["/bin/bash"]