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

# Install model-specific dependencies
RUN pip install \
    einops \
    xformers

# LLaVA dependencies
RUN pip install --no-deps git+https://github.com/haotian-liu/LLaVA.git

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