# SpectraVision Container for transformers==4.49.0
# Supports latest models: SmolVLM, Qwen2.5-VL, Pixtral, etc.
FROM spectravision-base:latest

# Copy version-specific VLMEvalKit (already patched)
COPY VLMEvalKit-4.49/ /workspace/VLMEvalKit/

# Install VLMEvalKit with transformers 4.49 compatibility
WORKDIR /workspace/VLMEvalKit
RUN pip install -e . --no-deps
WORKDIR /workspace

# Copy and install requirements (AFTER VLMEvalKit to override versions)
COPY docker/transformers-4.49/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --force-reinstall

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