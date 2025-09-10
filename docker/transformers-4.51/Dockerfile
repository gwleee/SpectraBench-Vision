# SpectraVision Container for transformers==4.51.0
# Supports cutting-edge models: Latest updates and experimental features
FROM spectravision-base:latest

# Copy version-specific VLMEvalKit (with Phi-4-Vision and Llama-4-Scout patches applied)
COPY VLMEvalKit-4.51/ /workspace/VLMEvalKit/

# Install VLMEvalKit with transformers 4.51 compatibility
WORKDIR /workspace/VLMEvalKit
RUN pip install -e . --no-deps
WORKDIR /workspace

# Copy and install requirements (AFTER VLMEvalKit to override versions)
COPY docker/transformers-4.51/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --force-reinstall

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