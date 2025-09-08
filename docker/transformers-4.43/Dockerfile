# SpectraVision Container for transformers==4.43.0
# Supports mid-range models: Phi-3.5-Vision, updated InternVL2, etc.
FROM spectravision-base:latest

# Copy version-specific VLMEvalKit (with Moondream2 and Phi-3.5-Vision patches applied)
COPY VLMEvalKit-4.43/ /workspace/VLMEvalKit/

# Copy and install requirements
COPY docker/transformers-4.43/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Install VLMEvalKit with transformers 4.43 compatibility
WORKDIR /workspace/VLMEvalKit
RUN pip install -e .
WORKDIR /workspace

# Phi-3.5-Vision specific dependencies (optional)
RUN pip install flash-attn==2.5.0 || echo "⚠️ flash-attn installation failed, continuing without it"

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