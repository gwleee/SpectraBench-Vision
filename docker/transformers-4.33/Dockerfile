# SpectraVision Container for transformers==4.33.0
# Supports legacy models: Qwen, VisualGLM, early mPLUG-Owl2, etc.
FROM spectravision-base:latest

# Copy version-specific VLMEvalKit (with appropriate patches already applied)
COPY VLMEvalKit-4.33/ /workspace/VLMEvalKit/

# Install VLMEvalKit with transformers 4.33 compatibility
WORKDIR /workspace/VLMEvalKit
RUN pip install -e . --no-deps
WORKDIR /workspace

# Copy and install requirements (AFTER VLMEvalKit to override versions)
COPY docker/transformers-4.33/requirements.txt /tmp/requirements.txt  
RUN pip install -r /tmp/requirements.txt --force-reinstall

# Install xformers separately (may fail on some GPU architectures)
RUN pip install xformers==0.0.22 || echo "⚠️ xformers installation failed, continuing without it"

# Fix NumPy compatibility issues (ensure 1.x after package installs)
RUN pip install --force-reinstall numpy==1.26.4

# Set container identifier
ENV CONTAINER_VERSION="transformers-4.33"
ENV CONTAINER_MODELS="VisualGLM_6b,qwen_chat,mPLUG-Owl2,monkey,XComposer2,idefics_9b_instruct,instructblip_13b,PandaGPT_13B"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

WORKDIR /workspace
CMD ["/bin/bash"]