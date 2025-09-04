# SpectraVision Container for transformers==4.37.2
# Supports stable models: InternVL2, LLaVA, ShareGPT4V, MMAlaya2, CogVLM, etc.
FROM spectravision-base:latest

# Copy version-specific VLMEvalKit (with MiniCPM-V and InternVL2 patches applied)
COPY VLMEvalKit-4.37/ /workspace/VLMEvalKit/

# Copy and install requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Install VLMEvalKit with transformers 4.37 compatibility
WORKDIR /workspace/VLMEvalKit
RUN pip install -e .
WORKDIR /workspace

# Install LLaVA (pinned to stable commit for reproducibility)
RUN pip install --no-deps git+https://github.com/haotian-liu/LLaVA.git@890a5d1bf46b31610d151779602e67f061c57fd0

# Install compatible xformers (without forcing torch upgrade)
RUN pip install xformers --no-deps || echo "⚠️ xformers installation failed, continuing without it"

# Ensure torch versions are correct (in case LLaVA installation changed them)
RUN pip install --force-reinstall \
    torch==2.1.2 \
    torchvision==0.16.2

# Fix NumPy compatibility issues (ensure 1.x after package installs)
RUN pip install --force-reinstall numpy==1.26.4

# Set container identifier
ENV CONTAINER_VERSION="transformers-4.37"
ENV CONTAINER_MODELS="InternVL2-2B,MiniCPM-V-2_6,llava_v1.5_7b,cogvlm-chat,InternVL2-8B,sharegpt4v_7b,MMAlaya2,llava_v1.5_13b"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

WORKDIR /workspace
CMD ["/bin/bash"]