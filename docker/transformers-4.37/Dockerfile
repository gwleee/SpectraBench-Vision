# Dockerfile for Transformers 4.37 - Stable VLM Models
# Models: LLaVA, InternVL2, CogVLM, ShareGPT4V, etc.
# Implements 3-stage dependency system: 잠금(Lock) → 검증(Verification) → 프로모션(Promotion)
FROM ghcr.io/gwleee/spectravision:base

LABEL version="4.37"
LABEL transformer_version="4.37.2"
LABEL models="InternVL2-2B,MiniCPM-V-2_6,LLaVA-1.5-7B,CogVLM-7B,InternVL2-8B,ShareGPT4V-7B,LLaVA-1.5-13B"
LABEL model_count="7"

# Switch to root for package installation (base image uses non-root user)
USER root

# Copy transformer-specific requirements
COPY docker/requirements/transformers-4.37.in /workspace/requirements/transformers-4.37.in
COPY docker/requirements/transformers-4.37.lock /workspace/requirements/transformers-4.37.lock

# Stage 1: 잠금(Lock) - Install locked transformer 4.37 dependencies with hash verification
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps --require-hashes \
    -r /workspace/requirements/transformers-4.37.lock

# DISABLED: Flash Attention causes ABI compatibility issues with PyTorch 2.1.2
# LLaVA works fine without flash-attn, just slightly slower
# If needed in future, must compile from source with exact PyTorch ABI match:
#   git clone https://github.com/Dao-AILab/flash-attention && cd flash-attention &&
#   MAX_JOBS=4 python setup.py install
# RUN --mount=type=cache,target=/root/.cache/pip \
#     python -m pip install --no-cache-dir "flash-attn==2.8.2" --no-build-isolation

# Install LLaVA package for LLaVA-based models (llava_v1.5_7b, llava_v1.5_13b, sharegpt4v_7b)
# CRITICAL: Must use editable install (-e) for VLMEvalKit to recognize LLaVA models
# Direct git+ installation does not work properly with VLMEvalKit
# VERSION PINNING: Using commit from 2024-01 for stability with transformers 4.37.2
COPY docker/scripts/patch_llava_builder.py /tmp/patch_llava_builder.py
RUN git clone https://github.com/haotian-liu/LLaVA.git /workspace/LLaVA && \
    cd /workspace/LLaVA && \
    git checkout v1.2.2.post1 && \
    python -m pip install --no-cache-dir -e . && \
    python3 /tmp/patch_llava_builder.py && \
    rm /tmp/patch_llava_builder.py

# Install VLMEvalKit with 4.37 compatibility
# NOTE: Install VLMEvalKit BEFORE verification to let it install dependencies
# Then pin transformers back to exact version
RUN cd /workspace/VLMEvalKit && \
    python -m pip install --no-cache-dir -e . && \
    python -m pip install --no-cache-dir transformers==4.37.2

# Stage 2: 검증(Verification) - Build-time smoke test for transformer 4.37
# IMPROVED: No network downloads, only import/version verification
# Copy verification script
COPY docker/scripts/verify_transformers.py /tmp/verify_transformers.py

RUN TRANSFORMERS_VERSION=4.37.2 python /tmp/verify_transformers.py && rm /tmp/verify_transformers.py

# Stage 3: 프로모션(Promotion) - Image ready for promotion pipeline
# This image has passed Lock + Verification stages
# Promotion pipeline tags:
#   - Build → ghcr.io/gwleee/spectravision:4.37-candidate
#   - After smoke tests pass → ghcr.io/gwleee/spectravision:4.37-stable
#   - After mini-bench pass → ghcr.io/gwleee/spectravision:4.37 (production)

# Configure environment for 4.37 models
# CACHE CONSISTENCY: All HF-related caches point to the same directory
ENV TRANSFORMERS_VERSION=4.37.2
ENV MODEL_CACHE_DIR=/workspace/.cache/huggingface
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface
ENV HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch

# Create cache directories
RUN mkdir -p $MODEL_CACHE_DIR $TORCH_HOME

# Copy model-specific scripts and entrypoint
COPY scripts/ /workspace/scripts/
COPY docker/scripts/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/scripts/*.sh 2>/dev/null || true && \
    chmod +x /workspace/entrypoint.sh

# Create outputs directory in VLMEvalKit and set permissions for ALL users
# This allows containers running with --user flag to write to /workspace
RUN mkdir -p /workspace/VLMEvalKit/outputs && \
    chmod -R 777 /workspace/VLMEvalKit && \
    chmod -R 777 /workspace/.cache || true

# Switch back to non-root user (but --user flag will override this)
USER spectravision

# Health check with transformer version and LLaVA import verification
HEALTHCHECK --interval=30s --timeout=15s --start-period=10s --retries=3 \
    CMD python -c "import transformers as tr; assert tr.__version__ == '4.37.2'; import torch; import llava; print(f'Transformers 4.37 + LLaVA healthy')" || exit 1

# NOTE: No ENTRYPOINT defined to maintain compatibility with test scripts
# The entrypoint.sh script is available at /workspace/entrypoint.sh and can be used manually:
# docker run --gpus all IMAGE /workspace/entrypoint.sh --data BENCHMARK --model MODEL

# Default command shows shell for interactive use
CMD ["/bin/bash"]
