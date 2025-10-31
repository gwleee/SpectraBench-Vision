# Dockerfile for Transformers 4.33 - Legacy VLM Models
# Models: Qwen-VL, InstructBLIP, Monkey, etc.
# Implements 3-stage dependency system: 잠금(Lock) → 검증(Verification) → 프로모션(Promotion)
#
# IMPORTANT: This version uses an older VLMEvalKit commit (7b6f488) to avoid
# Thyme model which requires transformers >= 4.36

FROM ghcr.io/gwleee/spectravision:base

LABEL version="4.33"
LABEL transformer_version="4.33.0"
LABEL models="InstructBLIP-7B,Qwen-VL-Chat,mPLUG-Owl2-7B,Monkey-7B,InternLM-XComposer2-7B,IDEFICS-9B,InstructBLIP-13B,PandaGPT-13B"
LABEL model_count="8"

# Switch to root for package installation
USER root

# IMPORTANT: Replace VLMEvalKit with transformers 4.33-compatible version
# Remove the VLMEvalKit from base image (which has latest version with Thyme)
# and install older version (commit 7b6f488, before Thyme was added)
RUN cd /workspace && \
    rm -rf VLMEvalKit && \
    git clone https://github.com/open-compass/VLMEvalKit.git && \
    cd VLMEvalKit && \
    git checkout 7b6f488

# Apply sample limit patch to VLMEvalKit
COPY patches/*.patch /workspace/patches/
RUN cd /workspace/VLMEvalKit && \
    for patch in /workspace/patches/*.patch; do \
        if [ -f "$patch" ]; then \
            git apply "$patch" || echo "Warning: Patch $patch failed to apply"; \
        fi \
    done

# Copy transformer-specific requirements
COPY docker/requirements/transformers-4.33.in /workspace/requirements/transformers-4.33.in
COPY docker/requirements/transformers-4.33.lock /workspace/requirements/transformers-4.33.lock

# Stage 1: 잠금(Lock) - Install locked transformer 4.33 dependencies with hash verification
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps --require-hashes \
    -r /workspace/requirements/transformers-4.33.lock

# OpenCV Fix: salesforce-lavis requires opencv-python-headless==4.5.5.64 which is incompatible with NumPy 1.x
# Forcefully uninstall opencv-python-headless and install opencv-python 4.9.0 (compatible with NumPy 1.x)
RUN python -m pip uninstall -y opencv-python-headless && \
    python -m pip install --no-cache-dir opencv-python==4.9.0.80

# Stage 2: 검증(Verification) - Build-time smoke test for transformer 4.33
# IMPROVED: No network downloads, only import/version verification
# Copy verification script
COPY docker/scripts/verify_transformers.py /tmp/verify_transformers.py

RUN TRANSFORMERS_VERSION=4.33.0 python /tmp/verify_transformers.py && rm /tmp/verify_transformers.py

# Install VLMEvalKit with 4.33 compatibility (older version without Thyme)
RUN cd /workspace/VLMEvalKit && \
    python -m pip install --no-cache-dir -e .

# Verify VLMEvalKit version for reproducibility
RUN cd /workspace/VLMEvalKit && \
    git rev-parse HEAD && \
    git log -1 --oneline

# Install mPLUG-Owl2 from source (not available on PyPI)
RUN cd /workspace && \
    git clone https://github.com/X-PLUG/mPLUG-Owl.git && \
    cd mPLUG-Owl/mPLUG-Owl2 && \
    python -m pip install --no-cache-dir -e .

# CRITICAL FIX: Reinstall pydantic v2 after VLMEvalKit/mPLUG-Owl2 installation
# These packages may have downgraded pydantic, but deepspeed requires v2 for InstructBLIP
RUN python -m pip install --no-cache-dir --force-reinstall 'pydantic>=2.0'

# CRITICAL FIX: Configure InstructBLIP to use HuggingFace Vicuna model instead of local path
# Replace placeholder "Please set the path to your vicuna-7b-v1.1" with actual HF model ID
RUN sed -i 's|llm_model: "Please set the path to your vicuna-7b-v1.1"|llm_model: "lmsys/vicuna-7b-v1.1"|' \
    /workspace/VLMEvalKit/vlmeval/vlm/misc/blip2_instruct_vicuna7b.yaml && \
    sed -i 's|llm_model: "Please set the path to your vicuna-13b-v1.1"|llm_model: "lmsys/vicuna-13b-v1.1"|' \
    /workspace/VLMEvalKit/vlmeval/vlm/misc/blip2_instruct_vicuna13b.yaml

# Stage 3: 프로모션(Promotion) - Image ready for promotion pipeline
# This image has passed Lock + Verification stages
# Promotion pipeline tags:
#   - Build → ghcr.io/gwleee/spectravision:4.33-candidate
#   - After smoke tests pass → ghcr.io/gwleee/spectravision:4.33-stable
#   - After mini-bench pass → ghcr.io/gwleee/spectravision:4.33 (production)

# Configure environment for 4.33 models
ENV TRANSFORMERS_VERSION=4.33.0
ENV MODEL_CACHE_DIR=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch

# Create cache directories
RUN mkdir -p $MODEL_CACHE_DIR $TORCH_HOME

# Copy model-specific scripts and entrypoint
COPY scripts/ /workspace/scripts/
COPY docker/scripts/entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/scripts/*.sh 2>/dev/null || true && \
    chmod +x /workspace/entrypoint.sh

# CRITICAL FIX: Pre-download Qwen model and patch tokenizer to fix SimSun.ttf permission issue
# The Qwen tokenizer tries to write SimSun.ttf to the current directory during import,
# which fails when running containers with --user flag (read-only directory).
# Solution: Download model during build and patch tokenizer to write to /tmp instead.
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen-VL-Chat', trust_remote_code=True)" || true

# Apply Qwen tokenizer patch to fix SimSun.ttf write location
RUN find /root/.cache/huggingface/modules/transformers_modules -name "tokenization_qwen.py" -exec sed -i \
    's|open("SimSun.ttf", "wb").write(ttf.content)|import tempfile; font_path = os.path.join(tempfile.gettempdir(), "SimSun.ttf"); open(font_path, "wb").write(ttf.content)|' {} \; || \
    echo "Warning: Qwen tokenizer patch may not have been applied"

# Create outputs directory in VLMEvalKit and set permissions for spectravision user
# Also give spectravision ownership of VLMEvalKit directory and cache
RUN mkdir -p /workspace/VLMEvalKit/outputs && \
    chown -R spectravision:spectravision /workspace/VLMEvalKit && \
    chown -R spectravision:spectravision /root/.cache/huggingface || true

# Switch back to non-root user
USER spectravision

# Health check with transformer version verification
HEALTHCHECK --interval=30s --timeout=15s --start-period=10s --retries=3 \
    CMD python -c "import transformers as tr; assert tr.__version__ == '4.33.0'; import torch; print(f'Transformers 4.33 healthy')" || exit 1

# NOTE: No ENTRYPOINT defined to maintain compatibility with test scripts
# The entrypoint.sh script is available at /workspace/entrypoint.sh and can be used manually:
# docker run --gpus all IMAGE /workspace/entrypoint.sh --data BENCHMARK --model MODEL

# Default command shows shell for interactive use
CMD ["/bin/bash"]