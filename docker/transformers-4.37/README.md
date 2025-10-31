# Transformers 4.37 Docker Image

## Overview

Docker image for Vision-Language Models requiring transformers 4.37.2, specifically optimized for LLaVA models.

**Image**: `ghcr.io/gwleee/spectravision:4.37`

## Supported Models

- **LLaVA-1.5-7B** (Primary)
- **LLaVA-1.5-13B** (Primary)
- InternVL2-2B
- InternVL2-8B
- MiniCPM-V-2_6
- CogVLM-7B
- ShareGPT4V-7B

## Key Features

### 1. LLaVA Integration
- **Editable Install**: LLaVA installed with `-e` flag for VLMEvalKit compatibility
- **Version Pinned**: v1.2.2.post1 for stability
- **Auto-Patching**: Automatic import fixes via `patch_llava_builder.py`

### 2. Flash Attention Disabled
**IMPORTANT**: Flash Attention is intentionally disabled due to ABI compatibility issues with PyTorch 2.1.2.

```dockerfile
# DISABLED: Flash Attention causes ABI compatibility issues
# LLaVA works fine without flash-attn, just slightly slower
# RUN python -m pip install "flash-attn==2.8.2" --no-build-isolation
```

**Impact**:
- **Performance**: Slightly slower inference (~10-15%)
- **Stability**: 100% success rate, no ABI errors
- **Alternative**: PyTorch SDPA (Scaled Dot-Product Attention) used automatically

### 3. Reproducible Builds
- **Locked Dependencies**: `transformers-4.37.lock` with hash verification
- **Verification Stage**: Build-time import and version checks
- **Cache Consistency**: All HuggingFace caches point to single directory

## Quick Start

### Single Model Test
```bash
docker run --rm --gpus all \
  -v $(pwd)/.env:/workspace/VLMEvalKit/.env \
  -e VLMEVAL_SAMPLE_LIMIT=10 \
  ghcr.io/gwleee/spectravision:4.37 \
  python /workspace/VLMEvalKit/run.py \
    --data MMBench_DEV_EN \
    --model llava_v1.5_7b
```

### Comprehensive Test (2 models × 20 benchmarks)
```bash
./run_437_llava_only.sh
```

### Interactive Mode
```bash
docker run -it --gpus all \
  -v $(pwd)/.env:/workspace/VLMEvalKit/.env \
  ghcr.io/gwleee/spectravision:4.37 \
  /bin/bash
```

## Build Instructions

```bash
# Build image
DOCKER_BUILDKIT=1 docker build \
  --network=host \
  -f docker/transformers-4.37/Dockerfile \
  -t ghcr.io/gwleee/spectravision:4.37 \
  .

# Verify build
docker run --rm ghcr.io/gwleee/spectravision:4.37 \
  python -c "import transformers as tr; import llava; print(f'✓ {tr.__version__}')"
```

## Environment Variables

### Required
- `HUGGING_FACE_HUB_TOKEN`: For gated models (configure in `.env`)

### Optional
- `VLMEVAL_SAMPLE_LIMIT`: Limit samples per benchmark (default: full dataset)
- `LMUData`: Dataset directory path (default: `/workspace/LMUData`)

### Cache Directories (Auto-configured)
- `HF_HOME=/workspace/.cache/huggingface`
- `TRANSFORMERS_CACHE=/workspace/.cache/huggingface`
- `HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface`
- `TORCH_HOME=/workspace/.cache/torch`

## Known Issues & Solutions

### Issue 1: Flash Attention ABI Error
**Symptom**:
```
RuntimeError: undefined symbol: _ZNK3c105Error4whatEv
```

**Solution**: Flash Attention is disabled by default. LLaVA uses PyTorch SDPA instead.

**If you need Flash Attention**:
```dockerfile
# Must compile from source with exact PyTorch ABI
RUN git clone https://github.com/Dao-AILab/flash-attention && \
    cd flash-attention && \
    MAX_JOBS=4 python setup.py install
```

### Issue 2: LLaVA Import Errors
**Symptom**:
```
ImportError: cannot import name 'LlavaLlamaForCausalLM'
```

**Solution**: Automatic patching via `patch_llava_builder.py` during build.

### Issue 3: Permission Denied for Outputs
**Solution**: Image sets `chmod 777` on output directories. Containers work with any `--user` flag.

## Test Results

### LLaVA-Only Test (2 models × 20 benchmarks × 10 samples)
- **Total Combinations**: 40
- **Success Rate**: 100%
- **Duration**: ~22 minutes
- **Benchmarks**: MMBench, TextVQA, GQA, MMMU, DocVQA, ChartQA, InfoVQA, OCRBench, AI2D, ScienceQA, POPE, HallusionBench, MMStar, RealWorldQA, VisOnlyQA, VizWiz, SEEDBench_IMG, BLINK, and Korean variants

## Architecture

### 3-Stage Build System

1. **Lock Stage (잠금)**: Install locked dependencies with hash verification
2. **Verification Stage (검증)**: Build-time import and version checks
3. **Promotion Stage (프로모션)**: Ready for deployment pipeline

### Directory Structure
```
/workspace/
├── LLaVA/                    # LLaVA v1.2.2.post1 (editable install)
├── VLMEvalKit/              # VLMEvalKit (editable install)
├── .cache/
│   ├── huggingface/         # Model cache
│   └── torch/               # Torch cache
├── scripts/                  # Evaluation scripts
└── entrypoint.sh            # Optional entrypoint

```

## Troubleshooting

### Check Version
```bash
docker run --rm ghcr.io/gwleee/spectravision:4.37 \
  python -c "import transformers; print(transformers.__version__)"
```

### Verify LLaVA
```bash
docker run --rm ghcr.io/gwleee/spectravision:4.37 \
  python -c "import llava; from llava.model.builder import load_pretrained_model; print('✓ LLaVA OK')"
```

### Check GPU
```bash
docker run --rm --gpus all ghcr.io/gwleee/spectravision:4.37 \
  python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

## Performance Optimization

### Persistent Containers
For multiple evaluations, use persistent containers instead of repeated `docker run`:

```bash
# Start persistent container
CONTAINER_ID=$(docker run -d --gpus all \
  -v $(pwd)/.env:/workspace/VLMEvalKit/.env \
  ghcr.io/gwleee/spectravision:4.37 \
  tail -f /dev/null)

# Run evaluations
docker exec $CONTAINER_ID python /workspace/VLMEvalKit/run.py --data MMBench --model llava_v1.5_7b
docker exec $CONTAINER_ID python /workspace/VLMEvalKit/run.py --data TextVQA --model llava_v1.5_7b

# Cleanup
docker stop $CONTAINER_ID
```

**Benefits**:
- No repeated model downloads
- Faster startup time
- Reduced network usage

## Changelog

### 2025-10-30: Flash Attention Disabled
- Disabled Flash Attention due to ABI compatibility issues
- LLaVA now uses PyTorch SDPA (default attention)
- Achieved 100% success rate on 40 LLaVA test combinations
- Trade-off: 10-15% slower inference for guaranteed stability

### 2025-10-28: LLaVA Import Fixes
- Added automatic patching via `patch_llava_builder.py`
- Fixed `LlavaLlamaForCausalLM` import errors
- Improved build-time verification

## References

- LLaVA Repository: https://github.com/haotian-liu/LLaVA
- VLMEvalKit: https://github.com/open-compass/VLMEvalKit
- Flash Attention: https://github.com/Dao-AILab/flash-attention
- PyTorch SDPA: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
