# SpectraVision Production Deployment Guide

## 🚀 Quick Production Setup

### Method 1: Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/gwleee/SpectraBench-Vision.git
cd SpectraBench-Vision

# Configure environment
cp .env.template .env
# Edit .env file: Add your HF_TOKEN and configure GPU settings

# Start production services
docker-compose -f docker/docker-compose.prod.yml --profile stable up -d

# Monitor logs
docker-compose -f docker/docker-compose.prod.yml logs -f

# Run evaluation
docker exec -it $(docker-compose -f docker/docker-compose.prod.yml ps -q transformers-4-37 | head -1) python scripts/main.py --models "InternVL2-8B" --benchmarks "MMBench"
```

### Method 2: Automatic Multi-Version CLI (Recommended for Production)

```bash
# Clone repository
git clone https://github.com/gwleee/SpectraBench-Vision.git
cd SpectraBench-Vision

# Configure environment
cp .env.template .env
# Edit .env: Add HF_TOKEN

# Automatic multi-version evaluation (zero configuration)
python scripts/main.py --multi-version --models "SmolVLM-1.7B,Qwen2.5-VL-7B,InternVL2-8B" --benchmarks "MMBench,TextVQA"

# System automatically:
# 1. Detects required transformer versions for each model
# 2. Downloads appropriate Docker images from ghcr.io 
# 3. Allocates GPUs and runs evaluations
# 4. Integrates results across all versions
```

### Method 3: All-in-One Container

```bash
# Start integrated container
docker run -it --gpus all -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/spectrabench-vision:latest

# Inside container, run evaluation
python scripts/main.py --models "SmolVLM-1.7B" --benchmarks "TextVQA"

# Or run specific evaluation directly
cd /workspace/VLMEvalKit
python run.py --model SmolVLM --data TextVQA --mode all
```

## 📦 Local Image Building

For offline environments or custom modifications:

```bash
# Build all transformer version images
./scripts/build_local_images.sh

# Build specific version
docker build -f docker/transformers-4.49/Dockerfile -t spectravision-4.49:local .

# Build from base
docker build -f docker/base/Dockerfile -t spectravision-base:local .
```

## 🔧 Production Configuration

### GPU Configuration

```bash
# Single GPU
NVIDIA_VISIBLE_DEVICES=0 docker-compose -f docker/docker-compose.prod.yml --profile stable up -d

# Multi-GPU (A100 dual)
GPU_COUNT=2 NVIDIA_VISIBLE_DEVICES=0,1 docker-compose -f docker/docker-compose.prod.yml --profile stable up -d

# High-density cluster (16 GPUs)
GPU_COUNT=16 NVIDIA_VISIBLE_DEVICES=all docker-compose -f docker/docker-compose.prod.yml --profile all up -d
```

### Scaling Services

```bash
# Scale transformers-4-37 to 3 instances
docker-compose -f docker/docker-compose.prod.yml --profile stable up -d --scale transformers-4-37=3

# Run different models simultaneously
docker-compose -f docker/docker-compose.prod.yml --profile all up -d
```

## 📊 Verified Model-Benchmark Combinations

### Fully Tested Combinations (✅)
| Model | MMBench | TextVQA | VQAv2 | GQA | SEED | Status |
|-------|---------|---------|-------|-----|------|--------|
| SmolVLM-1.7B | ✅ | ✅ | ✅ | ✅ | ✅ | Stable |
| InternVL2-8B | ✅ | ✅ | ✅ | ✅ | ✅ | Stable |
| Qwen2.5-VL-7B | ✅ | ✅ | ✅ | ✅ | ✅ | Stable |
| LLaVA-1.5-13B | ✅ | ✅ | ✅ | ✅ | ⚠️ | High Memory |

### Known Issues (⚠️)
- **Large models (>32B)**: Require multi-GPU setup (GPU_COUNT>=2)
- **Phi-4-Vision**: May need additional memory (45GB+)
- **Some benchmarks**: Require specific prompt formats

## 💻 System Requirements

### Minimum Requirements
- **GPU**: NVIDIA RTX 4080 (16GB) or better
- **RAM**: 32GB system memory
- **Storage**: 100GB available space (models + cache)
- **Docker**: 20.10+ with nvidia-container-toolkit

### Recommended Production Setup
- **GPU**: NVIDIA A100 80GB or H100
- **RAM**: 128GB system memory  
- **Storage**: 500GB NVMe SSD
- **Network**: High bandwidth for model downloads

## 📈 Performance Optimization

### Model Caching
```bash
# Pre-download models to cache
docker run --gpus all -v model_cache:/root/.cache/huggingface \
  ghcr.io/gwleee/spectravision-4.49:latest \
  python -c "from transformers import AutoModel; AutoModel.from_pretrained('HuggingFaceM4/SmolVLM-Instruct')"
```

### Batch Processing
```bash
# Run multiple evaluations in sequence
python scripts/main.py --models "SmolVLM-1.7B,InternVL2-8B,Qwen2.5-VL-7B" --benchmarks "MMBench,TextVQA,VQAv2"
```

## 🔍 Monitoring and Maintenance

### Health Checks
```bash
# Check container health
docker-compose -f docker/docker-compose.prod.yml ps

# View resource usage
docker stats

# Check GPU utilization
nvidia-smi
```

### Log Management
```bash
# View logs with timestamps
docker-compose -f docker/docker-compose.prod.yml logs -f --timestamps

# Save logs to file
docker-compose -f docker/docker-compose.prod.yml logs > evaluation_logs.txt
```

## 🔒 Security Considerations

- Use non-root user in production containers
- Limit network access for containers
- Regularly update base images for security patches
- Store HF_TOKEN securely (avoid committing to git)

## 📚 Additional Resources

- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Model Compatibility Matrix](COMPATIBILITY.md)
- [Docker Usage Guide](DOCKER_USAGE_GUIDE_EN.md)