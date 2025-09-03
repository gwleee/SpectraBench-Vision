# 🐳 SpectraBench-Vision Docker Usage Guide

## 🎯 System Overview

SpectraBench-Vision is a **Docker-based VLM evaluation system** designed for **reproducibility and ease of use**:

- **Multi-Container System**: Independent containers for each transformer version
- **DockerOrchestrator**: Intelligent automatic container management system
- **Docker-in-Docker Support**: Complete isolation with multi-version support

## 📦 System Architecture

### 🔧 Complete Docker Architecture

SpectraBench-Vision features a layered Docker architecture:

#### 🎯 1. Base Image (Common Foundation)
- **`spectravision-base:latest`** - Base image for all containers
- CUDA 11.8, Python 3, VLMEvalKit basic installation
- Common dependencies and SpectraVision codebase included

#### 🚀 2. Integrated System (Recommended)
- **`spectrabench-vision:latest`** - Docker-in-Docker integrated system
- Includes DockerOrchestrator for automatic container management
- User-friendly single entry point

#### 🔧 3. Individual Transformer Version Containers

| Image | Transformers | Main Models | Purpose |
|--------|-------------|-------------|---------|
| **spectravision-4.33** | 4.33.0 | Qwen-VL, VisualGLM | Legacy models |
| **spectravision-4.37** | 4.37.2 | InternVL2, LLaVA | Stable version |
| **spectravision-4.43** | 4.43.0 | Phi-3.5-Vision | Mid-range models |
| **spectravision-4.49** | 4.49.0 | SmolVLM, Qwen2.5-VL | Latest models |
| **spectravision-4.51** | 4.51.0 | Phi-4-Vision | Experimental |

#### 🏭 4. Production Environment
- **`docker-compose.prod.yml`** - Production environment orchestration
- Uses GitHub Container Registry images (`ghcr.io/gwleee/spectravision-*`)
- Scaling support and advanced resource management

## 📥 Image Downloads

### Available Images

| Image | Transformers | Key Models | Use Case | Download |
|-------|-------------|------------|----------|----------|
| **spectravision-base** | Base | Common foundation | Base image | `docker build -t spectravision-base:latest -f docker/base/Dockerfile .` |
| **spectrabench-vision** | Latest | Integrated system | Recommended | `docker build -t spectrabench-vision:latest -f docker/integrated/Dockerfile .` |
| **spectravision-4.33** | 4.33.0 | Qwen-VL, VisualGLM | Legacy models | `docker pull ghcr.io/gwleee/spectravision-4.33:latest` |
| **spectravision-4.37** | 4.37.2 | InternVL2, LLaVA | Stable version | `docker pull ghcr.io/gwleee/spectravision-4.37:latest` |
| **spectravision-4.43** | 4.43.0 | Phi-3.5-Vision | Mid-range models | `docker pull ghcr.io/gwleee/spectravision-4.43:latest` |
| **spectravision-4.49** | 4.49.0 | SmolVLM, Qwen2.5-VL | Latest models | `docker pull ghcr.io/gwleee/spectravision-4.49:latest` |
| **spectravision-4.51** | 4.51.0 | Phi-4-Vision | Experimental | `docker pull ghcr.io/gwleee/spectravision-4.51:latest` |

### Quick Download Methods (examples)
 
```bash
# Build integrated system (recommended)
docker build -t spectrabench-vision:latest -f docker/integrated/Dockerfile .

# Latest models (SmolVLM, Qwen2.5-VL)
docker pull ghcr.io/gwleee/spectravision-4.49:latest

# Stable versions (InternVL2, LLaVA)
docker pull ghcr.io/gwleee/spectravision-4.37:latest

# Build base image first
docker build -t spectravision-base:latest -f docker/base/Dockerfile .

# Download/build all images
./scripts/build_local_images.sh
```

## 🚀 Quick Start

### 🎯 Method 1: Using DockerOrchestrator (Recommended)

DockerOrchestrator is an **intelligent automatic container management system** that automatically selects and manages optimal transformer version containers for each model.

#### 📋 Available Modes
- **`--mode interactive`**: Interactive model/benchmark selection
- **`--mode batch`**: Direct command-line specification and execution
- **`--mode test`**: Test all container status and connections

#### 🎮 Interactive Mode (Recommended)
```bash
# 1. Start interactive mode - select all options via GUI
python3 docker/integrated/docker_main.py --mode interactive
```

**What you can do in interactive mode:**
- 📦 View model lists grouped by container
- 🎯 Select individual models or batch select by all/container/preset
- 📊 Choose from 24 benchmarks (all, korean, basic options)
- 🎮 Multi-GPU configuration and automatic distribution

#### ⚡ Batch Mode (Advanced Users)
```bash
# 2. Batch mode - direct command-line specification
python3 docker/integrated/docker_main.py --mode batch \
  --models "SmolVLM-256M" "InternVL2-2B" "Qwen2.5-VL-3B" \
  --benchmarks "MMBench" "TextVQA" "DocVQA" \
  --gpu-ids 0

# 3. Large model evaluation with multi-GPU
python3 docker/integrated/docker_main.py --mode batch \
  --models "Qwen2.5-VL-32B" "Qwen2.5-VL-72B" \
  --benchmarks "MMBench" "MMMU" \
  --gpu-ids 0 1 2 3
```

#### 🧪 System Test
```bash
# 4. Check entire system status
python3 docker/integrated/docker_main.py --mode test
```

**Test contents:**
- 🐳 Docker connection status check
- 🎮 GPU detection and configuration check
- 📦 All container image availability test
- 🚀 Container start/stop functionality test
- ⚙️ Basic operation check within each container

### Method 2: Direct Individual Container Usage
```bash
# Run latest model container
docker run --gpus all -it \
  -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/spectravision-4.49:latest

# Run evaluation inside container
cd /workspace/VLMEvalKit
python run.py --model SmolVLM-Instruct --data MMBench_DEV_EN --mode all
```

### Method 3: Docker Compose Usage (For Developers)
```bash
# Start specific version container
docker-compose -f docker/docker-compose.yml up -d transformers-4-49

# Access container for work
docker exec -it spectravision-transformers-4-49 /bin/bash
```

## ⚡ Multi-GPU Usage

```bash
# Multi-GPU usage with DockerOrchestrator
python3 docker/integrated/docker_main.py --mode batch \
  --models "Qwen2.5-VL-32B" "Qwen2.5-VL-72B" \
  --benchmarks "MMBench" "TextVQA" \
  --gpu-ids 0 1 2 3

# Specify GPUs for individual containers
docker run --gpus "device=0,1" -it \
  -e NVIDIA_VISIBLE_DEVICES=0,1 \
  -v $(pwd)/outputs:/workspace/outputs \
  ghcr.io/gwleee/spectravision-4.49:latest
```

## 🛠️ Advanced Configuration

### 🏭 Production Docker Compose Usage
```bash
# Production environment - Start all containers
docker-compose -f docker/docker-compose.prod.yml --profile all up -d

# Start specific profile only (latest models only)
docker-compose -f docker/docker-compose.prod.yml --profile latest up -d

# Multi-GPU configuration
GPU_COUNT=4 NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  docker-compose -f docker/docker-compose.prod.yml --profile all up -d

# Scaling support (start 3 instances of transformers-4-37)
docker-compose -f docker/docker-compose.prod.yml --profile stable up -d --scale transformers-4-37=3

# Download all production images
docker-compose -f docker/docker-compose.prod.yml pull
```

### 🔧 Development Docker Compose Usage
```bash
# Development environment - Start specific versions only
docker-compose -f docker/docker-compose.yml up -d transformers-4-49

# Direct container usage
docker run --gpus all -it ghcr.io/gwleee/spectravision-4.49:latest
```

### 🛠️ Building from Base Image
```bash
# Build base image
docker build -t spectravision-base:latest -f docker/base/Dockerfile .

# Build specific transformer version (requires base image)
docker build -t spectravision-4.49:latest -f docker/transformers-4.49/Dockerfile .

# Build integrated system
docker build -t spectrabench-vision:latest -f docker/integrated/Dockerfile .
```

## 🛠️ Troubleshooting

### Common Issues

**Docker daemon connection error**
```bash
sudo systemctl restart docker
docker version
```

**GPU not recognized**
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8-runtime-ubuntu22.04 nvidia-smi
```

**Memory insufficient**
```bash
docker system prune -f
docker volume prune -f
```

## 🎉 System Verification

### System Test
```bash
# 1. DockerOrchestrator system test
python3 docker/integrated/docker_main.py --mode test

# 2. Individual container basic functionality check
docker run --rm --gpus all \
  ghcr.io/gwleee/spectravision-4.49:latest \
  python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU Count: {torch.cuda.device_count()}')
print('SpectraVision container ready!')
"

# 3. Quick evaluation test (SmolVLM + MMBench)
python3 docker/integrated/docker_main.py --mode batch \
  --models "SmolVLM-256M" \
  --benchmarks "MMBench" \
  --gpu-ids 0
```

### Expected Results
```
SpectraBench-Vision Docker System Test
==================================================
Testing Docker connectivity...
SUCCESS: Docker client initialized successfully
GPU Configuration: 1 GPU(s) detected

Testing 5 container images...

Testing transformers_4_49...
SUCCESS: Image available
   Starting container...
   SUCCESS: Container started
   SUCCESS: Basic functionality test passed
   SUCCESS: GPU test: CUDA available: True
   SUCCESS: Container stopped

SUCCESS: All system tests passed! System is ready for evaluation.
```

---

## 🎊 Success! SpectraBench-Vision System Ready

**🎯 Congratulations!** SpectraBench-Vision Docker system has been successfully set up!

### ✨ What you can do now:
- 🚀 **Automated Container Management**: DockerOrchestrator automatically selects optimal containers per model
- 🤖 **30 Model Support**: Evaluate all models across 5 transformer versions
- 📈 **Scalability**: Easy addition of new models/versions
- 🔧 **Reproducibility**: Identical evaluation environment anywhere

### 🎯 Next Steps:
```bash
# Start powerful VLM evaluation with DockerOrchestrator!
python3 docker/integrated/docker_main.py --mode interactive
```

**Happy Evaluating! 🚀✨**

